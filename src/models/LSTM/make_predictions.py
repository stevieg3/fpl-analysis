import logging
import pandas as pd

from keras.models import load_model

from src.models.utils import \
    _load_all_historical_data, \
    _map_season_string_to_ordered_numeric, \
    _generate_known_features_for_next_gw, \
    _load_current_season_data, \
    _load_next_fixture_data, \
    _load_model_from_pickle
from src.models.constants import \
    COLUMNS_TO_DROP_FOR_TRAINING, \
    KICKOFF_MONTH_FEATURES

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# MinMaxScalar used in training
mms = _load_model_from_pickle('src/models/pickles/min_max_scalar_lstm_v4.pickle')
COLUMNS_TO_SCALE = _load_model_from_pickle('src/models/pickles/min_max_scalar_columns_v4.pickle')

lstm_model = load_model("src/models/pickles/v4_lstm_model.h5")


def load_live_data(previous_gw, save_file=False):
    """
    Load all live input data required for predictions: historical, current season, next fixture. Also creates any
    additional features.

    :param previous_gw: Previous gameweek
    :param save_file: Set to True to save current season data as parquet
    :return: DataFrame containing all required inputs for all gameweeks
    """
    # Load historical
    raw_historical_df = _load_all_historical_data()
    # Remove Brendan Galloway due to unexplained gap in gameweek data
    raw_historical_df = raw_historical_df[raw_historical_df['name'] != 'brendan_galloway']
    logging.info(f"Loaded historical data of shape: {raw_historical_df.shape}")

    # Load current
    logging.info(f"Loading current season data (up to gw {previous_gw})...")
    current_gw_df = _load_current_season_data(previous_gw=previous_gw, save_file=save_file)
    logging.info(f"Loaded current season data (up to gw {previous_gw}) of shape: {current_gw_df.shape}")

    assert len(set(current_gw_df.columns) - set(raw_historical_df.columns)) == 0

    # Load next
    logging.info(f"Loading next fixture (gw {previous_gw + 1}) data...")
    next_gw_df = _load_next_fixture_data(next_gw=previous_gw+1)
    logging.info(f"Loaded next fixture (gw {previous_gw+1}) data of shape: {next_gw_df.shape}")

    # Combine data
    input_data = raw_historical_df.append(current_gw_df, sort=False)
    input_data = input_data.append(next_gw_df, sort=False)
    logging.info(f"Combined data shape: {input_data.shape}")

    # Additional features
    _map_season_string_to_ordered_numeric(input_data)

    input_data.sort_values(['name', 'season_order', 'gw'], inplace=True)
    input_data.reset_index(drop=True, inplace=True)

    _generate_known_features_for_next_gw(input_data)

    input_data.drop('ID', axis=1, inplace=True)

    logging.info(f"Final input shape: {input_data.shape}")

    return input_data


def load_retro_data(current_season_data_filepath):
    """
    Load all retro input data required for predictions: historical, current season, next fixture. Also creates any
    additional features.

    Can only make predictions for up to 1 GW before the last GW in loaded data e.g. if you load current season data up
    to GW 29 of the current season you can only do previous_gw up to GW 28 of the current season. This is because next
    GW features need to be generated.

    :param current_season_data_filepath: File path of latest current season data.
    :return: DataFrame containing all required inputs for all gameweeks
    """
    # Load historical
    raw_historical_df = _load_all_historical_data()
    # Remove Brendan Galloway due to unexplained gap in gameweek data
    # TODO: Fix in generation of historical data:
    raw_historical_df = raw_historical_df[raw_historical_df['name'] != 'brendan_galloway']
    logging.info(f"Loaded historical data of shape: {raw_historical_df.shape}")

    # Load latest current season data
    current_season_data = pd.read_parquet(current_season_data_filepath)

    for col in list(set(KICKOFF_MONTH_FEATURES) - set(current_season_data.columns)):
        current_season_data[col] = 0

    input_data = raw_historical_df.append(current_season_data, sort=False)

    # Additional features
    _map_season_string_to_ordered_numeric(input_data)

    input_data.sort_values(['name', 'season_order', 'gw'], inplace=True)
    input_data.reset_index(drop=True, inplace=True)

    _generate_known_features_for_next_gw(input_data)

    input_data.drop('ID', axis=1, inplace=True)

    logging.info(f"Final input shape: {input_data.shape}")

    return input_data


class LSTMPlayerPredictor:
    """
    Class for making player points predictions for the next 5 gameweeks using a trained LSTM model.

    :param previous_gw: The previous gameweek to the one you want to predict for
    :param prediction_season_order: Season order number for previous_gw
    :param N_STEPS_IN: Number of steps for LSTM input
    :param previous_gw_was_double_gw: Boolean. True if previous gameweek was a double gameweek
    """
    def __init__(self, previous_gw, prediction_season_order, N_STEPS_IN=5, previous_gw_was_double_gw=False):
        self.previous_gw = previous_gw
        self.prediction_season_order = prediction_season_order
        self.N_STEPS_IN = N_STEPS_IN
        self.previous_gw_was_double_gw = previous_gw_was_double_gw

    def prepare_data_for_lstm(self, full_data):
        """
        Prepares data set so that it can be fed into LSTM model for predictions. Does the following steps:
        - Removes any data after, and not including, specified `previous_gw` in `prediction_season_order`
        - Filters out any players not available for selection (i.e. cannot make predictions for)
        - Removes any players with insufficient historical GW data needed as an input to LSTM

        :param full_data: DataFrame of player season-gameweek data as returned by `load_live_data` or `load_retro_data`
        :return: DataFrame of player season-gameweek data to be inputted into LSTM
        """
        full_data = full_data.copy()

        # Remove tail data not needed
        cutoff = int(
            str(self.prediction_season_order) +
            "{0:0=2d}".format(self.previous_gw)
        )

        full_data['season_order_gw'] = (
            full_data['season_order'].astype(str) +
            full_data['gw'].apply(lambda x: "{0:0=2d}".format(x))
        ).apply(int)

        full_data = full_data.copy()[full_data['season_order_gw'] <= cutoff]
        full_data.drop('season_order_gw', axis=1, inplace=True)

        # Get available players
        # TODO Include logic for dealing with double gameweeks and missing fixtures
        available_players = full_data.copy()[
            (full_data['gw'] == self.previous_gw) &
            (full_data['season_order'] == self.prediction_season_order)
        ][['name']].reset_index(drop=True)

        if self.previous_gw_was_double_gw:
            # Double GW players will be listed twice
            available_players.drop_duplicates(inplace=True)
        assert available_players['name'].nunique() == len(available_players), \
            'Duplicate names found in players from last GW'
        available_players['available_for_selection'] = 1

        full_data = full_data.merge(available_players, how='left', on='name')

        logging.info(f"Number of players available for selection: {full_data['available_for_selection'].sum()}")

        gw_prediction_data = full_data.copy()[full_data['available_for_selection'] == 1]
        gw_prediction_data.drop('available_for_selection', axis=1, inplace=True)

        logging.info(f"Player data shape before: {gw_prediction_data.shape}")

        # Drop players if they don't have enough GW data to be used by configured LSTM
        gw_prediction_data['total_number_of_gameweeks'] = \
            gw_prediction_data.groupby(['name']).transform('count')['team_name']

        gw_prediction_data = gw_prediction_data[
            gw_prediction_data['total_number_of_gameweeks'] >= self.N_STEPS_IN
        ]
        gw_prediction_data.drop('total_number_of_gameweeks', axis=1, inplace=True)

        logging.info(f"Player data shape after removing players with insufficient GW data: {gw_prediction_data.shape}")

        return gw_prediction_data

    def make_player_predictions(self, gw_prediction_data):
        # TODO Add functionality to adjust scores of certain players e.g. by using a JSON input
        """
        Make player predictions for next 5 gameweeks using LSTM model.

        :param gw_prediction_data: LSTM input DataFrame as returned by `prepare_data_for_lstm`
        :return: DataFrame containing individual player predictions sorted by total points predicted.
        """
        final_predictions = pd.DataFrame()

        for player in gw_prediction_data['name'].unique():

            player_df = gw_prediction_data[gw_prediction_data['name'] == player]

            player_df = player_df.tail(self.N_STEPS_IN)

            player_df[COLUMNS_TO_SCALE] = mms.transform(player_df[COLUMNS_TO_SCALE])

            player_df.drop(
                COLUMNS_TO_DROP_FOR_TRAINING,
                axis=1,
                inplace=True
            )

            X_player_df = player_df.values.reshape(1, self.N_STEPS_IN, player_df.shape[1])

            predictions = lstm_model.predict(X_player_df).reshape(5)

            prediction_df = pd.DataFrame(
                {
                    'name': [player],
                    'GW_plus_1': [predictions[0]],
                    'GW_plus_2': [predictions[1]],
                    'GW_plus_3': [predictions[2]],
                    'GW_plus_4': [predictions[3]],
                    'GW_plus_5': [predictions[4]]
                }
            )

            final_predictions = final_predictions.append(prediction_df)

        other_player_info = gw_prediction_data.copy()[
            (gw_prediction_data['gw'] == self.previous_gw) &
            (gw_prediction_data['season_order'] == self.prediction_season_order)
        ][[
            'name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID', 'team_name', 'next_match_value'
        ]]

        if self.previous_gw_was_double_gw:
            # Keep most recent for latest price
            other_player_info.drop_duplicates(subset='name', keep='last', inplace=True)

        assert other_player_info.shape[0] == final_predictions.shape[0]

        final_predictions = final_predictions.merge(other_player_info, on='name')

        assert other_player_info.shape[0] == final_predictions.shape[0]  # TODO Are both assert statements needed?

        final_predictions['sum'] = final_predictions['GW_plus_1'] + \
            final_predictions['GW_plus_2'] + \
            final_predictions['GW_plus_3'] + \
            final_predictions['GW_plus_4'] + \
            final_predictions['GW_plus_5']

        final_predictions.sort_values('sum', ascending=False, inplace=True)

        return final_predictions


# full_data = load_live_data(previous_gw=28, save_file=True)  # TODO Live file load

# final_predictions.to_parquet('data/gw_predictions/gw29_v4_lstm_player_predictions.parquet', index=False)
