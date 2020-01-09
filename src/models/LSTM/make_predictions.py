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
    COLUMNS_TO_DROP_FOR_TRAINING

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# MinMaxScalar used in training
mms = _load_model_from_pickle('src/models/pickles/min_max_scalar_lstm_v3.pickle')
COLUMNS_TO_SCALE = _load_model_from_pickle('src/models/pickles/min_max_scalar_columns_v3.pickle')


def _load_input_data(previous_gw, save_file=False):
    """
    Load all input data required for predictions: historical, current season, next fixture. Also creates any additional
    features

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


def _load_model_from_h5(model_filepath):
    model = load_model(model_filepath)
    return model


full_data = _load_input_data(previous_gw=21, save_file=True)

lstm_model = _load_model_from_h5("src/models/pickles/v3_lstm_model.h5")

previous_gw = 21
prediction_season_order = 4
N_STEPS_IN = 5

available_players = full_data.copy()[
    (full_data['gw'] == previous_gw) &
    (full_data['season_order'] == prediction_season_order)
][['name']].reset_index(drop=True)

assert available_players['name'].nunique() == len(available_players), 'Duplicate names found in players from last GW'
available_players['available_for_selection'] = 1

full_data = full_data.merge(available_players, how='left', on='name')
full_data['available_for_selection'].sum()

gw_prediction_data = full_data.copy()[full_data['available_for_selection'] == 1]
gw_prediction_data.drop('available_for_selection', axis=1, inplace=True)

print(gw_prediction_data.shape)

# Drop players if they don't have enough GW data to be used by configured LSTM
gw_prediction_data['total_number_of_gameweeks'] = gw_prediction_data.groupby(['name']).transform('count')['team_name']
gw_prediction_data = gw_prediction_data[
    gw_prediction_data['total_number_of_gameweeks'] >= N_STEPS_IN + 1  # Add 1 because current GW will be dropped
]
gw_prediction_data.drop('total_number_of_gameweeks', axis=1, inplace=True)

print(gw_prediction_data.shape)

# Make player predictions

final_predictions = pd.DataFrame()

for player in gw_prediction_data['name'].unique():

    player_df = gw_prediction_data[gw_prediction_data['name'] == player]

    # Drop GW you are predicting for
    player_df = player_df[~(
            (player_df['gw'] == previous_gw + 1) & (player_df['season_order'] == prediction_season_order)
    )]

    player_df = player_df.tail(N_STEPS_IN)

    player_df[COLUMNS_TO_SCALE] = mms.transform(player_df[COLUMNS_TO_SCALE])

    player_df.drop(
        COLUMNS_TO_DROP_FOR_TRAINING,
        axis=1,
        inplace=True
    )

    X_player_df = player_df.values.reshape(1, N_STEPS_IN, player_df.shape[1])

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
    (gw_prediction_data['gw'] == previous_gw) &
    (gw_prediction_data['season_order'] == prediction_season_order)
][[
    'name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID', 'team_name', 'next_match_value'
]]

assert other_player_info.shape[0] == final_predictions.shape[0]

final_predictions = final_predictions.merge(other_player_info, on='name')

assert other_player_info.shape[0] == final_predictions.shape[0]

final_predictions['sum'] = final_predictions['GW_plus_1'] + \
                           final_predictions['GW_plus_2'] + \
                           final_predictions['GW_plus_3'] + \
                           final_predictions['GW_plus_4'] + \
                           final_predictions['GW_plus_5']

final_predictions.sort_values('sum', ascending=False, inplace=True)

final_predictions.to_parquet('data/gw_predictions/gw22_v3_lstm_player_predictions.parquet', index=False)
