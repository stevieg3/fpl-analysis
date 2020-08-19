import logging

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from keras.models import load_model

from src.models.utils import _load_model_from_pickle
from src.models.constants import SEASON_ORDER_DICT
from src.data.s3_utilities import s3_filesystem

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

mms = _load_model_from_pickle('src/models/pickles/min_max_scalar_DeepFantasyFootball_v01.pickle')
"""
MinMaxScalar used in training
"""

deep_fantasy_football_model = load_model("src/models/pickles/DeepFantasyFootball_v01.h5")
"""
DeepFantasyFootball fitted model
"""

FINAL_FEATURES = [
    'Aerial Duels - Won - Percentage',
    'Assists',
    'Bad Touches',
    'Big Chances Created',
    'Caught Offside',
    'Chances From Counter Attack',
    'Clean Sheets',
    'Crosses - Open Play - Successful',
    'Crosses - Unsuccessful',
    'Distribution - Successful',
    'Dribbles - Successful Percentage',
    'Fouls',
    'Goals',
    'Goals Conceded',
    'Handballs',
    'ICT Creativity',
    'ICT Index',
    'Minutes Per Block',
    'Minutes Per Interception',
    'Minutes Per Save',
    'Minutes Per Tackle Won',
    'Minutes Per Touch',
    'xGI Expected Goal Involvement',
    'Pass Completion',
    'Pass Completion - Final Third',
    'Pass Completion - Opponents Half',
    'Passes - Backward',
    'Passes - Forward',
    'Premier League Straight Red Cards',
    'Premier League Total Red Cards',
    'Recoveries',
    'Saves (Shots Outside Box)',
    'Shot Accuracy',
    'Shots Blocked',
    'Shots On Target',
    'Subbed Off',
    'Subbed On',
    'Tackles - Won - Percentage',
    'Tackles Lost',
    'Take Ons',
    'Take Ons - Successful Percentage',
    'Throw Ins',
    'Time Played',
    'Touches - Final Third',
    'Touches - Penalty Area',
    'double_gameweek',
    'gw',
    'next_gameweek_double_gameweek',
    'next_gameweek_draw_odds',
    'next_gameweek_lose_odds',
    'next_gameweek_number_of_home_matches',
    'next_gameweek_number_of_promoted_side_opponent',
    'next_gameweek_number_of_top_6_last_season_opponent',
    'next_gameweek_win_odds',
    'number_of_home_matches',
    'number_of_top_6_last_season_opponent',
    'position_DEF',
    'position_FWD',
    'position_GK',
    'position_MID',
    'top_6_last_season',
    'total_points'
]

ffs_abbreviation_to_full = {
    'WHU': 'West Ham United',
    'BUR': 'Burnley',
    'HUD': 'Huddersfield Town',
    'ARS': 'Arsenal',
    'CRY': 'Crystal Palace',
    'WAT': 'Watford',
    'FUL': 'Fulham',
    'LIV': 'Liverpool',
    'BOU': 'Bournemouth',
    'WOL': 'Wolverhampton Wanderers',
    'EVE': 'Everton',
    'LEI': 'Leicester City',
    'WBA': 'West Bromwich Albion',
    'NEW': 'Newcastle United',
    'SOU': 'Southampton',
    'MUN': 'Manchester United',
    'SWA': 'Swansea City',
    'BHA': 'Brighton and Hove Albion',
    'CHE': 'Chelsea',
    'CAR': 'Cardiff City',
    'MCI': 'Manchester City',
    'TOT': 'Tottenham Hotspur',
    'STK': 'Stoke City',
    'AVL': 'Aston Villa',
    'BLA': 'Blackburn Rovers',
    'BOL': 'Bolton Wanderers',
    'HUL': 'Hull City',
    'MID': 'Middlesbrough',
    'NOR': 'Norwich City',
    'QPR': 'Queens Park Rangers',
    'RDG': 'Reading',
    'SHU': 'Sheffield United',
    'SUN': 'Sunderland',
    'WIG': 'Wigan Athletic'
}


def load_live_data(previous_gw, save_file=False):
    """
    Load all live input data required for predictions: historical, current season, next fixture. Also creates any
    additional features.

    :param previous_gw: Previous gameweek
    :param save_file: Set to True to save current season data as parquet
    :return: DataFrame containing all required inputs for all gameweeks
    """

    pass


def load_retro_data():
    """
    Load all retro input data required for predictions: historical, current season, next fixture. Also creates any
    additional features.

    Can only make predictions for up to 1 GW before the last GW in loaded data e.g. if you load current season data up
    to GW 29 of the current season you can only do previous_gw up to GW 28 of the current season. This is because next
    GW features need to be generated.

    :param current_season_data_filepath: File path of latest current season data.
    :return: DataFrame containing all required inputs for all gameweeks
    """

    ffs_all_data = pq.read_table(
        f"s3://fantasy-football-scout/processed/fantasy_football_scout_final_features_and_total_points.parquet",
        filesystem=s3_filesystem
    ).to_pandas()

    ffs_all_data.drop(columns=['Name', 'full_name'], inplace=True)

    # Add 0 minute events back into data
    ffs_data_names = ffs_all_data[['name', 'Team', 'position', 'season']].drop_duplicates()
    ffs_data_names['key'] = 1

    gw_df = pd.DataFrame({'gw': range(1, 39)})
    gw_df['key'] = 1

    all_player_season_gw_df = gw_df.merge(ffs_data_names, on='key')
    all_player_season_gw_df.drop('key', axis=1, inplace=True)

    ffs_data = all_player_season_gw_df.merge(ffs_all_data, on=['name', 'Team', 'position', 'gw', 'season'], how='left')
    ffs_data.fillna(0, inplace=True)

    # Position dummies
    ffs_data = pd.get_dummies(ffs_data, columns=['position'])
    ffs_data.rename(columns={'Team': 'team_name'}, inplace=True)

    # Merge fixture and odds data
    fixture_and_odds_features = pd.read_parquet(
        'data/processed/formatted_fixture_and_odds_features_2011_to_2020.parquet'
    )
    assert len(
        set(ffs_data['team_name'].replace(ffs_abbreviation_to_full)) - set(fixture_and_odds_features['team_name'])
    ) == 0

    assert len(
        set(fixture_and_odds_features['team_name']) - set(ffs_data['team_name'].replace(ffs_abbreviation_to_full))
    ) == 0

    ffs_data['team_name'].replace(ffs_abbreviation_to_full, inplace=True)

    # Combine feature and odds features
    ffs_data = ffs_data.merge(
        fixture_and_odds_features,
        on=['season', 'gw', 'team_name'],
        how='inner'
    )

    ffs_data['season'] = ffs_data['season'].str.replace('-20', '-')
    ffs_data['season_order'] = ffs_data['season'].map(SEASON_ORDER_DICT)

    ffs_data.sort_values(['name', 'season_order', 'gw'], inplace=True)

    return ffs_data


class DeepFantasyFootball:
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
        - Scale feature values as done in model training
        - Only keep historic player records as needed by LTSM
        - Drop any unused features

        :param full_data: DataFrame of player season-gameweek data as returned by `load_live_data` or `load_retro_data`
        :return: List of player names in prediction data and list of corresponding LSTM input data as a DataFrame
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
        available_players = full_data.copy()[
            (full_data['gw'] == self.previous_gw) &
            (full_data['season_order'] == self.prediction_season_order)
        ][['name']].reset_index(drop=True)

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

        # Scale values as done in training
        gw_prediction_data[FINAL_FEATURES] = mms.transform(gw_prediction_data[FINAL_FEATURES])

        # Only keep records needed for LSTM
        gw_prediction_data = gw_prediction_data.groupby('name').tail(self.N_STEPS_IN)
        logging.info(f"Player data shape after only keeping records needed for LSTM: {gw_prediction_data.shape}")

        # Only keep features needed for LSTM and name column
        gw_prediction_data = gw_prediction_data[FINAL_FEATURES + ['name']]

        # Get list of player names in prediction set and list of corresponding input data as a DataFrame
        player_list = []
        player_data_list = []

        for player, player_data in gw_prediction_data.groupby('name'):
            player_list.append(player)
            player_data_list.append(player_data.drop('name', axis=1))

        return player_list, player_data_list

    def make_player_predictions(self, player_data_list):
        """
        Make player predictions for next 5 gameweeks using LSTM model.

        :param player_data_list: List of corresponding player input data for player names in `player_list`.
        As returned by `prepare_data_for_lstm`

        :return: DataFrame containing individual player predictions.
        """
        input_array = np.concatenate(
            # Make each player player DataFrame into a 3D array
            [df.values.reshape(1, self.N_STEPS_IN, df.values.shape[1]) for df in player_data_list],
            axis=0
        )

        logging.info(f"LSTM input array shape: {input_array.shape}")

        raw_predictions = deep_fantasy_football_model.predict(input_array)

        final_predictions = pd.DataFrame(
            raw_predictions,
            columns=['GW_plus_1', 'GW_plus_2', 'GW_plus_3', 'GW_plus_4', 'GW_plus_5']
        )

        final_predictions['sum'] = \
            final_predictions['GW_plus_1'] + \
            final_predictions['GW_plus_2'] + \
            final_predictions['GW_plus_3'] + \
            final_predictions['GW_plus_4'] + \
            final_predictions['GW_plus_5']

        return final_predictions

    def format_predictions(self, player_list, final_predictions, full_data, double_gw_teams=[]):
        """
        Format predictions returned by `make_player_predictions()` by sorting and appending additional columns.

        :param final_predictions: DataFrame of final predictions as returned by `make_player_predictions()`
        :param full_data: DataFrame of player season-gameweek data as returned by `load_live_data` or `load_retro_data`
        :param player_list: List of players in same order as `final_predictions`
        :param double_gw_teams: List of double gameweek teams in upcoming gameweek
        :return: Formatted DataFrame of player predictions
        """

        final_predictions_copy = final_predictions.copy()
        final_predictions_copy['name'] = player_list

        other_player_info = full_data.copy()[
            (full_data['gw'] == self.previous_gw) &
            (full_data['season_order'] == self.prediction_season_order)
        ][
            [
                'name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID', 'team_name'
            ]
        ]

        final_predictions_formatted = final_predictions_copy.merge(other_player_info, on='name', how='left')

        assert final_predictions_formatted['team_name'].isnull().sum() == 0, 'Some players not in full_data'

        # for double_gw_team in double_gw_teams:
        #     logging.info(f'Doubling GW_plus_1 predictions for {double_gw_team}')
        #     final_predictions_formatted.loc[
        #         final_predictions_formatted['team_name'] == double_gw_team,
        #         'GW_plus_1'
        #     ] = \
        #         final_predictions_formatted.loc[
        #             final_predictions_formatted['team_name'] == double_gw_team,
        #             'GW_plus_1'
        #         ] * 2

        # final_predictions_formatted['sum'] = \
        #     final_predictions_formatted['GW_plus_1'] + \
        #     final_predictions_formatted['GW_plus_2'] + \
        #     final_predictions_formatted['GW_plus_3'] + \
        #     final_predictions_formatted['GW_plus_4'] + \
        #     final_predictions_formatted['GW_plus_5']

        final_predictions_formatted.sort_values('sum', ascending=False, inplace=True)

        return final_predictions_formatted
