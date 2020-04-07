import unittest
import pandas as pd
import numpy as np
import os
from parameterized import parameterized

os.chdir('/Users/stevengeorge/Documents/Github/fpl-analysis')  # TODO Fix paths

from src.models.LSTM.make_predictions import LSTMPlayerPredictor, load_retro_data, load_model


class TestMakePredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_run_predictions = pd.read_parquet("tests/data/gw29_v4_lstm_player_predictions.parquet")
        cls.previous_gw = 28
        cls.prediction_season_order = 4

        # Generate GW 29 predictions using retro data
        cls.full_data = load_retro_data(current_season_data_filepath='data/gw_player_data/gw_29_player_data.parquet')

        # TODO Can remove these lines once double GW logic implemented
        # Hacky way of getting missing players in previous GW in current predictions. Set last available GW to previous GW
        cls.full_data.loc[(cls.full_data['gw'] == cls.previous_gw - 1) & (cls.full_data['team_name'] == 'Manchester City'), 'gw'] = cls.previous_gw
        cls.full_data.loc[(cls.full_data['gw'] == cls.previous_gw - 1) & (cls.full_data['team_name'] == 'Aston Villa'), 'gw'] = cls.previous_gw
        cls.full_data.loc[(cls.full_data['gw'] == cls.previous_gw - 1) & (cls.full_data['team_name'] == 'Sheffield United'), 'gw'] = cls.previous_gw
        cls.full_data.loc[(cls.full_data['gw'] == cls.previous_gw - 1) & (cls.full_data['team_name'] == 'Arsenal'), 'gw'] = cls.previous_gw

        lstm_pred = LSTMPlayerPredictor(
            previous_gw=cls.previous_gw,
            prediction_season_order=cls.prediction_season_order
        )

        player_list, player_data_list = lstm_pred.prepare_data_for_lstm(full_data=cls.full_data)
        unformatted_predictions = lstm_pred.make_player_predictions(
            player_data_list=player_data_list
        )
        final_predictions = lstm_pred.format_predictions(
            player_list=player_list,
            final_predictions=unformatted_predictions,
            full_data=cls.full_data
        )

        # Account for double GW:
        for double_gw_team in ['Manchester City', 'Arsenal']:
            final_predictions.loc[final_predictions['team_name'] == double_gw_team, 'GW_plus_1'] = \
                final_predictions.loc[final_predictions['team_name'] == double_gw_team, 'GW_plus_1'] * 2

        # Update predictions based on known injury information:
        # TODO Can remove when it is possible to add this information into prediction function
        final_predictions.loc[final_predictions['name'] == 'adama_traoré', 'GW_plus_1'] = \
            final_predictions.loc[final_predictions['name'] == 'adama_traoré', 'GW_plus_1'] * 0.75

        final_predictions.loc[final_predictions['name'] == 'aaron_wan-bissaka', 'GW_plus_1'] = \
            final_predictions.loc[final_predictions['name'] == 'aaron_wan-bissaka', 'GW_plus_1'] * 0.75

        for gw in range(1, 6):
            final_predictions.loc[final_predictions['name'] == 'heung-min_son', f'GW_plus_{gw}'] = 0

        final_predictions['sum'] = final_predictions['GW_plus_1'] + \
            final_predictions['GW_plus_2'] + \
            final_predictions['GW_plus_3'] + \
            final_predictions['GW_plus_4'] + \
            final_predictions['GW_plus_5']

        final_predictions.sort_values('sum', ascending=False, inplace=True)

        cls.retro_run_predictions = final_predictions.copy()

    def test_retro_prediction_shape(cls):
        """
        Check that predictions in live and retro runs contain the same number of players.
        """
        cls.assertEqual(cls.live_run_predictions.shape[0], cls.retro_run_predictions.shape[0])

    def test_retro_no_duplicate_players(cls):
        """
        Check that there are no duplicate players in prediction data.
        """
        cls.assertFalse(cls.retro_run_predictions.duplicated().any())

    def test_retro_prediction_values(cls):
        """
        Check that prediction values (sum of next 5 GWs) are the same for retro and live runs. Depending on how far
        before the gameweek the live run was done prices may have changed for some players. This can cause predictions
        to differ between live and retro runs for these players. We therefore compare predictions for players where
        price was the same between live and retro input data.
        """
        combined = cls.retro_run_predictions.merge(
            cls.live_run_predictions,
            on='name',
            how='inner',
            suffixes=('_retro', '_live')
        )

        combined_same_prices = combined.copy()[combined['next_match_value_retro'] == combined['next_match_value_live']]

        # Suspect not exact due to new methodology for predictions - vectorised vs individual
        np.testing.assert_array_almost_equal(
            combined_same_prices['sum_retro'],
            combined_same_prices['sum_live'],
            decimal=5
        )

    # TODO Below tests should be replaced or removed following changes to make_predictions method.
    # @parameterized.expand([
    #     (21, 4),
    #     (1, 2),
    #     (38, 3),
    #     (5, 1)
    # ])
    # def test_prepare_data_for_lstm_max_gw(cls, previous_gw, prediction_season_order):
    #     """
    #     Checks removal of tail data by prepare_data_for_lstm() function. Check that the highest gameweek in the
    #     specified `prediction_season_order` is equal to the specified `previous_gw`
    #     """
    #     lstm_pred_test = LSTMPlayerPredictor(
    #         previous_gw=previous_gw,
    #         prediction_season_order=prediction_season_order
    #     )
    #     player_list, player_data_list = lstm_pred_test.prepare_data_for_lstm(full_data=cls.full_data)
    #
    #     gw_prediction_data = pd.concat(player_data_list, axis=0)
    #
    #     cls.assertEqual(
    #         gw_prediction_data[gw_prediction_data['season_order'] == prediction_season_order]['gw'].max(),
    #         previous_gw
    #     )
    #
    # @parameterized.expand([
    #     (21, 4),
    #     (1, 2),
    #     (38, 3),
    # ])
    # def test_prepare_data_for_lstm_gws_in_previous_seasons(cls, previous_gw, prediction_season_order):
    #     """
    #     Check that when prediction_season_order != 1, the number of unique gameweeks in prior seasons is 38 i.e. no data
    #     lost from previous seasons.
    #     """
    #     lstm_pred_test = LSTMPlayerPredictor(
    #         previous_gw=previous_gw,
    #         prediction_season_order=prediction_season_order
    #     )
    #     gw_prediction_data = lstm_pred_test.prepare_data_for_lstm(full_data=cls.full_data)
    #
    #     # Data before prediction_season_order
    #     gw_prediction_data_before = gw_prediction_data[gw_prediction_data['season_order'] != prediction_season_order]
    #
    #     cls.assertEqual(
    #         gw_prediction_data_before.groupby('season_order').nunique()['gw'].mean(),
    #         38
    #     )
