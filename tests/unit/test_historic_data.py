import unittest
import pandas as pd
import numpy as np

from src.data.historical_season_data import combine_all_season_data


class TestDataCombine(unittest.TestCase):

    def test_correct_team(self):
        """
        Tests that function which combines data gives correct teams for 10 test case name and season combinations
        """
        # Set-up
        full_data = combine_all_season_data(write_to_parquet=False, return_dataframe=True)
        name_season_team = full_data.groupby(
            ['name', 'season', 'team_name']
        ).count().reset_index()[
            ['name', 'season', 'team_name']
        ]
        test_cases = pd.read_csv('tests/data/historic_data_team_test.csv')

        # Exercise
        combined = name_season_team.merge(
            test_cases,
            on=['name', 'season'],
            how='inner'
        )
        np.testing.assert_array_equal(
            combined['team_name'],
            combined['expected_team_name']
        )

    def test_correct_match_details(self):
        """
        Tests that function which combines data gives correct match opponent, home/away and outcome for 10 test case
        name, gw, season, team_name combinations
        """
        # Set-up
        full_data = combine_all_season_data(write_to_parquet=False, return_dataframe=True)
        test_cases = pd.read_csv('tests/data/historic_data_match_details_test.csv')

        # Exercise
        combined = full_data.merge(
            test_cases,
            on=['name', 'gw', 'season', 'team_name'],
            how='inner'
        )

        np.testing.assert_array_equal(
            combined['team_name_opponent'],
            combined['expected_team_name_opponent']
        )

        np.testing.assert_array_equal(
            combined['was_home'],
            combined['expected_was_home']
        )

        np.testing.assert_array_equal(
            combined['team_a_score'],
            combined['expected_team_a_score']
        )

        np.testing.assert_array_equal(
            combined['team_h_score'],
            combined['expected_team_h_score']
        )
