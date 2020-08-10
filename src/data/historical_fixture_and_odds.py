import pandas as pd
import numpy as np

ODDS_DATA_TEAM_NAME_TO_FPL_TEAM_NAME = {
        'Hull': 'Hull City',
        'Leicester': 'Leicester City',
        'Manchester Utd': 'Manchester United',
        'Stoke': 'Stoke City',
        'Swansea': 'Swansea City',
        'Tottenham': 'Tottenham Hotspur',
        'West Brom': 'West Bromwich Albion',
        'West Ham': 'West Ham United',
        'Brighton': 'Brighton & Hove Albion',
        'Cardiff': 'Cardiff City',
        'Huddersfield': 'Huddersfield Town',
        'Newcastle': 'Newcastle United',
        'Wolves': 'Wolverhampton Wanderers'
    }
"""
Mapping of team names in Odds Portal to teams names in FPL data
"""


def create_historical_fixture_and_odds_data(save=False):
    """
    Function for creating historical fixture and odds data for seasons 2016/17 to 2018/19

    :param save: Boolean. True to save as parquet
    :return: DataFrame of fixture and odds data if `save` is False
    """
    # Fixtures data
    historical_fpl_data = pd.read_parquet('data/processed/fpl_data_2016_to_2019.parquet')

    fixtures = historical_fpl_data.copy()[['season', 'gw', 'team_name', 'team_name_opponent', 'was_home']]

    fixtures['home_team'] = np.where(
        fixtures['was_home'],
        fixtures['team_name'],
        fixtures['team_name_opponent']
    )

    fixtures['away_team'] = np.where(
        fixtures['was_home'],
        fixtures['team_name_opponent'],
        fixtures['team_name']
    )

    fixtures.drop_duplicates(['season', 'home_team', 'away_team'], inplace=True)
    fixtures.drop(columns=['team_name', 'team_name_opponent', 'was_home'], inplace=True)

    assert fixtures.shape[0] == 38 * 10 * 3  # 3 seasons worth of fixtures

    # Historical odds from Odds Portal
    odds_data = pd.read_parquet('data/external/oddsportal_odds_2016to2019.parquet')

    odds_data['season'] = odds_data['season'].map(
        {
            '2016-2017': '2016-17',
            '2017-2018': '2017-18',
            '2018-2019': '2018-19'
        }
    )

    odds_data['home_team'] = odds_data['home_team'].replace(ODDS_DATA_TEAM_NAME_TO_FPL_TEAM_NAME)
    odds_data['away_team'] = odds_data['away_team'].replace(ODDS_DATA_TEAM_NAME_TO_FPL_TEAM_NAME)

    assert len(
        set(fixtures['home_team']) - set(odds_data['home_team'])
    ) == 0  # Check that all team names have been mapped

    odds_data.drop(['KO', 'Match', 'num_available_bookmakers', 'Result'], axis=1, inplace=True)

    odds_data.rename(
        columns={
            '1': 'home_win',
            'X': 'draw',
            '2': 'away_win'
        },
        inplace=True
    )

    fixture_and_odds = fixtures.merge(
        odds_data,
        on=['home_team', 'away_team', 'season'],
        how='inner'
    )

    if save:
        fixture_and_odds.to_parquet('data/processed/fixture_and_odds_2016_to_2019.parquet', index=False)
    else:
        return fixture_and_odds
