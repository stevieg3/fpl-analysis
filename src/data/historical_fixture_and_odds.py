import pandas as pd

ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME = {
    'Blackburn': 'Blackburn Rovers',
    'Bolton': 'Bolton Wanderers',
    'Brighton': 'Brighton and Hove Albion',
    'Cardiff': 'Cardiff City',
    'Huddersfield': 'Huddersfield Town',
    'Hull': 'Hull City',
    'Leicester': 'Leicester City',
    'Manchester Utd': 'Manchester United',
    'Newcastle': 'Newcastle United',
    'Norwich': 'Norwich City',
    'QPR': 'Queens Park Rangers',
    'Sheffield Utd': 'Sheffield United',
    'Stoke': 'Stoke City',
    'Swansea': 'Swansea City',
    'Tottenham': 'Tottenham Hotspur',
    'West Brom': 'West Bromwich Albion',
    'West Ham': 'West Ham United',
    'Wigan': 'Wigan Athletic',
    'Wolves': 'Wolverhampton Wanderers'
}
"""
Mapping of team names in Odds Portal to teams names in FFS data
"""


def create_historical_fixture_and_odds_data(save=False):
    """
    Function for creating historical fixture and odds data for seasons 2016/17 to 2018/19

    :param save: Boolean. True to save as parquet
    :return: DataFrame of fixture and odds data if `save` is False
    """
    # Fixtures data
    all_fixtures = pd.read_parquet('data/external/all_premier_league_fixtures_by_gameweek_2011to2020.parquet')
    all_fixtures['season'] = all_fixtures['season'].str.replace('/', '-')
    all_fixtures.rename(
        columns={
            'Home': 'home_team',
            'Away': 'away_team'
        },
        inplace=True
    )

    # Historical odds from Odds Portal
    all_odds = pd.read_parquet('data/external/oddsportal_odds_2011to2020.parquet')
    all_odds.drop(columns=['KO', 'num_available_bookmakers'], inplace=True)
    all_odds.rename(
        columns={
            '1': 'home_win',
            'X': 'draw',
            '2': 'away_win'
        },
        inplace=True
    )
    # Rename teams to align
    all_odds['home_team'].replace(ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME, inplace=True)
    all_odds['away_team'].replace(ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME, inplace=True)

    # Combine
    fixture_and_odds = all_fixtures.merge(all_odds, on=['home_team', 'away_team', 'season'], how='inner')
    fixture_and_odds.drop(columns=['Score', 'Match', 'Result'], inplace=True)

    assert fixture_and_odds.shape[0] == 38 * 10 * 9  # 9 seasons worth of fixtures

    if save:
        fixture_and_odds.to_parquet('data/processed/fixture_and_odds_2011_to_2020.parquet', index=False)
    else:
        return fixture_and_odds
