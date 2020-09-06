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
"""
Final features used in DeepFantasyFootball model
"""

FFS_ABBREVIATION_TO_FULL = {
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
"""
FFS team abbreviation to full name
"""

FFS_TABLES = [f'FINAL_FEATURES_{x}_{pos}' for pos in ['DEF', 'FWD', 'MID', 'GK'] for x in range(1, 7)]
"""
List of Fantasy Football Scout tables containing final features for each position
"""

JOINING_KEYS = ['Name', 'Team', 'position', 'full_name', 'gw']
"""
Joining keys for merging the final feature tables together
"""

CURRENT_SEASON = '2020-2021'
"""
Season to load latest FFS data for
"""

FIXTURE_URL = "https://fantasy.premierleague.com/api/fixtures/"
"""
Official FPL API endpoint for fixtures
"""

CHROMEDRIVER_PATH = "../../Python/Chrome Driver/chromedriver"
"""
Path to Chrome Driver for Selenium
"""

ODDSPORTAL_ODDS_URL = 'https://www.oddsportal.com/soccer/england/premier-league/'
"""
URL for OddsPortal upcoming match odds
"""

SHOW_ALL_PL_MATCHES_BUTTON_XPATH = '//*[@id="show-all-link"]/div/div/div/div/p/a'
"""
Button to show odds for all available upcoming fixtures
"""

TABLE_XPATH = '//*[@id="tournamentTable"]'
"""
XPath for table of odds
"""

ODDS_TABLE_COLUMN_NAMES = ['KO', 'Match', 'Match_dup', '1', 'X', '2', 'num_available_bookmakers']
"""
Column names for data in odds table
"""

FIXTURE_FEATURE_AGGREGATIONS = {
    'draw_odds': 'mean',
    'win_odds': 'mean',
    'lose_odds': 'mean',
    'number_of_home_matches': 'sum',
    'number_of_promoted_side_opponent': 'sum',
    'number_of_top_6_last_season_opponent': 'sum',
    'promoted_side': 'mean',
    'top_6_last_season': 'mean',
    'number_of_matches': 'sum'
}
"""
Aggregations to perform on fixture-related features
"""
