FPL_DATA_PATH = 'data/external/FantasyPremierLeague/'  # TODO Rename to HISTORICAL_
"""
Data path of historical FPL data. Source: https://github.com/vaastav/Fantasy-Premier-League
"""

POSITION_MAP = {
    1: 'GK',
    2: 'DEF',
    3: 'MID',
    4: 'FWD'
}
"""
Dictionary mapping element_type to descriptive player position.
Source: https://github.com/vaastav/Fantasy-Premier-League/issues/1
"""

MAX_PROPORTION_NO_NAME_MATCHES = 0.01
"""
Maximum allowed proportion of names which cannot be matched between gameweek and player DataFrames for historical data
"""

STARTING_GAMEWEEK = 1
"""
First gameweek of any season
"""

LAST_GAMEWEEK = 38
"""
Last gameweek of any season
"""

VALUE_MULTIPLE = 10
"""
Scalar difference between player value in raw data and on website
"""

TEAM_SEASON_DATA = 'data/external/team_season_data.csv'
# TODO Add docstring
