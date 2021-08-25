from scipy.stats import uniform, randint


STATIC_FEATURES = [
    'was_home',
    'gw',
    'promoted_side',
    'top_6_last_season',
    'ID',
    'position_DEF',
    'position_FWD',
    'position_GK',
    'position_MID',
    'promoted_side_opponent', 'top_6_last_season_opponent',
    'kickoff_month_Apr', 'kickoff_month_Aug', 'kickoff_month_Dec', 'kickoff_month_Feb', 'kickoff_month_Jan',
    'kickoff_month_Mar', 'kickoff_month_May', 'kickoff_month_Nov', 'kickoff_month_Oct', 'kickoff_month_Sep',
    'late_kickoff',
    'early_kickoff'
]

TIME_SERIES_FEATURES = [
    'assists',
    'bonus',
    'target_missed',
    'errors_leading_to_goal_attempt',
    'creativity',
    'key_passes',
    'yellow_cards',
    'loaned_in',
    'team_a_score',
    'team_h_score',
    'clearances_blocks_interceptions',
    'offside',
    'dribbles',
    'loaned_out',
    'selected',
    'big_chances_missed',
    'ict_index',
    'red_cards',
    'value',
    'fouls',
    'influence',
    'errors_leading_to_goal',
    'ea_index',
    'goals_scored',
    'tackles',
    'transfers_balance',
    'transfers_in',
    'transfers_out',
    'attempted_passes',
    'completed_passes',
    'bps',
    'penalties_missed',
    'own_goals',
    'saves',
    'penalties_saved',
    'goals_conceded',
    'open_play_crosses',
    'total_points',
    'tackled',
    'clean_sheets',
    'winning_goals',
    'minutes',
    'big_chances_created',
    'penalties_conceded',
    'recoveries',
    'threat'
]


FPL_AVAILABLE_FEATURES_19_20 = [
    'assists',
    'bonus',
    'bps',
    'clean_sheets',
    'creativity',
    'goals_conceded',
    'goals_scored',
    'ict_index',
    'influence',
    'minutes',
    'own_goals',
    'penalties_missed',
    'penalties_saved',
    'red_cards',
    'gw',
    'saves',
    'selected',
    'team_a_score',
    'team_h_score',
    'threat',
    'total_points',
    'transfers_balance',
    'transfers_in',
    'transfers_out',
    'value',
    'was_home',
    'yellow_cards',
    'name',
    'team_name',
    'promoted_side',
    'top_6_last_season',
    'season',
    'position_DEF',
    'position_FWD',
    'position_GK',
    'position_MID',
    'team_name_opponent',
    'promoted_side_opponent',
    'top_6_last_season_opponent',
    'late_kickoff',
    'early_kickoff',
    'ID',
    'kickoff_month_Aug',
    'kickoff_month_Sep',
    'kickoff_month_Oct',
    'kickoff_month_Nov',
    'kickoff_month_Dec',
    'kickoff_month_Jan',
    'kickoff_month_Feb',
    'kickoff_month_Mar',
    'kickoff_month_Apr',
    'kickoff_month_May'
]
"""
Available features in FPL API for 2019/20 season plus kickoff month dummy variables 
"""


TIME_SERIES_FEATURES_19_20 = list(set(FPL_AVAILABLE_FEATURES_19_20).intersection(TIME_SERIES_FEATURES))
"""
Available time series features in 2019-20 season API
"""


SEASON_ORDER_DICT = {
    '2011-12': -4,
    '2012-13': -3,
    '2013-14': -2,
    '2014-15': -1,
    '2015-16': 0,
    '2016-17': 1,
    '2017-18': 2,
    '2018-19': 3,
    '2019-20': 4,
    '2020-21': 5,
    '2021-22': 6
}
"""
Order of seasons
"""


KNOWN_FEATURES_NEXT_GW = [
    'value',
    'was_home',
    'promoted_side_opponent',
    'top_6_last_season_opponent',
    'kickoff_month_Aug',
    'kickoff_month_Sep',
    'kickoff_month_Oct',
    'kickoff_month_Nov',
    'kickoff_month_Dec',
    'kickoff_month_Jan',
    'kickoff_month_Feb',
    'kickoff_month_Mar',
    'kickoff_month_Apr',
    'kickoff_month_May',
    'late_kickoff',
    'early_kickoff'
]
"""
Known features at GW t for GW t+1
"""


KICKOFF_MONTH_FEATURES = [
    'kickoff_month_Aug',
    'kickoff_month_Sep',
    'kickoff_month_Oct',
    'kickoff_month_Nov',
    'kickoff_month_Dec',
    'kickoff_month_Jan',
    'kickoff_month_Feb',
    'kickoff_month_Mar',
    'kickoff_month_Apr',
    'kickoff_month_May'
]
"""
Kickoff month dummy features
"""


LGBM_RANDOM_SEARCH_PARAMS = {
    'learning_rate': uniform(),
    'n_estimators': randint(3, 400),
    'num_leaves': randint(3, 400),
    'reg_alpha': uniform(),
    'reg_lambda': uniform()
}
"""
Parameters to use for randomised search of LGBM Regressor
"""


LGBM_PREDICTION_COLS_TO_EXCLUDE = ['name', 'team_name', 'season', 'team_name_opponent']
"""
Object columns to exclude when making predictions on data 
"""


LGBM_PICKLE_PATH = 'src/models/pickles/v2_2_lgbm_point_predictor.pickle'
"""
Path to LGBM model
"""


PREDICTIONS_OUTPUT_COLUMNS = [
    'name',
    'position_DEF',
    'position_FWD',
    'position_GK',
    'position_MID',
    'predictions',
    'team_name',
    'next_match_value'
]
"""
Columns to include in final predictions output
"""


# LSTM CONSTANTS

COLUMNS_TO_DROP_FOR_TRAINING = ['name', 'season', 'season_order', 'team_name', 'team_name_opponent', 'team_name_ffs']


SELL_ON_TAX = 0.5
"""
FPL tax on any profit through players sales. See https://twitter.com/officialfpl/status/810473725627957248?lang=en and
https://www.reddit.com/r/FantasyPL/comments/90pgwq/can_someone_explain_to_me_how_price_change_works/
"""


LOW_VALUE_PLAYER_UPPER_LIMIT = 4.1
"""
Value below which a player is classified as low value.
"""