from functools import reduce
import time
import logging

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from keras.models import load_model
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

from src.models.utils import \
    _load_model_from_pickle
from src.data.s3_utilities import s3_filesystem
from src.data.live_season_data import _get_fpl_json
from src.models.DeepFantasyFootball.utils import calculate_fpl_points
from src.data.historical_fixture_and_odds import ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME
from src.models.constants import SEASON_ORDER_DICT
from src.models.DeepFantasyFootball.constants import \
    FINAL_FEATURES, \
    FFS_ABBREVIATION_TO_FULL, \
    FFS_TABLES, \
    JOINING_KEYS, \
    FIXTURE_URL, \
    FIXTURE_FEATURE_AGGREGATIONS, \
    CHROMEDRIVER_PATH, \
    ODDSPORTAL_ODDS_URL, \
    SHOW_ALL_PL_MATCHES_BUTTON_XPATH, \
    TABLE_XPATH, \
    ODDS_TABLE_COLUMN_NAMES, \
    CURRENT_SEASON, \
    ODDS_TABLE_COLUMN_NAMES_COMPLETED_MATCHES

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

mms = _load_model_from_pickle('src/models/pickles/min_max_scalar_DeepFantasyFootball_v01.pickle')
"""
MinMaxScalar used in training
"""

deep_fantasy_football_model = load_model("src/models/pickles/DeepFantasyFootball_v01.h5")
"""
DeepFantasyFootball fitted model
"""

new_positions = {
    'matt_ritchie': 'DEF',
    'pierre-emerick_aubameyang': 'FWD',
    'stuart_dallas': 'MID',
    'cheikhou_kouyate': 'DEF',
    'ainsley_maitland-niles': 'MID',
    'allan_saint-maximin': 'FWD'
}
"""
Players who changed position for the 2021-22 season. Only includes players who could be scored using the 
DeepFantasyFootball model
"""


def _load_latest_ffs_season_data(season):
    """
    Loads all FFS data from a specified season

    :param season: Season string of the format 20XX-20XX
    :return: DataFrame of latest FFS data
    """
    test_filepath = f"s3://fantasy-football-scout/raw-member-data/table=FINAL_FEATURES_1_DEF/season={season}"
    if s3_filesystem.exists(test_filepath):
        logging.info(f'FFS data for {season} available')
    else:
        logging.info(f'No data for {season} available, returning empty DataFrame for latest FFS data')
        return pd.DataFrame()

    # Load individual FFS feature files
    ffs_df_dict = {}

    for table in FFS_TABLES:
        logging.info(f'Working on {table}')
        df = pq.read_table(
            f"s3://fantasy-football-scout/raw-member-data/table={table}/season={season}",
            filesystem=s3_filesystem
        ).to_pandas()
        ffs_df_dict[table] = df
        logging.info(f'Finished {table}')

    # Combine into a single DataFrame
    latest_ffs_all_data = pd.DataFrame()

    for position in ['FWD', 'GK', 'DEF', 'MID']:
        dfs = [ffs_df_dict[key] for key in [key for key in ffs_df_dict.keys() if f'_{position}' in key]]

        combined = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=JOINING_KEYS,
                how='outer'
            ),
            dfs
        )

        logging.info(f'Shape of data for {position}: {combined.shape}')

        latest_ffs_all_data = latest_ffs_all_data.append(combined)

    logging.info(f'Latest FFS data shape: {latest_ffs_all_data.shape}')

    # Processing
    latest_ffs_all_data = latest_ffs_all_data.apply(pd.to_numeric, errors='ignore')

    latest_ffs_all_data['name'] = latest_ffs_all_data['full_name'].str.split(" ", 1).apply(
        lambda x: '_'.join([x[0], x[1]])
    ).str.lower()

    latest_ffs_all_data['total_points'] = calculate_fpl_points(latest_ffs_all_data)

    latest_ffs_all_data['season'] = season

    return latest_ffs_all_data


def _add_0_minute_events(non_zero_events_df):
    """
    FFS only includes data for players who played more than 0 minutes in a given gameweek. We therefore add back these
    zero events before making any predictions. Note: This can lead to the creation of fixtures which did not occur in a
    given gameweek. This is resolved by inner joining to fixture data in a later step.

    :param non_zero_events_df:
    :return:
    """
    logging.info(f'Shape of data before zero events: {non_zero_events_df.shape}')
    ffs_data_names = non_zero_events_df[['name', 'Team', 'position', 'season']].drop_duplicates()
    ffs_data_names['key'] = 1

    gw_df = pd.DataFrame({'gw': range(1, 39)})
    gw_df['key'] = 1

    all_player_season_gw_df = gw_df.merge(ffs_data_names, on='key')
    all_player_season_gw_df.drop('key', axis=1, inplace=True)

    all_player_season_gw_df.sort_values(['name', 'season', 'gw'], inplace=True)

    ffs_data = all_player_season_gw_df.merge(
        non_zero_events_df,
        on=['name', 'Team', 'position', 'gw', 'season'],
        how='left'
    )
    ffs_data.sort_values(['name', 'season', 'gw'], inplace=True)

    # From an offline check the missing entries are 0 minutes players in a given GW. We can therefore fill all missing
    # data points with 0.
    ffs_data.fillna(0, inplace=True)

    logging.info(f'Shape of data after adding zero events: {ffs_data.shape}')

    return ffs_data


def _load_season_fixtures():
    """
    Load fixtures for the current season from FPL API.
    :return: DataFrame of fixtures by gameweek. team_a and team_h are specified by their FPL codes
    """
    logging.info('Fetching fixtures from FPL API')

    fixtures = pd.json_normalize(
        _get_fpl_json(FIXTURE_URL)
    )
    fixtures.rename(columns={'event': 'gw'}, inplace=True)
    fixtures = fixtures[['gw', 'team_a', 'team_h']]

    logging.info(f'Fixtures shape: {fixtures.shape}')

    return fixtures


def _load_latest_match_odds():
    logging.info('Fetching latest odds data from OddsPortal')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        CHROMEDRIVER_PATH,
        options=chrome_options,
    )
    logging.info("Driver created")

    driver.implicitly_wait(10)

    driver.get(ODDSPORTAL_ODDS_URL)
    logging.info("Reached OddsPortal website")
    time.sleep(1)

    try:
        driver.find_element_by_xpath(SHOW_ALL_PL_MATCHES_BUTTON_XPATH).click()  # Expand odds box
        logging.info('Expanding table of odds')
    except NoSuchElementException:
        pass

    tbl = driver.find_element_by_xpath(TABLE_XPATH).get_attribute("outerHTML")
    logging.info('Scraped table of odds')

    odds_table = pd.read_html(tbl, header=0)[0]

    odds_table.columns = ODDS_TABLE_COLUMN_NAMES
    odds_table.dropna(axis=0, how='all', inplace=True)  # Drop rows with all nulls

    # Keep matches only (some rows are repeats of the header)
    odds_table = odds_table[odds_table['Match'].str.contains('-')]

    for odd_col in ['1', 'X', '2']:
        odds_table[odd_col].replace('-', np.nan, inplace=True)
        odds_table.loc[
            ~odds_table[odd_col].isnull(),
            odd_col
        ] = odds_table.loc[
            ~odds_table[odd_col].isnull(),
            odd_col
        ].str.split('/').apply(lambda x: float(x[0]) / float(x[1]))

    odds_table['home_team'] = odds_table['Match'].str.split(' - ').apply(lambda x: x[0])
    odds_table['away_team'] = odds_table['Match'].str.split(' - ').apply(lambda x: x[1])

    odds_table.drop(columns=['Match', 'Match_dup', 'KO', 'num_available_bookmakers'], inplace=True)

    logging.info('Formatted odds table')
    logging.info(odds_table.head())

    driver.close()

    return odds_table


def _load_match_odds_for_completed_matches_in_current_season():
    logging.info('Fetching odds for completed matches in current season from OddsPortal')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        CHROMEDRIVER_PATH,
        options=chrome_options,
    )
    logging.info("Driver created")

    driver.implicitly_wait(10)

    season_odds_df = pd.DataFrame()

    for page_number in range(1, 9):
        try:
            driver.get(f'https://www.oddsportal.com/soccer/england/premier-league/results/#/page/{page_number}/')
            time.sleep(1)

            tbl = driver.find_element_by_xpath(TABLE_XPATH).get_attribute("outerHTML")
            odds_table = pd.read_html(tbl, header=0)[0]

            odds_table.columns = ODDS_TABLE_COLUMN_NAMES_COMPLETED_MATCHES
            odds_table.dropna(axis=0, how='all', inplace=True)

            # Keep matches only (some rows are repeats of the header)
            odds_table = odds_table[odds_table['Match'].str.contains('-')]

            for odd_col in ['1', 'X', '2']:
                odds_table[odd_col].replace('-', np.nan, inplace=True)
                odds_table.loc[
                    ~odds_table[odd_col].isnull(),
                    odd_col
                ] = odds_table.loc[
                    ~odds_table[odd_col].isnull(),
                    odd_col
                ].str.split('/').apply(lambda x: float(x[0]) / float(x[1]))

            odds_table['home_team'] = odds_table['Match'].str.split(' - ').apply(lambda x: x[0])
            odds_table['away_team'] = odds_table['Match'].str.split(' - ').apply(lambda x: x[1])

            logging.info(f'Scraped odds from page {page_number}')
            logging.info(odds_table.shape)

            season_odds_df = season_odds_df.append(odds_table)

        except:
            logging.info(f'No odds on page {page_number}')
            continue

        season_odds_df = season_odds_df[['1', 'X', '2', 'home_team', 'away_team']]

        driver.close()

        return season_odds_df


def load_live_fixture_and_odds_data():
    # Fixtures
    fixtures = _load_season_fixtures()

    # Odds
    odds_table = _load_latest_match_odds()
    completed_match_odds_current_season = _load_match_odds_for_completed_matches_in_current_season()
    odds_table = odds_table.append(completed_match_odds_current_season)

    # Team data
    logging.info('Loading team data')
    team_data = pd.read_csv('data/external/team_season_data.csv')
    team_data['season'] = team_data['season'].apply(lambda x: x.replace('-', '-20'))

    team_data = team_data.copy()[team_data['season'] == CURRENT_SEASON]
    team_data.drop(columns=['team_name', 'season'], inplace=True)
    team_data.rename(columns={'team_name_ffs': 'team_name'}, inplace=True)
    logging.info('Loaded team data')

    # Combine
    team_data_a = team_data.copy()
    team_data_a.columns = [col + '_a' for col in team_data_a.columns]

    fixtures = fixtures.merge(team_data_a, on='team_a')

    team_data_h = team_data.copy()
    team_data_h.columns = [col + '_h' for col in team_data_h.columns]

    fixtures = fixtures.merge(team_data_h, on='team_h')

    fixtures.sort_values('gw', inplace=True)

    # Align team names so all use FFS
    odds_table['home_team'].replace(ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME, inplace=True)
    odds_table['away_team'].replace(ODDS_DATA_TEAM_NAME_TO_FFS_TEAM_NAME, inplace=True)
    logging.info('Aligned team names between sources')

    # Merge
    fixtures = fixtures.merge(
        odds_table,
        left_on=['team_name_h', 'team_name_a'],
        right_on=['home_team', 'away_team'],
        how='left'
    )

    fixtures.rename(
        columns={
            '1': 'home_win',
            'X': 'draw',
            '2': 'away_win'
        },
        inplace=True
    )
    logging.info('Combined data sources')

    # Reformat fixture data (Amend fixture data so that it is at a team level and specifies whether that team played
    # home or away. This doubles the size of the dataset)
    teams_in_gw_1 = fixtures.copy()
    teams_in_gw_1['team_name'] = teams_in_gw_1['home_team']
    teams_in_gw_1['team_name_opponent'] = teams_in_gw_1['away_team']
    teams_in_gw_1['was_home'] = True

    teams_in_gw_2 = fixtures.copy()
    teams_in_gw_2['team_name'] = teams_in_gw_2['away_team']
    teams_in_gw_2['team_name_opponent'] = teams_in_gw_2['home_team']
    teams_in_gw_2['was_home'] = False

    teams_in_gw = teams_in_gw_1.append(teams_in_gw_2)
    logging.info('Reformatted fixture data to team-level')

    # Win/Lose odds
    teams_in_gw['win_odds'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['home_win'],
        teams_in_gw['away_win']
    )

    teams_in_gw['lose_odds'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['away_win'],
        teams_in_gw['home_win']
    )

    teams_in_gw.rename(columns={'draw': 'draw_odds'}, inplace=True)

    for col in ['win_odds', 'lose_odds', 'draw_odds']:
        teams_in_gw[col] = pd.to_numeric(teams_in_gw[col])

    teams_in_gw.drop(
        columns=['team_a', 'team_h', 'team_name_a', 'team_name_h', 'home_win', 'away_win', 'home_team', 'away_team'],
        inplace=True
    )

    # Promoted side and top 6 for team and opponent
    teams_in_gw['promoted_side'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['promoted_side_h'],
        teams_in_gw['promoted_side_a']
    )

    teams_in_gw['top_6_last_season'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['top_6_last_season_h'],
        teams_in_gw['top_6_last_season_a']
    )

    teams_in_gw['promoted_side_opponent'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['promoted_side_a'],
        teams_in_gw['promoted_side_h']
    )

    teams_in_gw['top_6_last_season_opponent'] = np.where(
        teams_in_gw['was_home'],
        teams_in_gw['top_6_last_season_a'],
        teams_in_gw['top_6_last_season_h']
    )

    teams_in_gw.drop(
        columns=['promoted_side_a', 'top_6_last_season_a', 'promoted_side_h', 'top_6_last_season_h'],
        inplace=True
    )

    # Groupby aggregations
    teams_in_gw.rename(
        columns={
            'was_home': 'number_of_home_matches',
            'promoted_side_opponent': 'number_of_promoted_side_opponent',
            'top_6_last_season_opponent': 'number_of_top_6_last_season_opponent',
        },
        inplace=True
    )
    teams_in_gw['number_of_matches'] = 1

    teams_in_gw_agg = teams_in_gw.groupby(['gw', 'team_name']).agg(FIXTURE_FEATURE_AGGREGATIONS).reset_index()
    logging.info('Made feature aggregations')

    # Double gameweek flag
    teams_in_gw_agg['double_gameweek'] = np.where(
        teams_in_gw_agg['number_of_matches'] == 2,
        1,
        0
    )
    teams_in_gw_agg.drop('number_of_matches', axis=1, inplace=True)

    # Next match/gameweek fixtures
    historical_fixture_and_odds_features = pd.read_parquet(
        'data/processed/formatted_fixture_and_odds_features_2011_to_2020.parquet'
    )

    latest_fixture_and_odds = teams_in_gw_agg.copy()
    latest_fixture_and_odds['season'] = CURRENT_SEASON

    fixture_and_odds_features = historical_fixture_and_odds_features.append(
        latest_fixture_and_odds
    )
    fixture_and_odds_features.sort_values(['season', 'team_name', 'gw'], inplace=True)

    for feature in [
        'draw_odds',
        'win_odds',
        'lose_odds',
        'number_of_home_matches',
        'number_of_promoted_side_opponent',
        'number_of_top_6_last_season_opponent',
        'double_gameweek'
    ]:
        fixture_and_odds_features[f'next_gameweek_{feature}'] = fixture_and_odds_features.groupby(['team_name'])[
            feature
        ].shift(-1)
    logging.info('Created next match fixtures')

    return fixture_and_odds_features


def load_live_data():
    """
    Load all live input data required for predictions: historical, current season, next fixture. Also creates any
    additional features.

    :return: DataFrame containing all required inputs for all gameweeks
    """
    # Historical FFS data
    logging.info('Loading historical FFS data')
    historical_ffs_all_data = pq.read_table(
        f"s3://fantasy-football-scout/processed/fantasy_football_scout_final_features_and_total_points_UpTo2021.parquet",
        filesystem=s3_filesystem
    ).to_pandas()

    historical_ffs_all_data.drop(columns=['Name', 'full_name'], inplace=True)

    logging.info(f'Loaded historical FFS data of shape {historical_ffs_all_data.shape}')

    # Latest FFS data
    logging.info('Loading latest FFS data')
    latest_ffs_all_data = _load_latest_ffs_season_data(season=CURRENT_SEASON)
    logging.info(f'Loaded latest FFS data of shape: {latest_ffs_all_data.shape}')

    # Append latest to historical
    combined_ffs_all_data = historical_ffs_all_data.append(latest_ffs_all_data)
    logging.info(f'Combined FFS data shape: {combined_ffs_all_data.shape}')

    # Add 0 minute events back into data
    ffs_data = _add_0_minute_events(combined_ffs_all_data)

    # Make position changes for players who changed position in 2020-21. Re-calculate what historical points would have
    # been for these players had they always played in this position
    for name, pos in new_positions.items():
        ffs_data.loc[ffs_data['name'] == name, 'position'] = pos
    ffs_data['total_points'] = calculate_fpl_points(ffs_data)  # Re-calculate total points

    # Position dummies
    ffs_data = pd.get_dummies(ffs_data, columns=['position'])
    ffs_data.rename(columns={'Team': 'team_name'}, inplace=True)

    # Fixture and odds data
    fixture_and_odds_features = load_live_fixture_and_odds_data()

    # Format FFS team names and fixture team names to match
    ffs_data['team_name'].replace(FFS_ABBREVIATION_TO_FULL, inplace=True)

    # Merge FFS data with fixture and odds data
    # Inner join also removes gw-season fixtures which did not happen but were filled with 0s

    ffs_data = ffs_data.merge(
        fixture_and_odds_features,
        on=['season', 'gw', 'team_name'],
        how='inner'
    )

    # Filter final features
    ffs_data = ffs_data[['name', 'team_name', 'season'] + FINAL_FEATURES]

    ffs_data['season'] = ffs_data['season'].str.replace('-20', '-')
    ffs_data['season_order'] = ffs_data['season'].map(SEASON_ORDER_DICT)

    ffs_data.sort_values(['name', 'season_order', 'gw'], inplace=True)

    return ffs_data


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
        f"s3://fantasy-football-scout/processed/fantasy_football_scout_final_features_and_total_points_UpTo2021.parquet",
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
        set(ffs_data['team_name'].replace(FFS_ABBREVIATION_TO_FULL)) - set(fixture_and_odds_features['team_name'])
    ) == 0

    assert len(
        set(fixture_and_odds_features['team_name']) - set(ffs_data['team_name'].replace(FFS_ABBREVIATION_TO_FULL))
    ) == 0

    ffs_data['team_name'].replace(FFS_ABBREVIATION_TO_FULL, inplace=True)

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

        final_predictions_formatted.sort_values('sum', ascending=False, inplace=True)

        return final_predictions_formatted
