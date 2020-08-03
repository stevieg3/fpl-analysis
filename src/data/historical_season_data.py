import pandas as pd
import numpy as np
import logging
import os

from src.data.constants import \
    HISTORICAL_FPL_DATA_PATH, \
    POSITION_MAP, \
    MAX_PROPORTION_NO_NAME_MATCHES, \
    STARTING_GAMEWEEK, \
    LAST_GAMEWEEK, \
    VALUE_MULTIPLE

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
os.chdir('/Users/stevengeorge/Documents/Github/fpl-analysis')  # TODO Find better way of setting root when using main()


########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# GAMEWEEK DATA                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


def _get_gameweek_data(season, starting_gameweek=STARTING_GAMEWEEK, last_gameweek=LAST_GAMEWEEK):
    gw_dataframe = pd.DataFrame()
    for gw in range(starting_gameweek, last_gameweek + 1):
        if season == '2019-20':
            gw_data = pd.read_csv(HISTORICAL_FPL_DATA_PATH + f'{season}/gws/gw{gw}.csv', encoding="utf8")
        else:
            gw_data = pd.read_csv(HISTORICAL_FPL_DATA_PATH + f'{season}/gws/gw{gw}.csv', encoding="ISO-8859-1")
        gw_data['gw'] = gw
        gw_data['season'] = season
        gw_dataframe = gw_dataframe.append(gw_data)

    # Change value so that it aligns with website
    gw_dataframe['value'] = gw_dataframe['value'] / VALUE_MULTIPLE

    # Format kick-off time
    gw_dataframe['kickoff_time'] = gw_dataframe['kickoff_time'].str.replace('T', ' ')
    gw_dataframe['kickoff_time'] = pd.to_datetime(gw_dataframe['kickoff_time'])
    gw_dataframe['kickoff_time'] = gw_dataframe['kickoff_time'].dt.tz_convert('Europe/London')

    if (season == '2018-19') | (season == '2019-20'):
        gw_dataframe['name'] = gw_dataframe['name'].str.replace(r"_(\d+)", "")  # remove trailing _number

    # Remove accents from names
    gw_dataframe['name'] = gw_dataframe['name'].str.lower()

    return gw_dataframe


########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# PLAYER DATA                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


def _get_player_data(season):
    players_raw = pd.read_csv(HISTORICAL_FPL_DATA_PATH + f'{season}/players_raw.csv', encoding="utf8")

    # Get player position
    players_raw['element_type'] = players_raw['element_type'].map(POSITION_MAP)
    players_raw.rename(columns={'element_type': 'position'}, inplace=True)

    players_raw = players_raw.copy()[['first_name', 'second_name', 'team', 'position']]
    players_raw['season'] = season

    players_raw['name'] = players_raw['first_name'] + '_' + players_raw['second_name']
    players_raw['name'] = players_raw['name'].str.lower()

    # Remove any players with the same name but different teams
    player_name_count = players_raw.groupby('name').size().reset_index(name='count')

    duplicates = player_name_count[player_name_count['count'] > 1]
    logging.info(f"Duplicate names in {season}: {list(duplicates['name'])}")

    players_raw = players_raw.merge(player_name_count, on='name', how='left')
    players_raw = players_raw[players_raw['count'] == 1]
    players_raw.drop('count', axis=1, inplace=True)

    # Get team information for each player
    team_data = pd.read_csv('data/external/team_season_data.csv')
    players_raw = players_raw.merge(team_data, on=['team', 'season'], how='left')

    return players_raw


########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# COMBINE ALL DATA                                                                                                     #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


def _combine_player_and_gameweek_data(season, starting_gameweek=STARTING_GAMEWEEK, last_gameweek=LAST_GAMEWEEK):
    gameweek_data = _get_gameweek_data(season=season, starting_gameweek=starting_gameweek, last_gameweek=last_gameweek)
    player_data = _get_player_data(season=season)

    # Check that number of player name non-matches due to accents isn't too high
    test = gameweek_data.merge(player_data, on=['name', 'season'], how='left')
    no_matches = test[test['first_name'].isnull()]

    logging.info(
        f"Number of players not matched: {no_matches['name'].nunique()}"
    )
    assert no_matches['name'].nunique() / test['name'].nunique() < MAX_PROPORTION_NO_NAME_MATCHES
    logging.info(
        f"Proportion of no name matches: {no_matches['name'].nunique() / test['name'].nunique()}"
    )

    full_season_data = gameweek_data.merge(player_data, on=['name', 'season'], how='inner')

    # Change team_name for mid-season transfers. This resolves a known issue where team_names are overwritten with the
    # new team joined in the middle of the season
    mid_season_transfers = pd.read_csv('data/external/mid_season_transfers_2016_to_2019.csv')
    mid_season_transfers = mid_season_transfers[mid_season_transfers['season'] == season]

    for _, row in mid_season_transfers.iterrows():
        name = row['name']
        transfer_date = row['transfer_date']
        old_team = row['old_team']

        full_season_data['team_name'] = np.where(
            (full_season_data['name'] == name) &
            (full_season_data['kickoff_time'] < transfer_date),
            old_team,
            full_season_data['team_name']
        )

    return full_season_data


COLUMNS_TO_DROP = [
    'element',
    'fixture',
    'id',
    'kickoff_time',
    'kickoff_time_formatted',
    'opponent_team',
    'round',
    'first_name',
    'second_name',
    'team',
    'team_opponent'
]


def combine_all_season_data(write_to_parquet=True, return_dataframe=False):
    logging.info(
        "Processing 2016/17 season data..."
    )
    gw_data_full_1617 = _combine_player_and_gameweek_data('2016-17')
    logging.info(
        "Finished processing 2016/17 season data!"
    )

    logging.info(
        "Processing 2017/18 season data..."
    )
    gw_data_full_1718 = _combine_player_and_gameweek_data('2017-18')
    logging.info(
        "Finished processing 2017/18 season data!"
    )

    logging.info(
        "Processing 2018/19 season data..."
    )
    gw_data_full_1819 = _combine_player_and_gameweek_data('2018-19')
    logging.info(
        "Finished processing 2018/19 season data!"
    )

    logging.info(
        "Combining season data and generating features..."
    )
    fpl_data_all_seasons = pd.DataFrame()
    for df in [gw_data_full_1617, gw_data_full_1718, gw_data_full_1819]:
        fpl_data_all_seasons = fpl_data_all_seasons.append(df)

    # Get player position dummies
    fpl_data_all_seasons = pd.get_dummies(fpl_data_all_seasons, columns=['position'])

    # Get opponent team data
    team_data = pd.read_csv('data/external/team_season_data.csv')
    fpl_data_all_seasons = fpl_data_all_seasons.merge(
        team_data,
        left_on=['opponent_team', 'season'],
        right_on=['team', 'season'],
        suffixes=('', '_opponent'),
        how='left'
    )

    # Get features from kickoff time
    fpl_data_all_seasons['kickoff_month'] = fpl_data_all_seasons['kickoff_time'].dt.strftime("%b")
    fpl_data_all_seasons = pd.get_dummies(fpl_data_all_seasons, columns=['kickoff_month'])

    fpl_data_all_seasons['kickoff_hour'] = fpl_data_all_seasons['kickoff_time'].dt.hour

    fpl_data_all_seasons['late_kickoff'] = np.where(
        fpl_data_all_seasons['kickoff_hour'] >= 17,
        1,
        0
    )

    fpl_data_all_seasons['early_kickoff'] = np.where(
        fpl_data_all_seasons['kickoff_hour'] <= 13,
        1,
        0
    )

    fpl_data_all_seasons.drop(columns=['kickoff_hour'], axis=1, inplace=True)

    # Drop irrelevant/unknown features
    fpl_data_all_seasons.drop(
        columns=COLUMNS_TO_DROP,
        axis=1,
        inplace=True
    )

    # Create unique ID for each player
    id_df = fpl_data_all_seasons.groupby(['name']).count().reset_index()[['name']]
    id_df['ID'] = id_df.index + 1

    logging.info(f"Number of unique players: {id_df.shape[0]}")

    fpl_data_all_seasons = fpl_data_all_seasons.merge(id_df, how='left', on=['name'])

    logging.info(
        "Finished combining all season data!"
    )

    logging.info(f"Shape of final DataFrame: {fpl_data_all_seasons.shape}")

    if write_to_parquet:
        fpl_data_all_seasons.to_parquet('data/processed/fpl_data_2016_to_2019.parquet', index=False)

    if return_dataframe:
        return fpl_data_all_seasons


if __name__ == "__main__":
    combine_all_season_data()
