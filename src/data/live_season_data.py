# TODO Add in time delta table refreshes - if not save each run in case old gw data is removed e.g. due to transfers

# TODO Create CURRENT_SEASON constant

# TODO Add header

# TODO Rename file

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import urllib.request

from src.data.constants import \
    POSITION_MAP, \
    VALUE_MULTIPLE

# TODO Move to constants
BOOTSTRAP_STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"

COLUMNS_TO_DROP = [
    'element',
    'fixture',
    'id',
    'kickoff_time',
    'opponent_team',
    'team'
]

# TODO Potentially make into a class with 'season' as parameter
# TODO Add docstrings


class GetFPLData:
    def __init__(self, season):
        self.season = season

    def get_player_data_from_api(self):  # TODO Rename function
        player_data_json = _get_fpl_json(BOOTSTRAP_STATIC_URL)
        players_raw = json_normalize(player_data_json, 'elements')  # TODO Remove hard-coded elements

        # Get player position
        players_raw['element_type'] = players_raw['element_type'].map(POSITION_MAP)
        players_raw.rename(columns={'element_type': 'position'}, inplace=True)

        players_raw = players_raw.copy()[['first_name', 'second_name', 'team', 'position', 'id']]
        players_raw['season'] = self.season

        players_raw['name'] = players_raw['first_name'] + '_' + players_raw['second_name']
        players_raw['name'] = players_raw['name'].str.lower()

        # Get team information for each player
        team_data = pd.read_csv('data/external/team_season_data.csv')  # TODO Move file path to constants
        players_raw = players_raw.merge(team_data, on=['team', 'season'], how='left')

        return players_raw

    def get_all_gameweek_data_from_api(self):
        player_data = self.get_player_data_from_api()
        name_id_dict = dict(
            zip(
                player_data['name'],
                player_data['id']
            )
        )

        gw_dataframe = pd.DataFrame()
        for name, player_id in name_id_dict.items():
            player_gw_data = _get_player_gameweek_data_from_api(name=name, player_id=player_id)
            gw_dataframe = gw_dataframe.append(player_gw_data)

        gw_dataframe.rename(columns={'round': 'gw'}, inplace=True)

        # TODO Most of below steps can be done separately / moved to features:

        # Change value so that it aligns with website
        gw_dataframe['value'] = gw_dataframe['value'] / VALUE_MULTIPLE

        # Format kick-off time # TODO Make this a separate method
        gw_dataframe['kickoff_time'] = gw_dataframe['kickoff_time'].str.replace('T', ' ')
        gw_dataframe['kickoff_time'] = pd.to_datetime(gw_dataframe['kickoff_time'])
        gw_dataframe['kickoff_time'] = gw_dataframe['kickoff_time'].dt.tz_convert('Europe/London')

        # Get position and team features for each player and season column
        gw_dataframe = gw_dataframe.merge(
            player_data[['name', 'id', 'team_name', 'position', 'promoted_side', 'top_6_last_season', 'season']]
        )

        # Get player position dummies
        gw_dataframe = pd.get_dummies(gw_dataframe, columns=['position'])

        # Get opponent team data
        team_data = pd.read_csv('data/external/team_season_data.csv')
        gw_dataframe = gw_dataframe.merge(
            team_data,
            left_on=['opponent_team', 'season'],
            right_on=['team', 'season'],
            suffixes=('', '_opponent'),
            how='left'
        )

        # Get features from kickoff time
        gw_dataframe['kickoff_month'] = gw_dataframe['kickoff_time'].dt.strftime("%b")
        gw_dataframe = pd.get_dummies(gw_dataframe, columns=['kickoff_month'])

        gw_dataframe['kickoff_hour'] = gw_dataframe['kickoff_time'].dt.hour

        gw_dataframe['late_kickoff'] = np.where(
            gw_dataframe['kickoff_hour'] >= 17,
            1,
            0
        )

        gw_dataframe['early_kickoff'] = np.where(
            gw_dataframe['kickoff_hour'] <= 13,
            1,
            0
        )

        gw_dataframe.drop(columns=['kickoff_hour'], axis=1, inplace=True)

        # Drop irrelevant/unknown features
        gw_dataframe.drop(
            columns=COLUMNS_TO_DROP,
            axis=1,
            inplace=True
        )

        # Create unique ID for each player
        # TODO Check if necessary to create ID here especially given that new season data is added
        id_df = gw_dataframe.groupby(['name']).count().reset_index()[['name']]
        id_df['ID'] = id_df.index + 1

        gw_dataframe = gw_dataframe.merge(id_df, how='left', on=['name'])

        return gw_dataframe


def _get_fpl_json(url):
    with urllib.request.urlopen(url) as open_url:
        json_file = json.loads(open_url.read())
    return json_file


def _get_player_gameweek_data_from_api(name, player_id):
    player_url = ELEMENT_SUMMARY_URL.format(player_id)
    player_gw_data = _get_fpl_json(player_url)

    player_gw_raw = json_normalize(player_gw_data, 'history')  # TODO Remove hard-coded elements
    player_gw_raw['name'] = name
    player_gw_raw['id'] = player_id

    return player_gw_raw
