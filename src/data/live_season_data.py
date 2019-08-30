# TODO Add in time delta table refreshes - if not save each run in case old gw data is removed e.g. due to transfers

# TODO Create CURRENT_SEASON constant

# TODO Add header

# TODO Rename file

# TODO Data types need to be changed to numeric

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import urllib.request
import os
import logging

from src.data.constants import \
    POSITION_MAP, \
    VALUE_MULTIPLE, \
    TEAM_SEASON_DATA
from src.features.simple_features import create_features_from_kickoff_time

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
#os.chdir('../..')  # TODO Work out best way of doing this

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

        players_raw = players_raw.copy()[
            [
                'first_name',
                'second_name',
                'team',
                'position',
                'id',
                'now_cost',
                'chance_of_playing_next_round',
                'chance_of_playing_this_round'
            ]
        ]
        players_raw['season'] = self.season

        players_raw['name'] = players_raw['first_name'] + '_' + players_raw['second_name']
        players_raw['name'] = players_raw['name'].str.lower()

        # Get team information for each player
        team_data = pd.read_csv(TEAM_SEASON_DATA)
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
            player_data[['name', 'id', 'team_name', 'position', 'promoted_side', 'top_6_last_season', 'season']],
            on=['name', 'id'],
            how='left'
        )

        # Get player position dummies
        gw_dataframe = pd.get_dummies(gw_dataframe, columns=['position'])

        # Get opponent team data
        team_data = pd.read_csv(TEAM_SEASON_DATA)
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

    def get_all_fixture_data_from_api(self):
        player_data = self.get_player_data_from_api()

        name_id_dict = dict(
            zip(
                player_data['name'],
                player_data['id']
            )
        )

        i = 1
        fixture_dataframe = pd.DataFrame()
        for name, player_id in name_id_dict.items():
            player_fixture_data = _get_player_upcoming_fixtures_data_from_api(name=name, player_id=player_id)
            fixture_dataframe = fixture_dataframe.append(player_fixture_data)

            if i % 10 == 0:
                logging.info(f"Completed {i}/{len(player_data)}")
            i += 1

        fixture_dataframe.rename(
            columns={
                'event': 'gw',
                'is_home': 'was_home'
            },
            inplace=True
        )

        fixture_dataframe = fixture_dataframe.merge(
            player_data[
                [
                    'name',
                    'id',
                    'team_name',
                    'position',
                    'promoted_side',
                    'top_6_last_season',
                    'season',
                    'now_cost',
                    'chance_of_playing_next_round',
                    'chance_of_playing_this_round'
                ]
            ],
            on=['name', 'id'],
            how='left'
        )

        # now_cost is latest player value
        fixture_dataframe.rename(columns={'now_cost': 'value'}, inplace=True)
        fixture_dataframe['value'] = fixture_dataframe['value'] / VALUE_MULTIPLE

        # Get opponent team data
        fixture_dataframe.loc[fixture_dataframe['was_home'] == True, 'opponent_team'] = fixture_dataframe['team_a']
        fixture_dataframe.loc[fixture_dataframe['was_home'] == False, 'opponent_team'] = fixture_dataframe['team_h']
        team_data = pd.read_csv(TEAM_SEASON_DATA)
        fixture_dataframe = fixture_dataframe.merge(
            team_data,
            left_on=['opponent_team', 'season'],
            right_on=['team', 'season'],
            suffixes=('', '_opponent'),
            how='left'
        )

        # Kickoff features
        fixture_dataframe = create_features_from_kickoff_time(fixture_dataframe)

        # Get player position dummies
        fixture_dataframe = pd.get_dummies(fixture_dataframe, columns=['position'])

        return fixture_dataframe


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


def _get_player_upcoming_fixtures_data_from_api(name, player_id):
    player_url = ELEMENT_SUMMARY_URL.format(player_id)
    player_gw_data = _get_fpl_json(player_url)

    player_fixture_raw = json_normalize(player_gw_data, 'fixtures')  # TODO Remove hard-coded elements
    player_fixture_raw['name'] = name
    player_fixture_raw['id'] = player_id

    return player_fixture_raw


if __name__ == "__main__":
    get_fpl_data = GetFPLData(season='2019-20')
    latest_gw_data = get_fpl_data.get_all_gameweek_data_from_api()
    latest_gw_data.to_parquet('data/processed/parquet', index=False)
