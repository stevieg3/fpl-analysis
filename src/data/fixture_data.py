# TODO NOTE THIS ONLY WORKS FOR FIXTURES AS OF START OF SEASON - WILL CHANGE DURING SEASON
# TODO Find better way of getting live fixture data

import pandas as pd
import numpy as np
import unidecode

from src.data.constants import \
    POSITION_MAP

players_raw = pd.read_csv(
    'data/external/FantasyPremierLeague/2019-20/players_raw.csv',
    usecols=['first_name', 'now_cost', 'second_name', 'team', 'element_type'],
    encoding='utf-8'
)

players_raw['element_type'] = players_raw['element_type'].map(POSITION_MAP)
players_raw.rename(columns={'element_type': 'position'}, inplace=True)


fixtures = pd.read_csv(
    'data/external/FantasyPremierLeague/2019-20/fixtures.csv',
    usecols=['event', 'kickoff_time', 'team_a', 'team_h']
)
fixtures.rename(columns={'event': 'gw'}, inplace=True)

# TODO Create method for this and put in utils.py - also used in historical_season_data.py
fixtures['kickoff_time'] = fixtures['kickoff_time'].str.replace('T', ' ')
fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time'])
fixtures['kickoff_time'] = fixtures['kickoff_time'].dt.tz_convert('Europe/London')

assert fixtures.shape[0] == 380


# Get fixtures at team level with was_home feature
away_fixtures = fixtures[['gw', 'kickoff_time', 'team_a']]
away_fixtures['was_home'] = False
away_fixtures.rename(columns={'team_a': 'team'}, inplace=True)

home_fixtures = fixtures[['gw', 'kickoff_time', 'team_h']]
home_fixtures['was_home'] = True
home_fixtures.rename(columns={'team_h': 'team'}, inplace=True)

each_team_fixtures = away_fixtures.append(home_fixtures, ignore_index=True)

assert each_team_fixtures.shape[0] == 380 * 2  # Check that dataframe doubles in length as home and away now split out
assert each_team_fixtures.groupby('team').sum()['was_home'].unique() == 19  # Each team has 19 home games

# Combine player info with fixtures data
combined = players_raw.merge(each_team_fixtures, on='team', how='left')
assert combined.shape[0] == players_raw.shape[0] * 38

# Get name for players
combined['name'] = combined['first_name'] + '_' + combined['second_name']
combined['name'] = combined['name'].str.lower()

# Create kickoff features
combined['kickoff_month'] = combined['kickoff_time'].dt.strftime("%b")
combined = pd.get_dummies(combined, columns=['kickoff_month'])

# TODO Create method for this and put in utils.py - also used in historical_season_data.py
combined['kickoff_hour'] = combined['kickoff_time'].dt.hour

combined['late_kickoff'] = np.where(
    combined['kickoff_hour'] >= 17,
    1,
    0
)

combined['early_kickoff'] = np.where(
    combined['kickoff_hour'] <= 13,
    1,
    0
)
combined.drop(columns=['kickoff_hour'], axis=1, inplace=True)

# Format value feature
combined.rename(columns={'now_cost': 'value'}, inplace=True)
combined['value'] = combined['value'] / 10

combined['season'] = '2019-20'

combined = pd.get_dummies(combined, columns=['position'])

# Get opponent data
opponents = fixtures.copy()[['gw', 'team_a', 'team_h']]
opponents.rename(columns={'team_a': 'team', 'team_h': 'opponent'}, inplace=True)

reverse = fixtures.copy()[['gw', 'team_h', 'team_a']]
reverse.rename(columns={'team_h': 'team', 'team_a': 'opponent'}, inplace=True)

opponent_df = opponents.append(reverse, ignore_index=True)
opponent_df.sort_values('gw', inplace=True)

combined_with_opponent = combined.merge(opponent_df, on=['gw', 'team'])
assert combined_with_opponent.shape[0] == combined.shape[0]

# Get team and opponent team names and features
team_data = pd.read_csv('data/external/team_season_data.csv')
combined_with_opponent = combined_with_opponent.merge(
    team_data,
    on=['team', 'season'],
    how='left'
)
combined_with_opponent = combined_with_opponent.merge(
    team_data,
    left_on=['opponent', 'season'],
    right_on=['team', 'season'],
    suffixes=('', '_opponent'),
    how='left'
)

# Drop irrelevant features to align with other season data

COLUMNS_TO_DROP = ['kickoff_time', 'team_opponent', 'first_name', 'second_name', 'team', 'opponent']

combined_with_opponent.drop(
    columns=COLUMNS_TO_DROP,
    axis=1,
    inplace=True
)

combined_with_opponent.to_parquet('data/processed/2019_20_fixtures_as_of_season_start.parquet', index=False)
