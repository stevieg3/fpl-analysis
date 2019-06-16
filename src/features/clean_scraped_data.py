import pandas as pd
import numpy as np
import re
import os

from src.features.utils import fpl_name_map


def clean_raw_scraped_data(raw_data_path, write_to_csv=False, new_csv_name=None):
    """
    Cleans raw data scraped from FPL website

    :param raw_data_path: file path containing raw data
    :param write_to_csv: If True write to csv in data/interim else return formatted DataFrame
    :param new_csv_name: Use if write_to_csv = True. Name of formatted file
    :return: None or formatted DataFrame
    """
    try:
        raw_df = pd.read_excel(raw_data_path)
    except:
        raw_df = pd.read_csv(raw_data_path)
    
    # Format column names
    old_columns = list(raw_df.columns)
    new_columns = [re.sub(' +', ' ', col).strip() for col in (col_temp.replace('\n', '') for col_temp in old_columns)]
    raw_df.columns = new_columns
    
    # Expand 'Opposition' column into features
    opp_expanded = (
        raw_df['OPP Opposition'].str.split(" ", expand=True)
                                .drop(3, axis=1)
                                .rename(columns={0: 'opposition', 1: 'home_or_away', 2: 'goals1', 4: 'goals2'})
    )
    
    raw_df = raw_df.merge(opp_expanded, left_index=True, right_index=True)
    raw_df['home_or_away'] = raw_df['home_or_away'].apply(lambda x: x[1])
    raw_df['team_goals'] = np.where(raw_df['home_or_away'] == 'H', raw_df['goals1'], raw_df['goals2'])
    raw_df['opposition_goals'] = np.where(raw_df['home_or_away'] == 'H', raw_df['goals2'], raw_df['goals1'])
    raw_df.drop(['goals1', 'goals2', 'OPP Opposition'], axis=1, inplace=True)

    # Format pound values
    raw_df['£ Value'] = raw_df['£ Value'].apply(lambda x: float(x[1:]))

    # Rename columns
    raw_df.rename(columns=fpl_name_map, inplace=True)

    if write_to_csv:
        raw_df.to_csv('data/interim/{}.csv'.format(new_csv_name), index=False)
    else:
        return raw_df


def main():
    os.chdir('../..')
    clean_raw_scraped_data('data/raw/FPL 2017:18 player stats.xlsx',
                           write_to_csv=True, new_csv_name='clean_scraped_2017_18')

    clean_raw_scraped_data('data/raw/2018_19_current_season_data.csv',
                           write_to_csv=True, new_csv_name='clean_scraped_2018_19')


if __name__ == "__main__":
    main()
