import pandas as pd
import numpy as np
import os
from scipy.special import expit

# add files from this repo to external file in data directory called 'FantasyPremierLeague

# this script concatenates gameweek-level stats, as well as merging on player positions from the aggregated 'players_raw' files

SEASONS = ["2016-17", "2017-18", "2018-19"]

SOURCE_DATA_PATH = "../../data/external/FootballResults/"

OUTPUT_DATA_PATH = "../../data/processed"


def main():
    file_names = os.listdir(SOURCE_DATA_PATH)

    combined_df = pd.DataFrame()
    for f in os.listdir(SOURCE_DATA_PATH):
        f_path = SOURCE_DATA_PATH + f
        season_df = pd.read_csv(f_path, parse_dates=["Date"], dayfirst=True)
        combined_df = combined_df.append(season_df, sort=True)

    # some nulls - removing
    combined_df = combined_df.loc[combined_df.HomeTeam.notna(), :]

    # adding result and goal difference info
    assert combined_df.FTR.isna().mean() == 0
    combined_df["result_val"] = combined_df.FTR.map({"H": 1, "A": 0, "D": 0.5})
    combined_df["goal_difference"] = combined_df.FTHG - combined_df.FTAG
    combined_df["expit_goal_difference"] = expit(combined_df.goal_difference)
    combined_df.sort_values(by="Date").to_pickle(
        OUTPUT_DATA_PATH + "/football_results.pkl"
    )


if __name__ == "__main__":
    main()
