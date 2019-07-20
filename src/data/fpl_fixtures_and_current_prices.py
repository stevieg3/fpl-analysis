import pandas as pd
import numpy as np
import os

FPL_2019_PATH = "../../data/external/Fantasy-Premier-League/data/2019-20/"

RAW_OUTPUT_PATH = "../../data/raw/"


def main():
    player_df = pd.read_csv(
        FPL_2019_PATH + "cleaned_players.csv")

    fixtures_df = pd.read_csv(FPL_2019_PATH + "fixtures.csv")

    player_df.to_pickle(RAW_OUTPUT_PATH + "cleaned_players.pkl")

    fixtures_df.to_pickle(RAW_OUTPUT_PATH + "fixtures.pkl")


if __name__ == "__main__":
    main()
