import pandas as pd
import numpy as np

# add files from this repo to external file in data directory called 'FantasyPremierLeague

# this script concatenates gameweek-level stats, as well as merging on player positions from the aggregated 'players_raw' files

SEASONS = ["2016-17", "2017-18", "2018-19"]

SOURCE_DATA_PATH = "../../data/external/FantasyPremierLeague/"

OUTPUT_DATA_PATH = "../../data/processed"


def load_data(seasons):
    gameweek_df = pd.DataFrame()
    for season in seasons:
        for gw in range(1, 39):
            gw_path = SOURCE_DATA_PATH + f"{season}/gws/gw{gw}.csv"
            g_df = pd.read_csv(gw_path, encoding="ISO-8859-1")
            g_df["gw"] = gw
            g_df["season"] = season
            gameweek_df = gameweek_df.append(g_df, sort=False)

    # this includes position info, and possibly other stuff
    player_df = pd.DataFrame()
    for season in seasons:
        players_path = SOURCE_DATA_PATH + f"{season}/players_raw.csv"
        p_df = pd.read_csv(players_path)
        p_df["season"] = season
        player_df = player_df.append(p_df, sort=False)

    return gameweek_df.reset_index(drop=True), player_df.reset_index(drop=True)


def add_positions(gameweek_df, player_df):
    gameweek_df["clean_name"] = gameweek_df.name.str.replace(r"_(\d+)", "")
    player_df["clean_name"] = player_df.first_name + "_" + player_df.second_name

    # mapping on position
    # https://github.com/vaastav/Fantasy-Premier-League/issues/1

    player_df["pos"] = player_df.element_type.map(
        {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    )

    gameweek_df = (
        player_df[["clean_name", "pos"]]
        .drop_duplicates(subset="clean_name")
        .merge(gameweek_df, how="inner")
    )

    gameweek_df["season_num"] = gameweek_df.season.map(
        {"2016-17": 1, "2017-18": 2, "2018-19": 3}
    )

    gameweek_df["position"] = player_df.pos
    gameweek_df = pd.get_dummies(gameweek_df, columns=["position"])
    return gameweek_df


def main():
    print("starting cleaning")
    gameweek_df, player_df = load_data(seasons=SEASONS)
    gameweek_with_positions_df = add_positions(gameweek_df, player_df)
    gameweek_with_positions_df.to_pickle(OUTPUT_DATA_PATH + "/gameweek_data.pkl")
    print("done")


if __name__ == "__main__":
    main()
