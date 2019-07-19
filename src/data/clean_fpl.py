import pandas as pd
import numpy as np

# add files from this repo to external file in data directory called 'FantasyPremierLeague'

# this script concatenates gameweek-level stats, as well as merging on player positions from the aggregated 'players_raw' files

SEASONS = ["2016-17", "2017-18", "2018-19"]

SOURCE_DATA_PATH = "../../data/external/FantasyPremierLeague/"

OUTPUT_DATA_PATH = "../../data/processed"
RAW_DATA_PATH = "../../data/raw"
TEAM_NUMBER_PATH = "../../data/external/FantasyPremierLeague/team_numbers.csv"


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


def add_position_and_team_num(gameweek_df, player_df):
    gameweek_df["clean_name"] = gameweek_df.name.str.replace(r"_(\d+)", "")
    player_df["clean_name"] = player_df.first_name + "_" + player_df.second_name

    # mapping on position
    # https://github.com/vaastav/Fantasy-Premier-League/issues/1

    player_df["pos"] = player_df.element_type.map(
        {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    )

    gameweek_df = (
        player_df[["clean_name", "pos", "team", "season"]]
        .drop_duplicates(subset=["clean_name", "season"])
        .merge(gameweek_df, how="inner", on=["clean_name", "season"])
    )

    gameweek_df["season_num"] = gameweek_df.season.map(
        {"2016-17": 1, "2017-18": 2, "2018-19": 3}
    )

    gameweek_df["position"] = player_df.pos
    gameweek_df = pd.get_dummies(gameweek_df, columns=["position"])
    return gameweek_df


def add_team_names(df, team_number_df):
    """
    I think this might fail for jan transfers
    """
    own_teams = df.rename(columns={"team": "team_number"}).merge(
        team_number_df, on=["season", "team_number"]
    )
    opp_team_number_df = team_number_df.rename(
        columns={"team": "opponent_team", "team_number": "opp_team_number"}
    )
    opp_teams = own_teams.rename(columns={"opponent_team": "opp_team_number"}).merge(
        opp_team_number_df, on=["season", "opp_team_number"]
    )
    # check nothing dropped
    assert df.shape[0] == opp_teams.shape[0]
    return opp_teams


def main():
    print("starting cleaning")
    gameweek_df, player_df = load_data(seasons=SEASONS)
    gameweek_df.to_pickle(RAW_DATA_PATH + "/gameweek_data.pkl")
    player_df.to_pickle(RAW_DATA_PATH + "/player_data.pkl")
    gameweek_with_player_info = add_position_and_team_num(gameweek_df, player_df)
    team_number_df = pd.read_csv(TEAM_NUMBER_PATH)
    gameweek_with_team_names = add_team_names(gameweek_with_player_info, team_number_df)
    gameweek_with_team_names.to_pickle(OUTPUT_DATA_PATH + "/gameweek_data.pkl")
    player_df.to_pickle(OUTPUT_DATA_PATH + "/player_data.pkl")
    print("done")


if __name__ == "__main__":
    main()
