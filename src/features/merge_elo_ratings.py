import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from IPython import embed

from utils import rating

INPUT_FIXTURES_PATH = "../../data/processed/matches_with_rating.pkl"
INPUT_GAMEWEEK_PATH = "../../data/processed/gameweek_data.pkl"
OUTPUT_PATH = "../../data/gameweek_with_elo.pkl"
RATED_COLUMN_MAPPING = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "team_h_score",
    "FTAG": "team_a_score",
}


def result_map(home_score, away_score):
    if home_score > away_score:
        return 1.0
    elif home_score < away_score:
        return 0.0
    else:
        return 0.5


def gameweek_cleaning(gameweek_df):
    gameweek_df["home_team"] = np.where(
        gameweek_df.was_home, gameweek_df.team, gameweek_df.opponent_team
    )
    gameweek_df["away_team"] = np.where(
        gameweek_df.was_home, gameweek_df.opponent_team, gameweek_df.team
    )
    gameweek_df["result"] = gameweek_df.apply(
        lambda x: result_map(x.team_h_score, x.team_a_score), axis=1
    )
    gameweek_df["Date"] = pd.to_datetime(gameweek_df.kickoff_time).dt.tz_localize(None)

    return gameweek_df


def rated_cleaning(rated, column_mapping):
    rated["FTHG"] = rated.FTHG.astype(int)
    rated["FTAG"] = rated.FTAG.astype(int)
    return rated.rename(columns=column_mapping)


def main():
    gameweek_df = gameweek_cleaning(pd.read_pickle(INPUT_GAMEWEEK_PATH))
    rated_df = rated_cleaning(pd.read_pickle(INPUT_FIXTURES_PATH), RATED_COLUMN_MAPPING)

    gameweek_with_elo = pd.merge_asof(
        gameweek_df.sort_values(by="Date"),
        rated_df,
        by=["home_team", "away_team", "team_h_score", "team_a_score"],
        on="Date",
        direction="nearest",
    )
    gameweek_with_elo["elo_outcome_rating_p"] = np.where(
        gameweek_with_elo.was_home,
        gameweek_with_elo.elo_outcome_rating_1,
        gameweek_with_elo.elo_outcome_rating_2,
    )

    gameweek_with_elo["elo_outcome_rating_o"] = np.where(
        gameweek_with_elo.was_home,
        gameweek_with_elo.elo_outcome_rating_2,
        gameweek_with_elo.elo_outcome_rating_1,
    )

    gameweek_with_elo["elo_outcome_rating_d"] = np.where(
        gameweek_with_elo.was_home,
        gameweek_with_elo.elo_outcome_rating_diff,
        gameweek_with_elo.elo_outcome_rating_diff * -1,
    )

    gameweek_with_elo["elo_outcome_rating_ex"] = np.where(
        gameweek_with_elo.was_home,
        gameweek_with_elo.elo_outcome_e,
        1.0 - gameweek_with_elo.elo_outcome_e,
    )

    gameweek_with_elo.to_pickle(OUTPUT_PATH)


if __name__ == "__main__":
    main()
