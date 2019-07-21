import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils import rating

pd.options.mode.chained_assignment = None  # default='warn'

INPUT_FIXTURE_PATH = "../../data/processed/football_results.pkl"
OUTPUT_FIXTURE_PATH = "../../data/processed/matches_with_rating.pkl"
K_FACTOR = 10


def add_ratings(match_df, rate_target):

    elo_outcomes = rating.ELO(
        fixtures=match_df,
        target=rate_target,
        player_1="HomeTeam",
        player_2="AwayTeam",
        rater_name=rate_target,
        hyperparams={"k_factor": K_FACTOR},
    )

    matches_with_rating, ratings_latest = elo_outcomes.process_all_fixtures()
    rating_vals = matches_with_rating[
        [
            f"{rate_target}_rating_1",
            f"{rate_target}_rating_2",
            f"{rate_target}_rating_diff",
            f"{rate_target}_e",
        ]
    ]

    return rating_vals, ratings_latest


def main():
    match_df = (
        pd.read_pickle(INPUT_FIXTURE_PATH)
        .reset_index(drop=True)
        .rename(
            columns={
                "result_val": "match_outcome",
                "expit_goal_difference": "exp_goal_diff",
            }
        )
    )

    for t in ["match_outcome", "exp_goal_diff"]:
        rating_vals, ratings_latest = add_ratings(match_df, t)
        match_df = match_df.merge(rating_vals)
        with open(f"../../data/processed/ratings_{t}.pkl", "wb") as handle:
            pickle.dump(dict(ratings_latest), handle)
    elo_outcomes = rating.ELO(
        fixtures=match_df,
        target="result_val",
        player_1="HomeTeam",
        player_2="AwayTeam",
        rater_name="elo_outcome",
        hyperparams={"k_factor": K_FACTOR},
    )

    match_df.to_pickle(OUTPUT_FIXTURE_PATH)


if __name__ == "__main__":
    main()
