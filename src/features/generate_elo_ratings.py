import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils import rating

pd.options.mode.chained_assignment = None  # default='warn'

INPUT_FIXTURE_PATH = "../../data/processed/football_results.pkl"
OUTPUT_FIXTURE_PATH = "../../data/processed/matches_with_rating.pkl"
OUTPUT_RATINGS_PATH = "../../data/processed/ratings_latest.pkl"
K_FACTOR = 10


def main():
    match_df = pd.read_pickle(INPUT_FIXTURE_PATH).reset_index(drop=True)

    match_df["start_year"] = np.where(
        match_df.Date.dt.month < 6, match_df.Date.dt.year - 1, match_df.Date.dt.year
    )

    elo_outcomes = rating.ELO(
        fixtures=match_df,
        target="result_val",
        player_1="HomeTeam",
        player_2="AwayTeam",
        rater_name="elo_outcome",
        hyperparams={"k_factor": K_FACTOR},
    )

    matches_with_rating, ratings_latest = elo_outcomes.process_all_fixtures()

    matches_with_rating[
        [
            "HomeTeam",
            "AwayTeam",
            "Date",
            "FTHG",
            "FTAG",
            "elo_outcome_rating_1",
            "elo_outcome_rating_2",
            "elo_outcome_rating_diff",
            "elo_outcome_e",
        ]
    ].to_pickle(OUTPUT_FIXTURE_PATH)

    with open(OUTPUT_RATINGS_PATH, "wb") as handle:
        pickle.dump(dict(ratings_latest), handle)


if __name__ == "__main__":
    main()
