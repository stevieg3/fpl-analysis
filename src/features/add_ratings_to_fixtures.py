import pandas as pd
import numpy as np
import pickle

PLAYERS = "../../data/raw/cleaned_players.pkl"
FIXTURES = "../../data/raw/fixtures.pkl"
TEAM_NUMBERS = "../../data/external/team_numbers.csv"
RATINGS = ["match_outcome", "exp_goal_diff"]
FIXTURES_WITH_RATINGS = "../../data/processed/fixtures_with_ratings_latest.pkl"
COLUMN_MAPPING = {
    "opp_team": "team",
    "team": "opp_team",
    "match_outcome_rating_1": "match_outcome_rating_2",
    "match_outcome_rating_2": "match_outcome_rating_1",
    "exp_goal_diff_rating_1": "exp_goal_diff_rating_2",
    "exp_goal_diff_rating_2": "exp_goal_diff_rating_1",
}


def elo_calculations(comb_fixtures, variable):
    comb_fixtures[f"{variable}_rating_diff"] = (
        comb_fixtures[f"{variable}_rating_1"] - comb_fixtures[f"{variable}_rating_2"]
    )

    comb_fixtures[f"{variable}_e"] = (
        10 ** (comb_fixtures[f"{variable}_rating_1"] / 400)
    ) / (
        (10 ** (comb_fixtures[f"{variable}_rating_1"] / 400))
        + (10 ** (comb_fixtures[f"{variable}_rating_2"] / 400))
    )

    return comb_fixtures


def add_ratings(players, fixtures, team_numbers, ratings):
    team_numbers = team_numbers.loc[
        team_numbers.season == "2018-19", ["team", "team_number"]
    ]

    fixtures_home_name = (
        fixtures.loc[:, ["event", "kickoff_time", "team_h", "team_a"]]
        .merge(team_numbers, left_on="team_h", right_on="team_number")
        .rename(columns={"team": "home_team"})
        .drop(columns="team_number")
    )

    fixtures_away_name = (
        fixtures_home_name.merge(team_numbers, left_on="team_a", right_on="team_number")
        .rename(columns={"team": "away_team"})
        .drop(columns="team_number")
        .sort_values(by="event")
    )

    for v in ratings:
        with open(f"../../data/processed/ratings_{v}.pkl", "rb") as handle:
            ratings_dict = pickle.load(handle)

        fixtures_away_name[f"{v}_rating_1"] = fixtures_away_name.home_team.map(
            ratings_dict
        )

        fixtures_away_name[f"{v}_rating_2"] = fixtures_away_name.away_team.map(
            ratings_dict
        )
        fixtures_away_name = elo_calculations(fixtures_away_name, v)

    fixtures_home = fixtures_away_name.drop(columns=["team_a", "team_h"]).rename(
        columns={"home_team": "team", "away_team": "opp_team", "event": "gw"}
    )

    fixtures_away = fixtures_home.copy().rename(columns=COLUMN_MAPPING)

    fixtures_home["is_home"] = True
    fixtures_away["is_home"] = False

    comb_fixtures = fixtures_home.append(fixtures_away, sort=True)

    return comb_fixtures


def main():
    players = pd.read_pickle(PLAYERS)

    fixtures = pd.read_pickle(FIXTURES)
    team_numbers = pd.read_csv(TEAM_NUMBERS)

    fixtures_with_ratings = add_ratings(players, fixtures, team_numbers, RATINGS)

    fixtures_with_ratings.to_pickle(FIXTURES_WITH_RATINGS)


if __name__ == "__main__":
    main()
