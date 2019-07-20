import pandas as pd
import numpy as np

PLAYERS = "../../data/raw/cleaned_players.pkl"
FIXTURES = "../../data/raw/fixtures.pkl"
TEAM_NUMBERS = "../../data/external/team_numbers.csv"
RATINGS = "../../data/processed/ratings_latest.pkl"
FIXTURES_WITH_RATINGS = "../../data/processed/ratings_latest.pkl"


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

    fixtures_away_name["elo_outcome_rating_1"] = fixtures_away_name.home_team.map(
        ratings
    )

    fixtures_away_name["elo_outcome_rating_2"] = fixtures_away_name.away_team.map(
        ratings
    )

    fixtures_away_name["elo_outcome_rating_diff"] = (
        fixtures_away_name.elo_outcome_rating_1
        - fixtures_away_name.elo_outcome_rating_2
    )

    fixtures_away_name["elo_outcome_rating_e"] = (
        10 ** (fixtures_away_name["elo_outcome_rating_1"] / 400)
    ) / (
        (10 ** (fixtures_away_name["elo_outcome_rating_1"] / 400))
        + (10 ** (fixtures_away_name["elo_outcome_rating_2"] / 400))
    )

    return fixtures_away_name


def main():
    players = pd.read_pickle(PLAYERS)

    fixtures = pd.read_pickle(FIXTURES)
    team_numbers = pd.read_csv(TEAM_NUMBERS)
    ratings = pd.read_pickle(RATINGS)

    fixtures_with_ratings = add_ratings(players, fixtures, team_numbers, ratings)

    fixtures_with_ratings.to_pickle(FIXTURES_WITH_RATINGS)


if __name__ == "__main__":
    main()
