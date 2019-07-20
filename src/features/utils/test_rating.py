import pytest

import rating
import pandas as pd

RESULTS_DF = pd.DataFrame(
    {
        "player_1": ["A", "B", "A", "C"],
        "player_2": ["B", "C", "C", "A"],
        "result": [0, 1, 1, 0.5],
    }
)


@pytest.fixture
def player_fixture():
    """
    Fixture - a single match fixture
    """
    return RESULTS_DF.iloc[1]


@pytest.fixture
def ELO_model():
    """
    Fixture - an ELO rater class
    """
    return rating.ELO(
        fixtures=RESULTS_DF,
        target="result",
        player_1="player_1",
        player_2="player_2",
        rater_name="outcome",
        hyperparams={"k_factor": 10},
    )


def test_rater_creation():
    """
    Check that we can successfully create ELO rater
    """
    rating.ELO(
        fixtures=RESULTS_DF,
        target="result",
        player_1="player_1",
        player_2="player_2",
        rater_name="outcome",
        hyperparams={"k_factor": 10},
    )


def test_win_probability(ELO_model):
    """
    Check that _win_probability outputs match offline calculations
    """
    assert ELO_model._win_probability(rating_1=100, rating_2=100) == 0.5
    assert ELO_model._win_probability(rating_1=1200, rating_2=1200) == 0.5
    win_prob_1 = ELO_model._win_probability(rating_1=1100, rating_2=1000)
    win_prob_2 = ELO_model._win_probability(rating_1=950, rating_2=1400)
    pytest.approx(win_prob_1, 0.640065)
    pytest.approx(win_prob_2, 0.069758287)


def test_get_player_ratings(ELO_model, player_fixture):
    """
    Check that get_player_ratings return correct ratings
    """

    assert ELO_model._get_player_ratings(player_fixture) == (1200, 1200, 1200, 1200)
    ELO_model.ratings["C"] = 1234
    assert ELO_model._get_player_ratings(player_fixture) == (1200, 1234, 1200, 1234)


def test_process_fixture(ELO_model, player_fixture):
    """
    Check that process_fixture leads to same rating updates as offline calculation
    """
    ELO_model._process_fixture(player_fixture)
    rating_1 = ELO_model.ratings[player_fixture[ELO_model.player_1]]
    rating_2 = ELO_model.ratings[player_fixture[ELO_model.player_2]]
    pytest.approx(rating_1, 1205)
    pytest.approx(rating_2, 1995)
    ELO_model._process_fixture(player_fixture)
    rating_1_new = ELO_model.ratings[player_fixture[ELO_model.player_1]]
    rating_2_new = ELO_model.ratings[player_fixture[ELO_model.player_2]]
    pytest.approx(rating_1_new, 1209.856128)
    pytest.approx(rating_2_new, 1190.143872)


def test_process_all_fixtures(ELO_model):
    """
    Check that process_all_fixtures leads to same rating updates as offline calculation
    """

    results, ratings = ELO_model.process_all_fixtures()
    pytest.approx(ratings["A"], 1199.858168439664)
    pytest.approx(ratings["B"], 1209.9280491829095)
    pytest.approx(ratings["C"], 1190.2137823774265)
    assert results.shape == (4, 7)
