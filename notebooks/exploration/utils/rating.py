import pandas as pd
from collections import defaultdict
import math
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import defaultdict
import math
from copy import deepcopy
import numpy as np
import trueskill
from scipy.stats import norm as sc_norm


class Rater:
    def __init__(
        self,
        fixtures,
        target,
        player_1="player_1",
        player_2="player_2",
        initial_ratings={},
        rater_name="",
        hyperparams={},
    ):
        self.fixtures = fixtures
        self.target = target
        self._create_ratings(initial_ratings)
        self.hyperparams = hyperparams
        self.rater_name = rater_name
        self.player_1 = player_1
        self.player_2 = player_2

    def _record_expectation(self, i, fixture):
        rating_1, rating_2, score_1, score_2 = self._get_player_ratings(fixture)
        self.fixtures.at[i, f"{self.rater_name}_rating_1"] = score_1
        self.fixtures.at[i, f"{self.rater_name}_rating_2"] = score_2
        self.fixtures.at[i, f"{self.rater_name}_rating_diff"] = score_1 - score_2
        self.fixtures.at[i, f"{self.rater_name}_e"] = self._win_probability(
            rating_1, rating_2
        )

    def process_all_fixtures(self):
        for i, f in self.fixtures.iterrows():
            if not math.isnan(f[self.target]):
                self._record_expectation(i, f)
                self._process_fixture(f)

        return self.fixtures, self.ratings

    def _create_ratings(self, initial_ratings):
        self.ratings = defaultdict(lambda: 1200)
        self.ratings.update(initial_ratings)

    def _get_player_ratings(self, fixture):
        rating_1 = self.ratings[fixture[self.player_1]]
        rating_2 = self.ratings[fixture[self.player_2]]
        return rating_1, rating_2, rating_1, rating_2

    def _process_fixture(self, fixture):
        pass
        # needs to be implemented by specific rating class
        # self.ratings[fixture.player_1] = rating_1
        # self.ratings[fixture.player_2] = rating_2

    def _win_probability(self, player_1, player_2):
        pass
        # needs to be implemented by specific rating class
        # return player_1_winning_prob


class TrueSkill(Rater):
    def _create_ratings(self, initial_ratings):
        self.ratings = defaultdict(lambda: trueskill.Rating())
        self.ratings.update(initial_ratings)

    def _get_player_ratings(self, fixture):
        rating_1 = self.ratings[fixture[self.player_1]]
        rating_2 = self.ratings[fixture[self.player_2]]
        score_1 = rating_1.mu
        score_2 = rating_2.mu
        return rating_1, rating_2, score_1, score_2

    def _process_fixture(self, fixture):
        rating_1, rating_2, _, _ = self._get_player_ratings(fixture)
        if fixture[self.target] == 1:
            update_rating_1, update_rating_2 = trueskill.rate_1vs1(rating_1, rating_2)
        else:
            update_rating_2, update_rating_1 = trueskill.rate_1vs1(rating_2, rating_1)
        self.ratings[fixture[self.player_1]] = update_rating_1
        self.ratings[fixture[self.player_2]] = update_rating_2

    def _win_probability(self, rating_1, rating_2):
        delta_mu = rating_1.mu - rating_2.mu
        delta_sigma = math.sqrt((rating_1.sigma ** 2) + (rating_2.sigma ** 2))
        return sc_norm.cdf(delta_mu / delta_sigma)


class HeadToHead(Rater):
    def _create_ratings(self, initial_ratings):
        ratings = {}
        unique_players = np.unique(self.fixtures[[self.player_1, self.player_2]].values)
        for p_1 in unique_players:
            ratings[p_1] = {}
            for p_2 in unique_players:
                ratings[p_1][p_2] = 0
        # todo if needed, add initial ratings option
        self.ratings = ratings

    def _get_player_ratings(self, fixture):
        rating_1 = self.ratings[fixture[self.player_1]][fixture[self.player_2]]
        rating_2 = self.ratings[fixture[self.player_2]][fixture[self.player_1]]
        return rating_1, rating_2, rating_1, rating_2

    def _process_fixture(self, fixture):
        if fixture[self.target] == 0:
            self.ratings[fixture[self.player_1]][fixture[self.player_2]] += 1
        elif fixture[self.target] == 0.5:
            self.ratings[fixture[self.player_1]][fixture[self.player_2]] += 0.5
            self.ratings[fixture[self.player_2]][fixture[self.player_1]] += 0.5
        else:
            self.ratings[fixture[self.player_2]][fixture[self.player_1]] += 1

    def _win_probability(self, rating_1, rating_2):
        if rating_1 == 0 and rating_2 == 0:
            return np.nan
        else:
            return rating_1 / (rating_1 + rating_2)


class ELO(Rater):
    def _process_fixture(self, fixture):
        rating_1, rating_2, _, _ = self._get_player_ratings(fixture)
        win_prob = self._win_probability(rating_1, rating_2)
        self.ratings[fixture[self.player_1]] = rating_1 + (
            (fixture[self.target] - win_prob) * self.hyperparams["k_factor"]
        )
        self.ratings[fixture[self.player_2]] = rating_2 + (
            (win_prob - fixture[self.target]) * self.hyperparams["k_factor"]
        )

    def _win_probability(self, rating_1, rating_2):
        q1 = 10.0 ** (rating_1 / 400)
        q2 = 10.0 ** (rating_2 / 400)
        return q1 / (q1 + q2)
