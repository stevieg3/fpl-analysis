import numpy as np
import logging
import lightgbm
import pickle
from sklearn.model_selection import RandomizedSearchCV

from src.models.utils import \
    _load_all_historical_data, \
    _map_season_string_to_ordered_numeric, \
    _generate_known_features_for_next_gw, _append_time_series_features
from src.models.constants import \
    LGBM_RANDOM_SEARCH_PARAMS

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def load_training_data():
    """
    Load all historical FPL player-gw data (2016-17 to 2018-19) and generate additional features

    :return: Pandas DataFrame with historical GW data and next GW features
    """

    fpl_all_historical = _load_all_historical_data()

    _map_season_string_to_ordered_numeric(fpl_all_historical)

    _generate_known_features_for_next_gw(fpl_all_historical)

    logging.info(f"Loaded historical data of shape {fpl_all_historical.shape}")

    return fpl_all_historical


def train_model(test_set_season, test_set_gw_start):
    """
    Train an LGBM Regressor using Randomised Search

    :param test_set_season: Season order of test set
    :param test_set_gw_start: First gw of test set season
    :return: Dictionary containing train and test data and instance of fitted RandomizedSearchCV
    """
    all_historical = load_training_data()
    all_historical_with_ts = _append_time_series_features(all_historical)

    all_historical_with_ts['in_test_set'] = np.where(
        (all_historical_with_ts['season_order'] == test_set_season) &
        (all_historical_with_ts['gw'] >= test_set_gw_start),
        1,
        0
    )

    all_historical_with_ts['total_points_plus_1_gw'] = all_historical_with_ts.groupby(
        ['name']
    )['total_points'].shift(-1)

    all_historical_with_ts = all_historical_with_ts[~all_historical_with_ts['total_points_plus_1_gw'].isnull()]

    training_data = all_historical_with_ts[(all_historical_with_ts['in_test_set'] == 0)]
    test_data = all_historical_with_ts[(all_historical_with_ts['in_test_set'] == 1)]

    training_data.drop('in_test_set', axis=1, inplace=True)
    test_data.drop('in_test_set', axis=1, inplace=True)

    logging.info(f"Training data shape: {training_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")

    assert all_historical_with_ts.shape[0] == training_data.shape[0] + test_data.shape[0]

    object_columns = list(training_data.select_dtypes('object').columns)
    logging.info(f"Object columns: {object_columns}")

    random_search_reg = RandomizedSearchCV(
        lightgbm.LGBMRegressor(),
        param_distributions=LGBM_RANDOM_SEARCH_PARAMS,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=3
    )

    logging.info(f"Starting random search...")

    random_search_reg.fit(
        training_data.drop(
            object_columns + ['total_points_plus_1_gw'],
            axis=1
        ),
        training_data['total_points_plus_1_gw']
    )

    logging.info(f"Finished random search!")

    return {
        'training_data': training_data,
        'test_data': test_data,
        'random_search_fitted': random_search_reg
    }


def main():
    """
    Train model using all data up from 2016-17 season to mid 2018-19 season. Save model as pickle and save train and
    test data as parquet files.

    :return: None
    """
    output_dict = train_model(test_set_season=3, test_set_gw_start=20)
    random_search_reg = output_dict['random_search_fitted']

    pickle.dump(
        random_search_reg.best_estimator_,
        open('src/models/pickles/v2_2_lgbm_point_predictor.pickle', 'wb')
    )

    logging.info(f"Best score: {random_search_reg.best_score_}")
    logging.info(f"Best params: {random_search_reg.best_params_}")

    training_data = output_dict['training_data']
    training_data.to_parquet('src/models/train_test_data/v_2_2_lgbm_training_data.parquet', index=False)

    test_data = output_dict['test_data']
    test_data.to_parquet('src/models/train_test_data/v_2_2_lgbm_test_data.parquet', index=False)
