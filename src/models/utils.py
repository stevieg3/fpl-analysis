import pandas as pd
import numpy as np
import logging
import pickle
import time

from src.data.live_season_data import GetFPLData
from src.features.custom_transformers import TimeSeriesFeatures
from src.models.constants import \
    FPL_AVAILABLE_FEATURES_19_20, \
    SEASON_ORDER_DICT, \
    KNOWN_FEATURES_NEXT_GW, \
    KICKOFF_MONTH_FEATURES, TIME_SERIES_FEATURES_19_20

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def _load_all_historical_data(available_features=FPL_AVAILABLE_FEATURES_19_20):
    """
    Load all available features in current API for historical FPL player-gw data (2016-17 to 2018-19)

    :param available_features: Features to include based on what is available in latest API
    :return: Pandas DataFrame
    """
    logging.info("Loading raw historical FPL data")
    with open(r'data/processed/fpl_data_2016_to_2020.parquet', 'rb') as f:
        fpl_all_historical = pd.read_parquet(f, engine='pyarrow')

    return fpl_all_historical[available_features]


def _map_season_string_to_ordered_numeric(gw_dataframe):
    """
    Creates numeric season_order column to allow sorting by season

    :param gw_dataframe: Player-GW data containing a string season column
    :return: None (modifies in-place)
    """
    logging.info("Creating season order column")
    gw_dataframe['season_order'] = gw_dataframe['season'].map(SEASON_ORDER_DICT)


def _generate_known_features_for_next_gw(gw_dataframe):
    """
    Create features for current GW based on what is known about the next GW

    :param gw_dataframe: Player-GW data containing a string season column
    :return: None (modifies in-place)
    """
    logging.info("Generating known features for next GW")
    assert all(feature in gw_dataframe.columns for feature in KNOWN_FEATURES_NEXT_GW)

    gw_dataframe.sort_values(['name', 'season_order', 'gw'], inplace=True)
    gw_dataframe.reset_index(drop=True, inplace=True)

    for feature in KNOWN_FEATURES_NEXT_GW:
        gw_dataframe[f'next_match_{feature}'] = gw_dataframe.groupby('name')[f'{feature}'].shift(-1)
        gw_dataframe[f'next_match_{feature}'] = pd.to_numeric(gw_dataframe[f'next_match_{feature}'])

    gw_dataframe[['next_match_' + x for x in KICKOFF_MONTH_FEATURES]] = \
        gw_dataframe[['next_match_' + x for x in KICKOFF_MONTH_FEATURES]].fillna(0)


def _load_current_season_data(previous_gw, save_file=False):
    """
    Load current season data (all gameweeks) from API and save as parquet

    :param previous_gw: GW which has just finished. The parquet file saved will be named using this value
    :param save_file: If True save as parquet file
    :return: Pandas DataFrame
    """
    get_fpl_data = GetFPLData(season='2019-20')
    current_gw_data = get_fpl_data.get_all_gameweek_data_from_api()

    if save_file:
        # TODO Save to S3
        logging.info(f"Writing current data as of gw {previous_gw} to gw_{previous_gw}_player_data.parquet")
        current_gw_data.to_parquet(f'data/gw_player_data/gw_{previous_gw}_player_data.parquet', index=False)

    # Create empty features for any months where there have been no matches yet
    for col in list(set(KICKOFF_MONTH_FEATURES) - set(current_gw_data.columns)):
        current_gw_data[col] = 0

    return current_gw_data


def _load_next_fixture_data(next_gw):
    """
    Load fixture data for `next_gw`

    :param next_gw: Next Gameweek (must be a Gameweek which hasn't occurred yet otherwise will not be present in API)
    :return: Pandas DataFame
    """
    get_fpl_data = GetFPLData(season='2020-21')
    upcoming_fixtures = get_fpl_data.get_all_fixture_data_from_api()

    next_fixture_data = upcoming_fixtures[upcoming_fixtures['gw'] == next_gw]

    # Only keep columns which are available in current
    static_columns = list(
        set(FPL_AVAILABLE_FEATURES_19_20).intersection(set(next_fixture_data.columns))
    )

    next_fixture_data = next_fixture_data[static_columns]
    next_fixture_data.drop('minutes', inplace=True, axis=1)  #  This is total minutes played so far in season

    return next_fixture_data


def _load_model_from_pickle(model_filepath):
    """
    Load model from pickle file

    :param model_filepath: File path containing pickled model
    :return: model
    """
    model = pickle.load(open(model_filepath, 'rb'))
    return model


def _append_time_series_features(
        dataframe,
        ts_halflife=4,
        ts_max_lag=4,
        ts_max_diff=4,
        ts_columns=TIME_SERIES_FEATURES_19_20
):
    """
    Append time series features based on `ts_columns`

    :param dataframe: Pandas DataFrame containing player-gw data
    :param ts_halflife: `halflife` for `TimeSeriesFeatures` instance
    :param ts_max_lag: `max_lag` for `TimeSeriesFeatures` instance
    :param ts_max_diff: `max_diff` for `TimeSeriesFeatures` instance
    :param ts_columns: `columns` for `TimeSeriesFeatures` instance

    :return: Pandas DataFrame with additional time series features
    """
    # Create ID column for each name to allow time series features to be generated
    dataframe.drop('ID', axis=1, inplace=True)
    id_df = dataframe.groupby(['name']).count().reset_index()[['name']]
    id_df['ID'] = id_df.index + 1

    dataframe = dataframe.merge(id_df, how='left', on=['name'])

    # Generate time series features
    dataframe.sort_values(['name', 'season_order', 'gw'], inplace=True)

    for col in TIME_SERIES_FEATURES_19_20:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

    ts_feature_generator = TimeSeriesFeatures(
        halflife=ts_halflife,
        max_lag=ts_max_lag,
        max_diff=ts_max_diff,
        columns=ts_columns
    )

    logging.info(f"Generating time series features...")

    start = time.time()

    fpl_data_all_seasons_with_ts = ts_feature_generator.fit_transform(dataframe)
    fpl_data_all_seasons_with_ts.drop('ID', axis=1, inplace=True)

    end = time.time()

    logging.info(f"Finished generating time series features (Duration: {np.round(end-start, 2)}s)")

    return fpl_data_all_seasons_with_ts


def round_down_to_nearest_10th(num):
    """
    Round number down to nearest 10th
    :param num: Number
    :return: Number rounded down to nearest 10th
    """
    return np.floor(num*10) / 10

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# LSTM UTILITIES                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


def split_sequences(df, target_column, n_steps_in, n_steps_out):
    """
    Split Pandas DataFrame containing multivariate time series data and target variable into samples for a multiple
    input multi-step output LSTM.

    Adapted from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

    :param df: DataFrame containing ordered time series features and target variable which we are trying to predict
    :param target_column: Name of target column
    :param n_steps_in: Number of input time steps to use
    :param n_steps_out: Number of output time steps to predict
    :return: X of dim (samples, timesteps, features) and y of dim (samples, output steps)
    """
    assert df.apply(lambda s: pd.to_numeric(s, errors='coerce').isnull().all()).sum() == 0

    # Move target column to end of DataFrame
    df = df.copy()[[col for col in df if col not in [target_column]] + [target_column]]

    sequences = df.values  # DataFrame to NumPy array

    X, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def custom_train_test_split(full_data, random_state, rand_sample_prop=0.005):
    """
    Split player-gameweek data into training and test set. Data is in panel data format so test sets are the end GWs for
    randomly selected players (i.e. preserving the order). The test set for each randomly selected player can vary in
    length. The total proportion of test data should be approximately equal to 20% (when each name is a separate series)
    given the rand_sample_prop of 0.5%. Note: Data includes total points column.

    :param full_data: Player-GW data
    :param rand_sample_prop: Proportion of rows to select
    :return: Train and test splits
    """
    full_data = full_data.copy()
    # Randomly assign 1s to rows (player-GW data points) from DataFrame
    np.random.seed(random_state)
    full_data['in_test_set'] = np.random.choice(
        [np.nan, 1],
        size=(full_data.shape[0],),
        p=[1-rand_sample_prop, rand_sample_prop])

    # Forward fill so that all future GWs for a randomly selected player are also part of test set
    full_data['in_test_set'] = full_data.groupby(['name'])['in_test_set'].fillna(method='ffill')
    full_data['in_test_set'] = full_data['in_test_set'].fillna(0)

    print(f"Proportion in test set: {full_data['in_test_set'].sum() / full_data.shape[0]}")

    test_set = full_data[full_data['in_test_set'] == 1]
    print(f"Test set size: {test_set.shape}")
    training_set = full_data[full_data['in_test_set'] == 0]
    print(f"Training set size: {training_set.shape}")

    test_set.drop('in_test_set', axis=1, inplace=True)
    training_set.drop('in_test_set', axis=1, inplace=True)

    return training_set, test_set
