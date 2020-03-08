import logging
from src.models.utils import \
    _load_all_historical_data, \
    _map_season_string_to_ordered_numeric, \
    _generate_known_features_for_next_gw, \
    _load_current_season_data, \
    _load_next_fixture_data, \
    _load_model_from_pickle, \
    _append_time_series_features
from src.models.constants import \
    LGBM_PREDICTION_COLS_TO_EXCLUDE, \
    LGBM_PICKLE_PATH, \
    PREDICTIONS_OUTPUT_COLUMNS

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def _load_input_data(previous_gw, save_file=False):
    """
    Load all input data required for predictions: historical, current season, next fixture. Also creates any additional
    features

    :param previous_gw: Previous gameweek
    :param save_file: Set to True to save current season data as parquet
    :return: DataFrame containing all required inputs for all gameweeks
    """
    # Load historical
    raw_historical_df = _load_all_historical_data()
    logging.info(f"Loaded historical data of shape: {raw_historical_df.shape}")

    # Load current
    logging.info(f"Loading current season data (up to gw {previous_gw})...")
    current_gw_df = _load_current_season_data(previous_gw=previous_gw, save_file=save_file)
    logging.info(f"Loaded current season data (up to gw {previous_gw}) of shape: {current_gw_df.shape}")

    assert len(set(current_gw_df.columns) - set(raw_historical_df.columns)) == 0

    # Load next
    logging.info(f"Loading next fixture (gw {previous_gw + 1}) data...")
    next_gw_df = _load_next_fixture_data(next_gw=previous_gw+1)
    logging.info(f"Loaded next fixture (gw {previous_gw+1}) data of shape: {next_gw_df.shape}")

    # Combine data
    input_data = raw_historical_df.append(current_gw_df, sort=False)
    input_data = input_data.append(next_gw_df, sort=False)
    logging.info(f"Combined data shape: {input_data.shape}")

    # Additional features
    _map_season_string_to_ordered_numeric(input_data)

    input_data.sort_values(['name', 'season_order', 'gw'], inplace=True)
    input_data.reset_index(drop=True, inplace=True)

    _generate_known_features_for_next_gw(input_data)

    input_data_with_ts = _append_time_series_features(input_data)

    logging.info(f"Final input shape: {input_data_with_ts.shape}")

    return input_data_with_ts


def _write_predictions_to_parquet(full_predictions_df, gw_predictions):
    """
    Save predictions as parquet

    :param full_predictions_df: Full prediction dataframe output
    :param gw_predictions: GW you are predicting for. Used for naming parquet file
    :return: None
    """
    predictions_output = full_predictions_df.copy()
    predictions_output.sort_values('predictions', ascending=False, inplace=True)
    predictions_output = predictions_output[PREDICTIONS_OUTPUT_COLUMNS]
    predictions_output.to_parquet(
        f'data/gw_predictions/gw{gw_predictions}_v2.2_player_predictions.parquet',
        index=False
    )


def make_predictions(previous_gw, prediction_season_order, save_file, input_df=None):
    """
    Make predictions for previous_gw + 1 in prediction_season

    :param previous_gw: Previous gameweek
    :param prediction_season_order: Season you are predicting for
    :param save_file: Save current game week data and predictions. Overwrites existing files
    :param input_df: DataFrame of required inputs
    :return: Full dataframe containing predictions
    """

    logging.info(f"Loading input data...")

    try:
        full_data_for_prediction = input_df.copy()
    except AttributeError:
        full_data_for_prediction = _load_input_data(previous_gw=previous_gw, save_file=save_file)

    logging.info(f"Creating target variable...")

    full_data_for_prediction['total_points_plus_1_gw'] = \
        full_data_for_prediction.groupby(
            ['name']
        )['total_points'].shift(-1)

    gw_prediction_data = full_data_for_prediction.copy()[
        (full_data_for_prediction['gw'] == previous_gw) &
        (full_data_for_prediction['season_order'] == prediction_season_order)
    ]

    new_lgbm = _load_model_from_pickle(LGBM_PICKLE_PATH)

    logging.info(f"Making predictions...")

    gw_prediction_data['predictions'] = new_lgbm.predict(
        gw_prediction_data.drop(
            LGBM_PREDICTION_COLS_TO_EXCLUDE + ['total_points_plus_1_gw'],
            axis=1
        )
    )

    gw_prediction_data.sort_values('predictions', ascending=False, inplace=True)
    gw_prediction_data.reset_index(drop=True, inplace=True)

    logging.info(
        f"Predictions complete! " +
        f"{gw_prediction_data.head(1)['name'].item()} is predicted to do best in GW{previous_gw+1}!"
    )

    if save_file:
        logging.info(f"Saving predictions as parquet...")
        _write_predictions_to_parquet(gw_prediction_data, previous_gw+1)

    return gw_prediction_data
