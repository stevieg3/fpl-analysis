import logging

from src.models.LSTM.make_predictions import \
    load_live_data, \
    load_retro_data, \
    LSTMPlayerPredictor
from src.models.constants import \
    SEASON_ORDER_DICT
from src.data.s3_utilities import \
    S3_BUCKET_PATH, \
    GW_PREDICTIONS_SUFFIX, \
    GW_RETRO_PREDICTIONS_SUFFIX, \
    write_dataframe_to_s3

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def fpl_scorer(
        previous_gw,
        prediction_season_order,
        live_run=False,
        double_gw_teams=[],
        previous_gw_was_double_gw=False
):
    """
    Main function for calculating end-to-end player predictions. Predictions saved to S3.

    :param previous_gw: Gameweek prior to the one you want to make a selection for
    :param prediction_season_order: Season order number (2019/20 season is 4). Season order number for the previous_gw
    i.e. for gameweek 1 predictions you will need to provide the season order number for the previous season
    :param live_run: Boolean. True if using live data or False for retro scoring
    :param double_gw_teams: List of teams with a double gameweek in the next gameweek. Next gameweek predictions for
    these teams are multiplied by 2
    :param previous_gw_was_double_gw: Boolean. True if previous gameweek was a double gameweek. Required to remove
    duplicate player entries

    :return: None. Output saved to S3
    """
    if live_run:
        full_data = load_live_data(previous_gw=previous_gw, save_file=True)
    else:
        # TODO Smarter way of finding latest file:
        full_data = load_retro_data(current_season_data_filepath='data/gw_player_data/gw_37_player_data.parquet')

    lstm_pred = LSTMPlayerPredictor(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order,
        previous_gw_was_double_gw=previous_gw_was_double_gw
    )

    player_list, player_data_list = lstm_pred.prepare_data_for_lstm(full_data=full_data)

    unformatted_predictions = lstm_pred.make_player_predictions(
        player_data_list=player_data_list
    )

    final_predictions = lstm_pred.format_predictions(
        player_list=player_list,
        final_predictions=unformatted_predictions,
        full_data=full_data,
        double_gw_teams=double_gw_teams
    )

    reversed_season_order_dict = {v: k for k, v in SEASON_ORDER_DICT.items()}
    if previous_gw == 38:
        prediction_season_order += 1  # Roll over to next season
        final_predictions['season'] = reversed_season_order_dict[prediction_season_order]
        final_predictions['gw'] = 1
    else:
        final_predictions['season'] = reversed_season_order_dict[prediction_season_order]
        final_predictions['gw'] = previous_gw + 1
    final_predictions['model'] = 'lstm_v4'

    logging.info(final_predictions.head())

    if live_run:
        write_dataframe_to_s3(
            df=final_predictions,
            s3_root_path=S3_BUCKET_PATH + GW_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw']
        )
        logging.info('Saved live prediction data to S3')
    else:
        write_dataframe_to_s3(
            df=final_predictions,
            s3_root_path=S3_BUCKET_PATH + GW_RETRO_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw']
        )
        logging.info('Saved retro prediction data to S3')
