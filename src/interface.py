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


def fpl_scorer(previous_gw, prediction_season_order, live_run=False, double_gw_teams=[]):
    # TODO Docstring
    if live_run:
        full_data = load_live_data(previous_gw=previous_gw, save_file=True)
    else:
        # TODO Smarter way of finding latest file:
        full_data = load_retro_data(current_season_data_filepath='data/gw_player_data/gw_29_player_data.parquet')

    lstm_pred = LSTMPlayerPredictor(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order
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
    final_predictions['season'] = reversed_season_order_dict[prediction_season_order]
    final_predictions['gw'] = previous_gw + 1  # TODO Need to account for GW 38
    final_predictions['model'] = 'lstm_v4'

    if live_run:
        write_dataframe_to_s3(
            df=final_predictions,
            s3_root_path=S3_BUCKET_PATH + GW_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw']
        )
    else:
        write_dataframe_to_s3(
            df=final_predictions,
            s3_root_path=S3_BUCKET_PATH + GW_RETRO_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw']
        )
