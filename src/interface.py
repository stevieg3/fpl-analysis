from src.models.LSTM.make_predictions import \
    load_live_data, \
    load_retro_data, \
    LSTMPlayerPredictor


def fpl_scorer(previous_gw, prediction_season_order, live_run=False):
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

    gw_prediction_data = lstm_pred.prepare_data_for_lstm(full_data=full_data)
    final_predictions = lstm_pred.make_player_predictions(gw_prediction_data=gw_prediction_data)

    if live_run:
        final_predictions.to_parquet(
            f'data/gw_predictions/gw{previous_gw+1}_v4_lstm_player_predictions.parquet',
            index=False
        )
    else:
        final_predictions.to_parquet(
            'data/gw_retro_predictions/gw{}_season_{}_v4_lstm_player_predictions.parquet'.format(
                previous_gw+1,
                prediction_season_order
            ),
            index=False
        )
