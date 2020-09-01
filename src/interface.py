import logging

import pandas as pd

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
        logging.info("First gameweek prediction so checking for positional changes")
        # Account for position changes between seasons
        # Note: Predictions are made using past position but new position needed for team selection
        make_position_changes(
            df_with_old_positions=final_predictions,
            season_1=reversed_season_order_dict[prediction_season_order],
            season_2=reversed_season_order_dict[prediction_season_order+1],
            live_run=live_run,
            previous_gw=previous_gw
        )
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


def make_position_changes(df_with_old_positions, season_1, season_2, live_run, previous_gw=None):
    """
    Some players change position at the start of a new season. This can lead to issues with team selection because
    predictions for GW 1 will use the previous season's position. Therefore the team selection criteria may be
    breached when making the team selection for GW 2.

    This function identifies players who changed position between seasons and changes the position for these players in
    a provided DataFrame.

    :param df_with_old_positions: DataFrame which uses positions from season_1 e.g. predictions for GW 1
    :param season_1: First season
    :param season_2: Consecutive season
    :param live_run: Boolean. Live or retro predictions
    :param previous_gw: Previous GW
    :return: Modifies in-place
    """
    if live_run:
        gw_data = load_live_data(previous_gw=previous_gw, save_file=False)
    else:
        # TODO Smarter way of finding latest file:
        gw_data = load_retro_data(current_season_data_filepath='data/gw_player_data/gw_37_player_data.parquet')

    names_and_pos_season_1 = gw_data[gw_data['season'] == season_1][
        ['name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID']].drop_duplicates()

    names_and_pos_season_1.loc[names_and_pos_season_1['position_DEF'] == 1, 'position'] = 'DEF'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_FWD'] == 1, 'position'] = 'FWD'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_GK'] == 1, 'position'] = 'GK'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_MID'] == 1, 'position'] = 'MID'

    names_and_pos_season_2 = gw_data[gw_data['season'] == season_2][
        ['name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID']].drop_duplicates()

    names_and_pos_season_2.loc[names_and_pos_season_2['position_DEF'] == 1, 'position'] = 'DEF'
    names_and_pos_season_2.loc[names_and_pos_season_2['position_FWD'] == 1, 'position'] = 'FWD'
    names_and_pos_season_2.loc[names_and_pos_season_2['position_GK'] == 1, 'position'] = 'GK'
    names_and_pos_season_2.loc[names_and_pos_season_2['position_MID'] == 1, 'position'] = 'MID'

    comp = names_and_pos_season_1[['name', 'position']].merge(
        names_and_pos_season_2[['name', 'position']],
        on='name',
        how='inner',
        suffixes=('_season_1', '_season_2')
    )

    position_changes = comp[comp['position_season_1'] != comp['position_season_2']]

    position_changes.rename(columns={'position_season_2': 'position'}, inplace=True)
    position_changes.drop(columns=['position_season_1'], inplace=True)
    position_changes = pd.get_dummies(position_changes, columns=['position'])

    logging.info(f"Players who changed position between {season_1} and {season_2}: {set(position_changes['name'])}")

    for _, row in position_changes.iterrows():
        df_with_old_positions.loc[
            (df_with_old_positions['name'] == row['name']),
            ['position_DEF', 'position_FWD', 'position_MID']
        ] = row['position_DEF'], row['position_FWD'], row['position_MID']
