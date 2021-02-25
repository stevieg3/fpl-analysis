import logging
import unidecode
import difflib

import pandas as pd
import numpy as np

from src.models.LSTM import make_predictions as old_model_make_predictions
from src.models.DeepFantasyFootball import make_predictions as new_model_make_predictions
from src.models.constants import \
    SEASON_ORDER_DICT
from src.data.s3_utilities import \
    S3_BUCKET_PATH, \
    GW_PREDICTIONS_SUFFIX, \
    GW_RETRO_PREDICTIONS_SUFFIX, \
    write_dataframe_to_s3

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


CURRENT_SEASON_FPL = '2020-21'


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
    # 1. Old model using FPL features
    logging.info('Making predictions using FPL model')
    if live_run:
        full_data = old_model_make_predictions.load_live_data(previous_gw=previous_gw, save_file=True)
    else:
        # TODO Smarter way of finding latest file:
        full_data = old_model_make_predictions.load_retro_data(
            current_season_data_filepath='data/gw_player_data/gw_37_player_data.parquet'
        )
        full_data['team_name_ffs'] = ''

    # Find teams who did not play in the previous gameweek
    teams_not_in_previous_gw = []
    if previous_gw != 38:
        teams_in_previous_gw = set(
            full_data[
                (full_data['gw'] == previous_gw) &
                (full_data['season_order'] == prediction_season_order)
            ]['team_name']
        )
        teams_in_next_gw = set(
            full_data[
                (full_data['gw'] == previous_gw + 1) &
                (full_data['season_order'] == prediction_season_order)
                ]['team_name']
        )
        teams_not_in_previous_gw = list(teams_in_next_gw - teams_in_previous_gw)
        logging.info(f'Teams not in previous gameweek: {teams_not_in_previous_gw}')

    if len(teams_not_in_previous_gw) > 0:
        _include_players_missing_in_previous_gameweek(
            full_data,
            previous_gw,
            prediction_season_order,
            teams_not_in_previous_gw,
            use_fantasy_football_scout_team_names=False
        )

    lstm_pred = old_model_make_predictions.LSTMPlayerPredictor(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order,
        previous_gw_was_double_gw=previous_gw_was_double_gw
    )

    player_list, player_data_list = lstm_pred.prepare_data_for_lstm(full_data=full_data)

    unformatted_predictions = lstm_pred.make_player_predictions(
        player_data_list=player_data_list
    )

    old_model_predictions = lstm_pred.format_predictions(
        player_list=player_list,
        final_predictions=unformatted_predictions,
        full_data=full_data,
        double_gw_teams=double_gw_teams
    )
    logging.info('Made predictions using FPL model')
    logging.info(old_model_predictions.head())

    # 2. New model using FFS data
    logging.info('Making predictions using Fantasy Football Scout model')
    if live_run:
        full_data = new_model_make_predictions.load_live_data()
    else:
        full_data = new_model_make_predictions.load_retro_data()

    # Adjustment for missing teams
    if len(teams_not_in_previous_gw) > 0:
        _include_players_missing_in_previous_gameweek(
            full_data,
            previous_gw,
            prediction_season_order,
            teams_not_in_previous_gw,
            use_fantasy_football_scout_team_names=True
        )

    deep_fantasy_football = new_model_make_predictions.DeepFantasyFootball(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order
    )

    player_list, player_data_list = deep_fantasy_football.prepare_data_for_lstm(full_data=full_data)

    unformatted_predictions = deep_fantasy_football.make_player_predictions(
        player_data_list=player_data_list
    )

    final_predictions = deep_fantasy_football.format_predictions(
        player_list=player_list,
        final_predictions=unformatted_predictions,
        full_data=full_data,
        double_gw_teams=[]
    )
    logging.info('Made predictions using Fantasy Football Scout model')
    logging.info(final_predictions.head())

    # 3. Merge predictions
    combined = merge_model_predictions(old_model_predictions=old_model_predictions, final_predictions=final_predictions)
    logging.info('Merged predictions from FPL model and Fantasy Football Scout model')

    reversed_season_order_dict = {v: k for k, v in SEASON_ORDER_DICT.items()}
    if previous_gw == 38:
        logging.info("First gameweek prediction so checking for positional changes")
        # Account for position changes between seasons
        # Note: Predictions are made using past position but new position needed for team selection
        make_position_changes(
            df_with_old_positions=combined,
            season=reversed_season_order_dict[prediction_season_order + 1],
            live_run=live_run,
            previous_gw=previous_gw
        )
        prediction_season_order += 1  # Roll over to next season
        combined['season'] = reversed_season_order_dict[prediction_season_order]
        combined['gw'] = 1
    else:
        combined['season'] = reversed_season_order_dict[prediction_season_order]
        combined['gw'] = previous_gw + 1

    logging.info('Final prediction output:')
    logging.info(combined.head())

    if live_run:
        write_dataframe_to_s3(
            df=combined,
            s3_root_path=S3_BUCKET_PATH + GW_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw'],
            partition_filename_cb=lambda x: f'{x[0]}-{x[1]}.parquet'
        )
        logging.info('Saved live prediction data to S3')
    else:
        write_dataframe_to_s3(
            df=combined,
            s3_root_path=S3_BUCKET_PATH + GW_RETRO_PREDICTIONS_SUFFIX,
            partition_cols=['season', 'gw'],
            partition_filename_cb=lambda x: f'{x[0]}-{x[1]}.parquet'
        )
        logging.info('Saved retro prediction data to S3')


def merge_model_predictions(old_model_predictions, final_predictions):
    """
    Combine predictions from FPL model and FFS model. FFS model predictions are used by default and FPL predictions are
    used if FFS not available.

    :param old_model_predictions: FPL model predictions
    :param final_predictions: FFS model predictions
    :return: Consolidated DataFrame of predictions
    """
    old_model_predictions = old_model_predictions.copy()
    final_predictions = final_predictions.copy()

    final_predictions['name_formatted'] = final_predictions['name'].str.replace(' ', '_').apply(
        lambda string: unidecode.unidecode(string)
    )
    old_model_predictions['name_formatted'] = old_model_predictions['name'].str.replace(' ', '_').apply(
        lambda string: unidecode.unidecode(string)
    )
    # TODO Short-term fix - need better name matching system
    final_predictions['name_formatted'] = np.where(
        final_predictions['name_formatted'] == 'diogo_jose_teixeira_da_silva',
        'diogo_jota',
        final_predictions['name_formatted']
    )
    final_predictions['name_formatted'] = np.where(
        final_predictions['name_formatted'] == 'roberto_firmino_barbosa_de_oliveira',
        'roberto_firmino',
        final_predictions['name_formatted']
    )
    final_predictions['name_formatted'] = np.where(
        final_predictions['name_formatted'] == 'nelson_semedo',
        'nelson_cabral_semedo',
        final_predictions['name_formatted']
    )
    final_predictions['name_formatted'] = np.where(
        final_predictions['name_formatted'] == 'daniel_podence',
        'daniel_castelo_podence',
        final_predictions['name_formatted']
    )

    # Look for closest name match
    name_matches = old_model_predictions[['name_formatted']].merge(
        final_predictions['name_formatted'],
        on='name_formatted',
        how='outer',
        indicator=True
    )

    name_matches.loc[name_matches['_merge'] == 'right_only', 'name_closest_fpl_match'] = name_matches.loc[
        name_matches['_merge'] == 'right_only', 'name_formatted'].apply(
        lambda x: difflib.get_close_matches(
            x,
            name_matches[name_matches['_merge'] == 'left_only']['name_formatted'],
            n=1,
            cutoff=0  # Set to 0 to prevent empty list from being returned
        )[0]
    )

    final_predictions = final_predictions.merge(
        name_matches,
        on='name_formatted',
        how='left'
    )

    final_predictions.drop(columns=['_merge'], inplace=True)
    final_predictions['name_join_key'] = np.where(
        final_predictions['name_closest_fpl_match'].isnull(),
        final_predictions['name_formatted'],
        final_predictions['name_closest_fpl_match']
    )

    assert final_predictions['name_join_key'].isnull().sum() == 0

    # Align team names to both use FPL
    team_data = pd.read_csv('data/external/team_season_data.csv')
    ffs_team_name_to_fpl = dict(
        zip(
            team_data[team_data['season'] == CURRENT_SEASON_FPL]['team_name_ffs'],
            team_data[team_data['season'] == CURRENT_SEASON_FPL]['team_name']
        )
    )
    final_predictions['team_name'].replace(ffs_team_name_to_fpl, inplace=True)

    # Format and merge
    final_predictions.drop(columns=['name_closest_fpl_match'], inplace=True)

    combined = old_model_predictions.merge(
        final_predictions,
        left_on=['name_formatted', 'team_name'],
        right_on=['name_join_key', 'team_name'],
        how='outer',
        indicator=True,
        suffixes=('_old', '_new')
    )

    players_ffs_only = combined[combined['_merge'] == 'right_only'][['name_formatted_new', 'team_name']]

    logging.info(f"Player names in FFS data only: {players_ffs_only}")

    # Only keep player in both sources or FPL only:
    combined = combined[combined['_merge'] != 'right_only']

    combined.rename(columns={'name_formatted_old': 'name'}, inplace=True)

    for feature in [
        'GW_plus_1', 'GW_plus_2', 'GW_plus_3', 'GW_plus_4', 'GW_plus_5', 'sum', 'position_DEF', 'position_FWD',
        'position_GK', 'position_MID'
    ]:
        combined[feature] = np.where(
            combined['sum_new'].isnull(),
            combined[f'{feature}_old'],
            combined[f'{feature}_new']
        )

    combined['model'] = np.where(
        combined['sum_new'].isnull(),
        'lstm_v4',
        'DeepFantasyFootball_v02'
    )

    combined.sort_values('sum', ascending=False, inplace=True)

    logging.info('Model used breakdown:')
    logging.info(
        combined[~combined['sum'].isnull()]['model'].value_counts()
    )

    return combined


def make_position_changes(df_with_old_positions, season, live_run, previous_gw=None):
    """
    Some players change position at the start of a new season. This can lead to issues with team selection because
    predictions for GW 1 will use the previous season's position. Therefore the team selection criteria may be
    breached when making the team selection for GW 2.

    This function identifies players who changed position between seasons and changes the position for these players in
    a provided DataFrame.

    :param df_with_old_positions: DataFrame which uses positions from season_1 e.g. predictions for GW 1
    :param season: New season
    :param live_run: Boolean. Live or retro predictions
    :param previous_gw: Previous GW
    :return: Modifies in-place
    """
    if live_run:
        gw_data = old_model_make_predictions.load_live_data(previous_gw=previous_gw, save_file=False)
    else:
        # TODO Smarter way of finding latest file:
        gw_data = old_model_make_predictions.load_retro_data(
            current_season_data_filepath='data/gw_player_data/gw_37_player_data.parquet'
        )

    gw_data['name'] = gw_data['name'].str.replace(' ', '_').apply(
        lambda string: unidecode.unidecode(string)
    )

    names_and_pos_season_1 = gw_data[gw_data['season'] == season][
        ['name', 'position_DEF', 'position_FWD', 'position_GK', 'position_MID']].drop_duplicates()

    names_and_pos_season_1.loc[names_and_pos_season_1['position_DEF'] == 1, 'position'] = 'DEF'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_FWD'] == 1, 'position'] = 'FWD'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_GK'] == 1, 'position'] = 'GK'
    names_and_pos_season_1.loc[names_and_pos_season_1['position_MID'] == 1, 'position'] = 'MID'

    names_and_pos_season_2 = df_with_old_positions.copy()[
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

    position_changes.rename(columns={'position_season_1': 'position'}, inplace=True)
    position_changes.drop(columns=['position_season_2'], inplace=True)
    position_changes = pd.get_dummies(position_changes, columns=['position'])

    logging.info(
        f"Players who changed position in {season}: {set(position_changes['name'])}"
    )

    # Make sure dummy column created for all positions. Sometimes players who changed position only covers 2 positions
    for position_col in ['position_DEF', 'position_MID', 'position_FWD']:
        if position_col in position_changes.columns:
            continue
        else:
            position_changes[position_col] = 0

    print(position_changes.head())

    for _, row in position_changes.iterrows():
        df_with_old_positions.loc[
            (df_with_old_positions['name'] == row['name']),
            ['position_DEF', 'position_FWD', 'position_MID']
        ] = row['position_DEF'], row['position_FWD'], row['position_MID']


def _include_players_missing_in_previous_gameweek(
        full_data,
        previous_gw,
        prediction_season_order,
        teams_not_in_previous_gw,
        use_fantasy_football_scout_team_names=False
):
    """
    For teams who did not play in the previous gameweek we use the gameweek before to make predictions. We do this by
    changing the gameweek from (previous_gameweek - 1) to previous_gameweek for these teams. Note whilst we may have
    next fixture data for these teams (e.g. odds) this method ignores it.

    :param full_data: DataFrame of player data
    :param previous_gw: Gameweek prior to the one you want to make a selection for
    :param prediction_season_order: Season order number (2019/20 season is 4). Season order number for the previous_gw
    i.e. for gameweek 1 predictions you will need to provide the season order number for the previous season
    :param teams_not_in_previous_gw: List of teams (FPL team names) who were not in the previous gameweek fixtures
    :param use_fantasy_football_scout_team_names: Boolean. Set to True to if applying function to fantasy football scout
    data

    :return: None. Modifies in-place
    """

    if use_fantasy_football_scout_team_names:
        team_data_names = pd.read_csv('data/external/team_season_data.csv')
        fpl_team_name_to_ffs = dict(
            zip(
                team_data_names[team_data_names['season'] == CURRENT_SEASON_FPL]['team_name'],
                team_data_names[team_data_names['season'] == CURRENT_SEASON_FPL]['team_name_ffs']
            )
        )
        teams_not_in_previous_gw = [
            fpl_team_name_to_ffs[fpl_team_name] for fpl_team_name in teams_not_in_previous_gw
        ]

    for missing_team in teams_not_in_previous_gw:
        # TODO Remove previous_gw == 1 logic. Safe to assume this won't happen again
        if previous_gw == 1:
            full_data.loc[
                (full_data['gw'] == 38) &
                (full_data['season_order'] == prediction_season_order - 1) &
                (full_data['team_name'] == missing_team) &
                (~full_data['name'].isin(['jeff_hendrick', 'joe_hart'])),  # Players who transferred to new PL team
                ['gw', 'season_order']
            ] = previous_gw, prediction_season_order
        else:
            full_data.loc[
                (full_data['gw'] == previous_gw - 1) &
                (full_data['season_order'] == prediction_season_order) &
                (full_data['team_name'] == missing_team),
                'gw'
            ] = previous_gw
