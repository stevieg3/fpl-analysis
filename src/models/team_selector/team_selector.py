import pandas as pd
import numpy as np
from pulp import \
    lpSum, \
    LpProblem, \
    LpMaximize, \
    LpVariable, \
    LpInteger

from src.models.constants import SELL_ON_TAX
from src.models.utils import round_down_to_nearest_10th


def load_player_predictions(prediction_filepath):
    """
    Load saved player points predictions as Pandas DataFrame
    :param prediction_filepath: Filepath to predictions parquet file
    :return: Pandas DataFrame
    """
    predictions = pd.read_parquet(prediction_filepath)
    if 'rank' in predictions.columns:
        predictions.drop('rank', inplace=True, axis=1)

    _format_player_predictions(predictions)

    return predictions


def _format_player_predictions(predictions_df):
    """
    Formats names in raw predictions DataFrame and sets 'predictions' column.
    :param predictions_df: Raw predictions DataFrame as loaded from parquet
    :return: None. Modifies DataFrame in-place
    """
    predictions_df.rename(columns={'sum': 'predictions'}, inplace=True)

    predictions_df['name'] = predictions_df['name'].str.replace(' ', '_')
    predictions_df['name'] = predictions_df['name'].str.replace('-', '_')


def _get_prev_predictions_for_missing_players_in_previous_team(previous_predictions, current_predictions_for_prev_team):
    """
    Returns DataFrame of previous predictions for players who are in the previously selected team but do not have points
    predictions for the current gameweek.

    :param previous_predictions: DataFrame of previous predictions
    :param current_predictions_for_prev_team: DataFrame of current predictions for previous team. Points predictions
    should be null for some players.
    :return: DataFrame
    """
    previous_predictions_missing_players = previous_predictions.merge(
        current_predictions_for_prev_team[current_predictions_for_prev_team['predictions'].isnull()][['name']],
        on='name',
        how='inner'
    )  # TODO Need to ensure they are not substituted out - could give arbitrary number of points to guarantee selection

    return previous_predictions_missing_players


def get_budget(previous_team_selection, current_predictions_df, money_in_bank=0.):
    """
    Get available budget. Factors in tax on any sales profits made. See
    https://twitter.com/officialfpl/status/810473725627957248?lang=en;
    https://www.reddit.com/r/FantasyPL/comments/90pgwq/can_someone_explain_to_me_how_price_change_works/
    for details.

    :param previous_team_selection: DataFrame of players selected in previous GW containing 'purchase_price' column.
    :param current_predictions_df: DataFrame of current predictions containing 'next_match_value' column.
    :param money_in_bank: Money in FPL bank
    :return: Budget
    """
    budget_calculation_df = previous_team_selection[['name', 'purchase_price']].merge(
        current_predictions_df[['name', 'next_match_value']],
        how='left',
        on='name'
    )
    budget_calculation_df.rename(columns={'next_match_value': 'current_price'}, inplace=True)

    budget_calculation_df['profit'] = budget_calculation_df['current_price'] - budget_calculation_df['purchase_price']
    budget_calculation_df['profit'] = np.round(budget_calculation_df['profit'], 2)  # To prevent Python precision errors
    budget_calculation_df['profit_after_tax'] = (budget_calculation_df['profit'] * SELL_ON_TAX).apply(
        round_down_to_nearest_10th
    )

    budget_calculation_df['selling_price'] = np.where(
        budget_calculation_df['profit'] > 0,
        budget_calculation_df['purchase_price'] + budget_calculation_df['profit_after_tax'],
        budget_calculation_df['current_price']
    )

    budget = budget_calculation_df['selling_price'].sum()
    budget += money_in_bank

    return budget


# INTERFACE
previous_gw = 23

previous_predictions = load_player_predictions('data/gw_predictions/gw23_v3_lstm_player_predictions.parquet')
current_predictions = load_player_predictions('data/gw_predictions/gw24_v3_lstm_player_predictions.parquet')

previous_team_selection = pd.read_parquet('data/gw_team_selections/gw23_v3_lstm_team_selections.parquet')
previous_team_selection['in_gw_1_team'] = 1  # TODO Rename column to something else (replace everywhere)

previous_team_selection_names = previous_team_selection.copy()[['name', 'in_gw_1_team']]

current_predictions_for_prev_team = current_predictions.merge(previous_team_selection_names, on='name', how='inner')

if current_predictions_for_prev_team['predictions'].isnull().sum() != 0:
    # TODO Log this event
    print('Some players missing')
    # Players not playing in next GW still appear but have null values for points predictions and next match value
    current_predictions.dropna(axis=0, how='any', inplace=True)

    previous_predictions_missing_players = _get_prev_predictions_for_missing_players_in_previous_team(
        previous_predictions=previous_predictions,
        current_predictions_for_prev_team=current_predictions_for_prev_team
    )

    # Append previous predictions for missing players who are in current team:
    current_predictions = current_predictions.append(previous_predictions_missing_players)


current_predictions = current_predictions.merge(previous_team_selection_names, on='name', how='left')
current_predictions['in_gw_1_team'] = current_predictions['in_gw_1_team'].fillna(0)
assert current_predictions['in_gw_1_team'].sum() == 15


# Create top 3 flag:
current_predictions.loc[0:2, 'in_top_3'] = 1
current_predictions['in_top_3'].fillna(0, inplace=True)

# Create low value player flag:
current_predictions['low_value_player'] = np.where(
    current_predictions['next_match_value'] < 4.1,  # TODO Save as constant
    1,
    0
)

# Rashford unlikely to play next GW
current_predictions.loc[current_predictions['name'] == 'marcus_rashford', 'GW_plus_1'] = 0

current_predictions['predictions'] = current_predictions[
    ['GW_plus_1', 'GW_plus_2', 'GW_plus_3', 'GW_plus_4', 'GW_plus_5']
].sum(axis=1)


budget = get_budget(
    previous_team_selection=previous_team_selection,
    current_predictions_df=current_predictions,
    money_in_bank=5.1
)


# PICK TEAM

def solve_fpl_team_selection_problem(
        current_predictions_df,
        budget_constraint,
        max_permitted_transfers=1,
        include_top_3=False,
        include_low_value_player=False,
        min_spend=0
):
    """
    Use PuLP to select the best team of 15 players which maximises total predicted points subject to constraints.

    :param current_predictions_df: Predictions DataFrame
    :param budget_constraint: Total budget available
    :param max_permitted_transfers: Maximum number of transfers allowed
    :param include_top_3: Include top 3 players in final squad (identified using `in_top_3` column)
    :param include_low_value_player: Include low value player in squad (identified using `low_value_player` column)
    :param min_spend: Minimum amount spent on final squad. Use to prevent value of team being too low

    :return: Solved LpProblem object
    """
    current_predictions = current_predictions_df.copy()

    team_names = current_predictions['team_name'].unique()
    current_predictions = pd.get_dummies(current_predictions, columns=['team_name'])
    players = list(current_predictions['name'])

    # CREATE NAME-VALUE DICTIONARIES FOR USE IN CONSTRAINTS

    team_dict = {}
    for team in team_names:
        team_dict[team] = dict(zip(current_predictions['name'], current_predictions[f'team_name_{team}']))

    costs = dict(zip(current_predictions['name'], current_predictions['next_match_value']))

    predictions = dict(zip(current_predictions['name'], current_predictions['predictions']))

    DEF_flag = dict(zip(current_predictions['name'], current_predictions['position_DEF']))

    FWD_flag = dict(zip(current_predictions['name'], current_predictions['position_FWD']))

    GK_flag = dict(zip(current_predictions['name'], current_predictions['position_GK']))

    MID_flag = dict(zip(current_predictions['name'], current_predictions['position_MID']))

    GW1_team = dict(zip(current_predictions['name'], current_predictions['in_gw_1_team']))

    in_top_3 = dict(zip(current_predictions['name'], current_predictions['in_top_3']))

    low_value_flag = dict(zip(current_predictions['name'], current_predictions['low_value_player']))

    # SET OBJECTIVE FUNCTION

    prob = LpProblem('FPL team selection', LpMaximize)
    player_vars = LpVariable.dicts('player', players, 0, 1, LpInteger)

    prob += lpSum([predictions[p] * player_vars[p] for p in players]), "Total predicted points"

    # DEFINE CONSTRAINTS

    # Rules-of-the-game constraints:

    prob += lpSum([costs[p] * player_vars[p] for p in players]) <= budget_constraint, "Total cost less than X"

    prob += lpSum(player_vars[p] for p in players) == 15, "Select 15 players"

    prob += lpSum(DEF_flag[p] * player_vars[p] for p in players) == 5, "5 defenders"

    prob += lpSum(GK_flag[p] * player_vars[p] for p in players) == 2, "2 goalkeepers"

    prob += lpSum(MID_flag[p] * player_vars[p] for p in players) == 5, "5 midfielders"

    prob += lpSum(FWD_flag[p] * player_vars[p] for p in players) == 3, "3 forwards"

    for team in team_dict.keys():
        prob += lpSum(team_dict[team][p] * player_vars[p] for p in players) <= 3, f"Max 3 players in the same {team}"

    # Additional constraints:

    if include_top_3:
        prob += lpSum(in_top_3[p] * player_vars[p] for p in players) == 3, "Top 3 must be included"

    if include_low_value_player:
        prob += lpSum(low_value_flag[p] * player_vars[p] for p in players) == 1, "Include 1 low value player"

    if min_spend > 0:
        prob += lpSum([costs[p] * player_vars[p] for p in players]) >= min_spend, "Total cost greater than X"

    prob += lpSum(GW1_team[p] * player_vars[p] for p in players) >= 15 - max_permitted_transfers, \
        "At least 15-`max_permitted_transfers` players from original team"

    # SOLVE OBJECTIVE FUNCTION SUBJECT TO CONSTRAINTS

    prob.solve()
    assert prob.status == 1, 'FPL team selection problem not solved!'

    return prob


def fpl_team_selection(current_predictions_df, solved_prob):
    """
    Produces DataFrame of selected players and prints transfer information.

    :param current_predictions_df: Predictions DataFrame
    :param solved_prob: Solved LpProblem object
    :return: DataFrame of selected players
    """
    chosen_players = []
    for v in solved_prob.variables():
        if v.varValue == 0:
            continue
        else:
            chosen_players.append(v.name.replace('player_', ''))

    test_selection = current_predictions_df[current_predictions_df['name'].isin(chosen_players)]

    if test_selection.sum()['in_gw_1_team'] == 15:
        print("""
        --------------------------------------------------------------
        No transfers made.
        --------------------------------------------------------------
        """)
    else:
        print(f"""
        --------------------------------------------------------------
        {15 - test_selection.sum()['in_gw_1_team']} transfer(s) made.
        
        Players out:
        {list(set(previous_team_selection_names['name']) - set(test_selection['name']))}
        
        Players in:
        {list(test_selection[test_selection['in_gw_1_team'] == 0]['name'])}
        --------------------------------------------------------------
        """)
    # print(f"""
    #     --------------------------------------------------------------
    #     Low value player:
    #     {test_selection[test_selection['low_value_player'] == 1]['name'].item()}
    #     --------------------------------------------------------------
    # """)

    return test_selection


# Select starting 11

solved_problem = solve_fpl_team_selection_problem(current_predictions_df=current_predictions, budget_constraint=budget)
selected_team = fpl_team_selection(current_predictions_df=current_predictions, solved_prob=solved_problem)


# 0% chance of playing
# selected_team.loc[selected_team['name'] == 'diego_rico', 'predictions'] = -1


def solve_starting_11_problem(selected_team_df):
    """
    Use PuLP to select the starting 11 which maximises total predicted points subject to constraints.

    :param selected_team_df: DataFrame of players in selected team
    :return: Solved LpProblem object
    """
    selected_team = selected_team_df.copy()

    players = list(selected_team['name'])

    # CREATE NAME-VALUE DICTIONARIES FOR USE IN CONSTRAINTS

    predictions = dict(zip(selected_team['name'], selected_team['predictions']))

    DEF_flag = dict(zip(selected_team['name'], selected_team['position_DEF']))

    FWD_flag = dict(zip(selected_team['name'], selected_team['position_FWD']))

    GK_flag = dict(zip(selected_team['name'], selected_team['position_GK']))

    MID_flag = dict(zip(selected_team['name'], selected_team['position_MID']))

    # SET OBJECTIVE FUNCTION

    prob = LpProblem('FPL team selection', LpMaximize)
    player_vars = LpVariable.dicts('player', players, 0, 1, LpInteger)

    prob += lpSum([predictions[p] * player_vars[p] for p in players]), "Total predicted points"

    # DEFINE CONSTRAINTS

    # Rules of the game constraints:

    prob += lpSum(player_vars[p] for p in players) == 11, "Select 11 players"

    prob += lpSum(DEF_flag[p] * player_vars[p] for p in players) >= 3, "At least 3 defenders"

    prob += lpSum(GK_flag[p] * player_vars[p] for p in players) == 1, "1 goalkeeper"

    prob += lpSum(FWD_flag[p] * player_vars[p] for p in players) >= 1, "At least 1 forward"

    # SOLVE OBJECTIVE FUNCTION SUBJECT TO CONSTRAINTS

    prob.solve()
    assert prob.status == 1, 'FPL team selection problem not solved!'

    return prob


def starting_11_selection(current_predictions_df, solved_prob):
    """
    Produces DataFrame of starting 11 players and prints expected points and recommended captain.

    :param current_predictions_df: Predictions DataFrame
    :param solved_prob: Solved LpProblem object
    :return:
    """
    chosen_players = []
    for v in solved_prob.variables():
        if v.varValue == 0:
            continue
        else:
            chosen_players.append(v.name.replace('player_', ''))

    test_selection_11 = current_predictions_df[current_predictions_df['name'].isin(chosen_players)]
    test_selection_11.reset_index(drop=True, inplace=True)

    print(f"""
        --------------------------------------------------------------
        Recommended captain:
        {test_selection_11[test_selection_11['GW_plus_1'] == test_selection_11['GW_plus_1'].max()]['name'].item()}
    
        Expected points:
        {test_selection_11['GW_plus_1'].sum() + test_selection_11.loc[0, 'GW_plus_1']}
        --------------------------------------------------------------
    """)

    return test_selection_11


solved_11 = solve_starting_11_problem(selected_team)
starting_11 = starting_11_selection(current_predictions_df=current_predictions, solved_prob=solved_11)


# Save

starting_11 = starting_11.reset_index(drop=True)[['name']]
starting_11['starting_11'] = 1

gw_selection_df = selected_team.merge(
    starting_11,
    on=['name'],
    how='left'
)
gw_selection_df['starting_11'] = gw_selection_df['starting_11'].fillna(0)

# Add purchase_price and gw_bought_in columns
gw_selection_df = gw_selection_df.merge(
    previous_team_selection[['name', 'purchase_price', 'gw_introduced_in']],
    how='left',
    on='name'
)

gw_selection_df.loc[
    (gw_selection_df['purchase_price'].isnull()) & (gw_selection_df['in_gw_1_team'] == 0),
    'purchase_price'
] = gw_selection_df['next_match_value']

gw_selection_df.loc[
    (gw_selection_df['gw_introduced_in'].isnull()) & (gw_selection_df['in_gw_1_team'] == 0),
    'gw_introduced_in'
] = previous_gw + 1

assert gw_selection_df[['purchase_price', 'gw_introduced_in']].isnull().sum().sum() == 0

gw_selection_df.to_parquet('data/gw_team_selections/gw24_v3_lstm_team_selections.parquet', index=False)
