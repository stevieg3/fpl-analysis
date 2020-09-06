import numpy as np


def _calculate_minutes_points_per_match(minutes):
    """
    Calculate points scored due to minutes played in a _single_ match

    :param minutes: Number of minutes played
    :return: Points
    """
    points = 0
    if minutes >= 60:
        points += 2
    elif minutes > 0:
        points += 1
    return points


def calculate_minutes_points_total_matches(total_minutes):
    """
    FFS collapses double gameweeks into a single row entry. This function calculates the points for minutes played over
    2 fixtures.

    :param total_minutes: Total minutes played over 2 fixtures
    :return: Points
    """
    second_match_mins = np.max([total_minutes - 90, 0])
    first_match_mins = total_minutes - second_match_mins

    second_match_points = _calculate_minutes_points_per_match(second_match_mins)
    first_match_points = _calculate_minutes_points_per_match(first_match_mins)

    total_points = first_match_points + second_match_points
    return total_points


def calculate_fpl_points(df):
    """
    Calculate points scored by a player according to FPL rules: https://fantasy.premierleague.com/help/rules

    :param df: DataFrame containing player-level actions which score points
    :return: Series of points scored
    """
    df = df.copy()
    df['points'] = 0

    # Minutes
    df['points'] += df['Time Played'].apply(calculate_minutes_points_total_matches)

    # Goals
    df['points'] += np.where(
        df['position'].isin(['GK', 'DEF']),
        6 * df['Goals'],
        0
    )

    df['points'] += np.where(
        df['position'] == 'MID',
        5 * df['Goals'],
        0
    )

    df['points'] += np.where(
        df['position'] == 'FWD',
        4 * df['Goals'],
        0
    )

    # Assists
    df['points'] += df['Assists'] * 3

    # Clean sheets
    df['points'] += np.where(
        df['position'].isin(['GK', 'DEF']),
        4 * df['Clean Sheets'],
        0
    )

    df['points'] += np.where(
        df['position'] == 'MID',
        df['Clean Sheets'],
        0
    )

    # Shot saves
    df['points'] += np.where(
        df['position'] == 'GK',
        (df['Saves'] / 3).astype(int),
        0
    )

    # Penalty saves
    df['points'] += np.where(
        df['position'] == 'GK',
        df['Saves From Penalty'] * 5,
        0
    )

    # Penalty misses
    df['points'] += -2 * df['Penalties Missed']

    # Goals conceded
    df['points'] += np.where(
        df['position'].isin(['GK', 'DEF']),
        -(df['Goals Conceded'] / 2).astype(int),
        0
    )

    # Yellow cards
    df['points'] += -df['Premier League Yellow Cards']

    # Red cards
    df['points'] += -df['Premier League Total Red Cards'] * 3

    # Own goals
    df['points'] += -df['Own Goals'] * 2

    # 0 minutes
    df['points'] = np.where(
        df['Time Played'] == 0,
        0,
        df['points']
    )

    return df['points']
