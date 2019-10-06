import pandas as pd
import numpy as np


def _format_kickoff_time(raw_kickoff_time):
    """
    Format raw string kickoff_time Series to datetime Series

    :param raw_kickoff_time: Series of raw "kickoff_time" column
    :return: Kickoff time as datetime Series
    """

    kickoff_time = raw_kickoff_time.str.replace('T', ' ')
    kickoff_time = pd.to_datetime(kickoff_time)
    kickoff_time = kickoff_time.dt.tz_convert('Europe/London')

    return kickoff_time


def create_features_from_kickoff_time(df):
    """
    Create additional features from raw kickoff time:
    - kickoff month dummy
    - late kickoff flag (past 17:00)
    - early kickoff flag (before 13:00)

    :param df: DataFrame containing raw kickoff_time column
    :return: DataFrame with additional kickoff time columns
    """

    assert "kickoff_time" in df.columns, "kickoff_time not in DataFrame"

    df = df.copy()

    df['kickoff_time'] = _format_kickoff_time(df['kickoff_time'])

    # Get features from kickoff time
    df['kickoff_month'] = df['kickoff_time'].dt.strftime("%b")
    df = pd.get_dummies(df, columns=['kickoff_month'])

    df['kickoff_hour'] = df['kickoff_time'].dt.hour

    df['late_kickoff'] = np.where(
        df['kickoff_hour'] >= 17,
        1,
        0
    )

    df['early_kickoff'] = np.where(
        df['kickoff_hour'] <= 13,
        1,
        0
    )

    df.drop(columns=['kickoff_hour'], axis=1, inplace=True)

    return df
