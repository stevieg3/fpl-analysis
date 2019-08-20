# TODO Remove commented out lines from transform method


from sklearn.base import TransformerMixin, BaseEstimator


class TimeSeriesFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to create time series features
    """
    def __init__(self, halflife, max_lag, max_diff, columns):
        """
        Add exponential weighted moving average, lags and difference features to DataFrame

        :param halflife: decay in terms of half-life
        :param max_lag: maximum number of lagged features to include
        :param max_diff: maximum number of difference features to include
        :param columns: list of columns of time series features
        """
        self.halflife = halflife
        self.max_lag = max_lag
        self.max_diff = max_diff
        self.columns = columns

    def fit(self, X, y=None):
        """
        :param X: Pandas DataFrame
        :param y: None
        :return:
        """
        self.X = X
        return self

    def transform(self, X):
        features = self.X.copy()
        for col in self.columns:

            # EWM
            features[col + '_EMA'] = features.groupby('ID')[col].apply(lambda x: x.ewm(halflife=self.halflife).mean())
            #features[col + '_EMA'] = features.groupby('ID')[col + '_EMA']  #.shift(
            #    1)  # Shift to prevent use of current GW features

            if (self.max_lag is None) and (self.max_diff is None):
                continue
            else:
                # Lagged variables
                for lag in range(1, self.max_lag + 1):
                    features[col + f'_L{lag}'] = features.groupby('ID')[col].shift(lag)

                # Difference variables
                for diff in range(1, self.max_diff + 1):
                    features[col + f'_D{diff}'] = features.groupby('ID')[col].diff(diff) #.shift(1).diff(
                #        diff)  # Shift to prevent use of current GW features

                # Drop current GW features
                # if col != 'value':  # Keep value as it should be known before match starts (note: total_points dropped)
                #    features.drop(col, axis=1, inplace=True)
                # else:
                #    continue

        # features.drop(columns=['ID'], axis=1, inplace=True)
        # features.drop(columns=['gw'], axis=1, inplace=True)

        return features
