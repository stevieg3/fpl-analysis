"""
Some mock code to calculate permutation feature importance for LSTM v3. Use this methodology when training new model to
narrow down the number of features. Found similar performance (MSE) with reduced features (20 total) to original with
+60 features.

# TODO Many of the data prep stages should be made into functions to make it easier to do training and prediction.
# TODO For prediction change so that all predictions done in one go rather than player-by-player.
"""
import numpy as np
import logging
import pickle
import pandas as pd

from keras.models import load_model
from sklearn.metrics import mean_squared_error

from src.models.utils import \
    _load_all_historical_data, \
    _map_season_string_to_ordered_numeric, \
    _generate_known_features_for_next_gw, \
    custom_train_test_split, \
    split_sequences, \
    _load_model_from_pickle
from src.models.constants import \
    COLUMNS_TO_DROP_FOR_TRAINING

pd.options.mode.chained_assignment = None
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def load_all_data():
    full_data = _load_all_historical_data()
    full_data.sort_values(['name', 'season', 'gw'], inplace=True)

    _map_season_string_to_ordered_numeric(full_data)
    _generate_known_features_for_next_gw(full_data)

    # Remove Brendan Galloway due to unexplained gap in gameweek data
    full_data = full_data[full_data['name'] != 'brendan_galloway']
    full_data.drop('ID', axis=1, inplace=True)

    logging.info(f"Loaded historical data of shape {full_data.shape}")

    return full_data


def _load_model_from_h5(model_filepath):
    model = load_model(model_filepath)
    return model

# SET MULTI-STEP INPUT AND OUTPUT PARAMETERS

N_STEPS_IN = 5
N_STEPS_OUT = 5


# LOAD ALL AVAILABLE DATA

all_data = load_all_data()


# CREATE TARGET

all_data['total_points_plus1_gw'] = all_data.groupby(['name'])['total_points'].shift(-1)
all_data = all_data[~all_data['total_points_plus1_gw'].isnull()]  # drop nulls (last gw)

# TODO Should use original training data for consistency
# SPLIT INTO TRAINING AND TEST

training_df, test_df = custom_train_test_split(all_data)

# FURTHER SPLIT TRAINING INTO TRAINING SUBSET AND VALIDATION

training_subset_df, validation_df = custom_train_test_split(training_df)

training_subset_df['total_number_of_gameweeks'] = training_subset_df.groupby(['name']).transform('count')['team_name']
# Drop players if they don't have enough GW data to be used by configured LSTM
training_subset_df = training_subset_df[
    training_subset_df['total_number_of_gameweeks'] >= (N_STEPS_IN + N_STEPS_OUT - 1)
]
training_subset_df.drop('total_number_of_gameweeks', axis=1, inplace=True)

validation_df['total_number_of_gameweeks'] = validation_df.groupby(['name']).transform('count')['team_name']
validation_df = validation_df[validation_df['total_number_of_gameweeks'] >= N_STEPS_IN + N_STEPS_OUT - 1]
validation_df.drop('total_number_of_gameweeks', axis=1, inplace=True)


# NORMALISE LARGE INPUTS

# MinMaxScalar used in training
mms = _load_model_from_pickle('src/models/pickles/min_max_scalar_lstm_v3.pickle')
COLUMNS_TO_SCALE = _load_model_from_pickle('src/models/pickles/min_max_scalar_columns_v3.pickle')

COLUMNS_TO_NOT_SCALE = [
    'name', 'season', 'team_name', 'team_name_opponent', 'total_points_plus1_gw'
]

training_subset_df[COLUMNS_TO_SCALE] = mms.fit_transform(training_subset_df[COLUMNS_TO_SCALE])


# TRANSFORM TRAINING SUBSET DATA INTO REQUIRED SHAPE FOR LSTM

X_list = []
y_list = []

for player in list(training_subset_df['name'].unique()):
    player_df = training_subset_df[training_subset_df['name'] == player]
    player_df.drop(
        COLUMNS_TO_DROP_FOR_TRAINING,
        axis=1,
        inplace=True
    )
    X_player, y_player = split_sequences(
        df=player_df,
        target_column='total_points_plus1_gw',
        n_steps_in=N_STEPS_IN,
        n_steps_out=N_STEPS_OUT
    )
    X_list.append(X_player)
    y_list.append(y_player)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
print(X.shape)
print(y.shape)


lstm_model = _load_model_from_h5("src/models/pickles/v3_lstm_model.h5")

predictions = lstm_model.predict(X)

# Make predictions into 1D array
predictions_1d = np.ravel(predictions)
y_1d = np.ravel(y)

original_model_error = mean_squared_error(predictions_1d, y_1d)

FEATURES_TO_PERMUTE = set(training_subset_df.columns) - set(COLUMNS_TO_DROP_FOR_TRAINING)


# TODO Could do multiple permutations of same feature to calculate error around PFI value

def permutation_feature_importance(features_to_permute):
    """
    Based on methodology described here: https://christophm.github.io/interpretable-ml-book/feature-importance.html
    :param features_to_permute:
    :return:
    """
    pfi_dict = {}
    logging.info(f'{len(features_to_permute)} features to permute')
    logging.info(features_to_permute)

    for feature in features_to_permute:
        logging.info(f'Permuting {feature}')
        df_copy = training_subset_df.copy()
        # Randomly permute feature:
        df_copy[feature] = np.array(df_copy[feature].sample(frac=1))

        # Calculate error with permuted data
        X_list = []
        y_list = []

        for player in list(df_copy['name'].unique()):
            player_df = df_copy[df_copy['name'] == player]
            player_df.drop(
                COLUMNS_TO_DROP_FOR_TRAINING,
                axis=1,
                inplace=True
            )
            X_player, y_player = split_sequences(
                df=player_df,
                target_column='total_points_plus1_gw',
                n_steps_in=N_STEPS_IN,
                n_steps_out=N_STEPS_OUT
            )
            X_list.append(X_player)
            y_list.append(y_player)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        predictions = lstm_model.predict(X)

        predictions_1d = np.ravel(predictions)
        y_1d = np.ravel(y)

        permutation_error = mean_squared_error(predictions_1d, y_1d)

        pfi = permutation_error / original_model_error
        logging.info(f'PFI for {feature}: {pfi}')

        pfi_dict[feature] = pfi

    return pfi_dict


calculated_pfi = permutation_feature_importance(FEATURES_TO_PERMUTE)

pickle.dump(
    mms,
    open('src/models/pickles/calculated_pfi.pickle', 'wb')
)
