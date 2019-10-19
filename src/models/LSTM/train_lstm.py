# TODO Need to sort out name series
import numpy as np
import logging
import pickle

from keras.models import load_model
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import \
    LSTM,\
    Dense, \
    RepeatVector, \
    TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from src.models.utils import \
    _load_all_historical_data, \
    _map_season_string_to_ordered_numeric, \
    _generate_known_features_for_next_gw, \
    custom_train_test_split, \
    split_sequences
from src.models.constants import \
    COLUMNS_TO_DROP_FOR_TRAINING

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


# SET MULTI-STEP INPUT AND OUTPUT PARAMETERS

N_STEPS_IN = 10
N_STEPS_OUT = 5


# LOAD ALL AVAILABLE DATA

all_data = load_all_data()


# CREATE TARGET

all_data['total_points_plus1_gw'] = all_data.groupby(['name'])['total_points'].shift(-1)
all_data = all_data[~all_data['total_points_plus1_gw'].isnull()]  # drop nulls (last gw)


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

COLUMNS_TO_NOT_SCALE = [
    'name', 'season', 'team_name', 'team_name_opponent', 'total_points_plus1_gw'
]

COLUMNS_TO_SCALE = []

for col in set(training_subset_df.columns) - set(COLUMNS_TO_NOT_SCALE):
    if (training_subset_df[col].max() > 1) | (training_subset_df[col].min() < -1):
        COLUMNS_TO_SCALE.append(col)

mms = MinMaxScaler()

training_subset_df[COLUMNS_TO_SCALE] = mms.fit_transform(training_subset_df[COLUMNS_TO_SCALE])

pickle.dump(
    mms,
    open('src/models/pickles/min_max_scalar_lstm.pickle', 'wb')
)

pickle.dump(
    COLUMNS_TO_SCALE,
    open('src/models/pickles/min_max_scalar_columns.pickle', 'wb')
)


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


# TRANSFORM VALIDATION DATA INTO REQUIRED SHAPE FOR LSTM

validation_df[COLUMNS_TO_SCALE] = mms.transform(validation_df[COLUMNS_TO_SCALE])

X_list = []
y_list = []

for player in list(validation_df['name'].unique()):
    player_df = validation_df[validation_df['name'] == player]
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

X_val = np.concatenate(X_list, axis=0)
y_val = np.concatenate(y_list, axis=0)
print(X_val.shape)
print(y_val.shape)


# DEFINE MODEL

N_FEATURES = X.shape[2]

model = Sequential()
model.add(
    LSTM(
        100,
        activation='relu',
        return_sequences=True,
        input_shape=(N_STEPS_IN, N_FEATURES),
        kernel_regularizer=l2(0.01),
        recurrent_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        dropout=0.2
    )
)
model.add(
    LSTM(
        100,
        activation='relu',
        kernel_regularizer=l2(0.01),
        recurrent_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
        dropout=0.2
    )
)
model.add(Dense(N_STEPS_OUT))
model.compile(optimizer='adam', loss='mse')


# ENCODER-DECODER

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(N_STEPS_IN, N_FEATURES)))
model.add(RepeatVector(N_STEPS_OUT))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')


# FIT MODEL

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(X, y, batch_size=60, epochs=50, verbose=1, validation_data=(X_val, y_val), callbacks=[es])

# TODO Increase number of epochs as it did not early stop on 50

# save model and architecture to single file
model.save("src/models/pickles/test_lstm_model.h5")
print("Saved model to disk")


# load model
# model_loaded = load_model('model.h5')
