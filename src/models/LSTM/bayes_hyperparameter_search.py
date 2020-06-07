import numpy as np
from keras.layers import \
    Dense, \
    LSTM, \
    BatchNormalization, \
    Dropout, \
    Activation
from keras import \
    initializers, \
    optimizers, \
    Sequential
from hyperopt import \
    hp, \
    fmin, \
    rand, \
    tpe, \
    Trials
from sklearn.metrics import mean_squared_error


SPACE = {
    'n_lstm_layers': hp.uniformint('n_lstm_layers', 1, 2),
    'lstm_units': hp.uniformint('lstm_units', 10, 200),
    'lstm_dropout': hp.quniform('lstm_dropout', low=0, high=0.5, q=0.01),
    'lstm_recurrent_dropout': hp.quniform('lstm_recurrent_dropout', low=0, high=0.5, q=0.01),
    'lstm_output_dropout': hp.quniform('lstm_output_dropout', low=0, high=0.5, q=0.01),
    'n_dense_layers': hp.uniformint('n_dense_layers', 0, 3),
    'dense_units': hp.uniformint('dense_units', 10, 200),
    'dense_dropout': hp.quniform('dense_dropout', low=0, high=0.5, q=0.01),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)),
    'batch_size': hp.uniformint('batch_size', 32, 1024)
}

INTEGER_PARAMS = ['n_lstm_layers', 'lstm_units', 'n_dense_layers', 'dense_units', 'batch_size']

N_STEPS_IN = 5
N_STEPS_OUT = 5


X_train = None
y_train = None
X_dev = None
y_dev = None


def create_model(
    n_steps_in,
    n_steps_out,
    n_features,
    n_lstm_layers=1,
    lstm_units=50,
    lstm_dropout=0.0,
    lstm_recurrent_dropout=0.0,
    lstm_output_dropout=0.3,
    n_dense_layers=1,
    dense_units=50,
    dense_dropout=0.3
):
    model = Sequential(name='lstm_model')

    # LSTM layers
    if n_lstm_layers == 1:
        model.add(
            LSTM(
                units=lstm_units,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_recurrent_dropout,
                input_shape=(n_steps_in, n_features),
                name='lstm_layer_1'
            )
        )
        model.add(
            Dropout(
                rate=lstm_output_dropout,
                name='lstm_layer_1_dropout'
            )
        )
    else:
        model.add(
            LSTM(
                units=lstm_units,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_recurrent_dropout,
                return_sequences=True,
                input_shape=(n_steps_in, n_features),
                name='lstm_layer_1'
            )
        )
        for i in range(n_lstm_layers-1):
            model.add(
                LSTM(
                    units=lstm_units,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_recurrent_dropout,
                    name=f'lstm_layer_{i+2}'
                )
            )
        model.add(
            Dropout(
                rate=lstm_output_dropout,
                name='lstm_multilayer_dropout'
            )
        )

    # Fully connected layers
    for i in range(n_dense_layers):
        model.add(Dense(dense_units, name=f'dense_layer_{i+1}'))
        model.add(BatchNormalization(name=f'dense_batch_norm_{i+1}'))
        model.add(Activation('relu', name=f'dense_activation_{i+1}'))
        model.add(Dropout(rate=dense_dropout, name=f'dense_dropout_{i+1}'))

    # Output layer
    model.add(Dense(
        n_steps_out, kernel_initializer=initializers.glorot_normal(), name='dense_output'
    ))

    return model


def objective(hyperparameters):
    # Hacky workaround for bug in package which saves integer hyperparameters as floats in the final dictionary. We
    # explicitly cast as integers to avoid errors.
    for hyper in INTEGER_PARAMS:
        try:
            as_int = int(hyperparameters[hyper])
            hyperparameters[hyper] = as_int
        except:
            continue

    # Extract keys not needed for model creation
    fitting_params = {}
    fitting_params['learning_rate'] = hyperparameters['learning_rate']
    fitting_params['batch_size'] = hyperparameters['batch_size']

    del hyperparameters['learning_rate'], hyperparameters['batch_size']

    # Create model
    lstm_model = create_model(n_steps_in=N_STEPS_IN, n_steps_out=N_STEPS_OUT, n_features=63, **hyperparameters)

    # Compile model
    optimizer = optimizers.Adam(learning_rate=fitting_params['learning_rate'])
    lstm_model.compile(loss='mse', optimizer=optimizer)

    # Fit model
    lstm_model.fit(X_train, y_train, batch_size=fitting_params['batch_size'], epochs=30, verbose=0)

    # Evaluate on dev set
    predictions = lstm_model.predict(X_dev)
    mse = mean_squared_error(y_dev, predictions)

    return mse  # Has to be a minimisation problem
