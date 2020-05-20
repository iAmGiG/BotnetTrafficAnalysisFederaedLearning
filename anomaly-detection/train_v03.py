# %%
import functools
from absl import app
from absl import flags
from absl import logging
import sys
import os
from typing import Callable
import pandas as pd
from glob import iglob
import numpy as np
import grpc
# from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.optimizers import SGD
import tensorflow_federated as tff
import tensorflow as tf
from tensorflow_federated.python.learning.model_utils import EnhancedModel

FLAGS = flags.FLAGS
flags.DEFINE_string('host', None, 'The host to connect to.')
flags.mark_flag_as_required('host')
flags.DEFINE_string('port', '8000', 'The port to connect to.')
flags.DEFINE_integer('n_clients', 10, 'Number of clients.')
flags.DEFINE_integer('n_rounds', 3, 'Number of rounds.')
NUM_EPOCHS = 10
BATCH_SIZE = 20


# %%
def make_remote_executor(inferred_cardinalities):
    """Make remote executor."""

    def create_worker_stack_on(ex):
        return tff.framework.ReferenceResolvingExecutor(
            tff.framework.ThreadDelegatingExecutor(ex))

    client_ex = []
    num_clients = inferred_cardinalities.get(tff.CLIENTS, None)
    if num_clients:
        print('Inferred that there are {} clients'.format(num_clients))
    else:
        print('No CLIENTS placement provided')

    for _ in range(num_clients or 0):
        channel = grpc.insecure_channel('{}:{}'.format(FLAGS.host, FLAGS.port))
        client_ex.append(
            create_worker_stack_on(
                tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING')))

    federated_ex = tff.framework.FederatingExecutor({
        None: create_worker_stack_on(tff.framework.EagerTFExecutor()),
        tff.SERVER: create_worker_stack_on(tff.framework.EagerTFExecutor()),
        tff.CLIENTS: client_ex,
    })

    return tff.framework.ReferenceResolvingExecutor(federated_ex)


# %%
def train(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)

    fisher = pd.read_csv('../fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    # split randomly shuffled data into 3 equal parts
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))
    x_train = scaler.transform(x_train)
    x_opt = scaler.transform(x_opt)
    x_test = scaler.transform(x_test)

    model = create_model(top_n_features)
    model.compile(loss="mean_squared_error",
                  optimizer="sgd")
    cp = ModelCheckpoint(filepath=f"models/model_{top_n_features}.h5",
                         save_best_only=True,
                         verbose=0)
    tb = TensorBoard(log_dir=f"./logs",
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
    NAME = "//trainData"
    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), histogram_freq=1, profile_batch=100000000)
    tensorboard = TensorBoard(log_dir=f"./logs",
                              histogram_freq=1,
                              profile_batch=100000000)
    print(f"Training model for all data combined")
    model.fit(x_train, x_train,
              epochs=5,
              batch_size=64,
              validation_data=(x_opt, x_opt),
              verbose=1,
              callbacks=[tensorboard]
              )

    print("Calculating threshold")
    x_opt_predictions = model.predict(x_opt)
    print("Calculating MSE on optimization set...")
    mse = np.mean(np.power(x_opt - x_opt_predictions, 2), axis=1)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())
    tr = mse.mean() + mse.std()
    with open(f'threshold_{top_n_features}', 'w') as t:
        t.write(str(tr))
    print(f"Calculated threshold is {tr}")

    x_test_predictions = model.predict(x_test)
    print("Calculating MSE on test set...")
    mse_test = np.mean(np.power(x_test - x_test_predictions, 2), axis=1)
    over_tr = mse_test > tr
    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")


# %%
def create_model(input_dim):
    autoencoder = Sequential()
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    autoencoder.add(Dense(input_dim))
    return autoencoder


# %%
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    warnings.simplefilter('ignore')

    np.random.seed(0)

    emnist_train, _ = tff.simulation.datasets.emnist.load_data()

    sample_clients = emnist_train.client_ids[0:FLAGS.n_clients]

    federated_train_data = make_federated_data(emnist_train, sample_clients)

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    preprocessed_example_dataset = preprocess(example_dataset)
    input_spec = preprocessed_example_dataset.element_spec

    def model_fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
        return tff.learning.from_keras_model(
            model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))

    # Set the default executor to be a RemoteExecutor
    tff.framework.set_default_executor(
        tff.framework.create_executor_factory(make_remote_executor))

    state = iterative_process.initialize()

    state, metrics = iterative_process.next(state, federated_train_data)
    print('round  1, metrics={}'.format(metrics))

    for round_num in range(2, FLAGS.n_rounds + 1):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))


# %%
if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    train(*sys.argv[1:])
