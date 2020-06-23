#!/usr/bin/python
# %%
import sys
from glob import iglob
import functools
import os
from collections.abc import Callable
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.optimizers import SGD
import tensorflow_federated as tff
# from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.optimization.emnist import dataset
from tensorflow_federated.python.research.compression import compression_process_adapter
from tensorflow_federated.python.research.compression import sparsity
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

with utils_impl.record_new_flags() as hparam_flags:
    utils_impl.define_optimizer_flags('server')
    utils_impl.define_optimizer_flags('client')
    flags.DEFINE_integer('clients_per_round', 2,
                         'How many clients to sample per round.')
    flags.DEFINE_integer('client_epochs_per_round', 1,
                         'Number of epochs in the client to take per round.')
    flags.DEFINE_integer('client_batch_size', 20,
                         'Batch size used on the client.')
    flags.DEFINE_boolean('use_compression', False,
                         'Whether to use compression code path.')
    flags.DEFINE_boolean('use_sparsity_in_aggregation', False,
                         'Whether to add sparsity to the aggregation. This will '
                         'only be used for client to server compression.')

FLAGS = flags.FLAGS


# %%
def get_train_data(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    y_train = []
    with open("../data/labels.txt", 'r') as labels:
        for lines in labels:
            y_train.append(lines.rstrip())
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    return df, y_train


# %%
def create_scalar(x_opt, x_test, x_train):
    scalar = StandardScaler()
    scalar.fit(x_train.append(x_opt))
    x_train = scalar.transform(x_train)
    x_opt = scalar.transform(x_opt)
    x_test = scalar.transform(x_test)
    return x_train, x_opt, x_test


# %%
def calculating_threshold(model, top_n_features, x_opt):
    # threshold
    print("Calculating threshold")
    x_opt_predictions = model.predict(x_opt)
    # mean squared error
    print("Calculating MSE on optimization set...")
    mse = np.mean(np.power(x_opt - x_opt_predictions, 2), axis=1)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())
    # threshold calculation
    tr = mse.mean() + mse.std()
    with open(f"threshold_{top_n_features}", 'w') as t:
        t.write(str(tr))
    print(f"Calculated threshold is {tr}")
    return tr


# %%
# break monolith into multiple parts
# data gathering
# data splitting
# data transformation
# fitting
#
# so need to be able to call this local train function, not directly, but from within a few method, that is the
# federated computation handler method.
def train_fn(top_n_features=10):
    """
    the federated computation is much to the input of the data, not as much as pulling data out of the process.
    this does not appear obvious as the @ may indicate some callable, but this is a strongly type solution to avoid
    path-ing in the traditional way.
    the main reason this appears to be done is to have the ability to just broadcast on the network to listening nodes.
    these nodes would then received a package from some server, and it has to be ready to retrieve the packages in a
    generic way.
    so we tell it what to possibly expect, rather than what to exactly expect
    this is why we see in the examples that the high level federated computation is a simple MVC style method, is that
    the other methods will be callable from within the code.
    :param top_n_features:
    :return:
    """
    df, y_train = get_train_data(top_n_features)
    # split randomly shuffled data into 3 equal parts
    # this need reevaluation for a real time state one day.
    # what does the data look like split up across a real time state?
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    # craft scalar
    x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)
    # create the model from the top features
    model = create_model(top_n_features)
    # compile the model
    model.compile(
        loss="mean_squared_error",
        optimizer="sgd")

    # all of the call backs
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
    # end oc all backs
    print(f"Training model for all data combined")

    # fitting the model
    model.fit(x_train,
              x_train,
              epochs=1,
              batch_size=64,
              validation_data=(x_opt, x_opt),
              verbose=1,
              callbacks=[tensorboard]
              )

    # %%

    # threshold calculation
    tr = calculating_threshold(model, top_n_features, x_opt)

    # prediction time, then the comparision of over the threshold, any false_positives,
    x_test_predictions = model.predict(x_test)
    print("Calculating MSE on test set...")

    #  Returns the average of the array elements. The average is taken over
    #  the flattened array by default, otherwise over the specified axis.
    #  `float64` intermediate and return values are used for integer inputs.
    mse_test = np.mean(np.power(x_test - x_test_predictions, 2), axis=1)

    over_tr = exceeds_threshold_fn(mse_test, tr)

    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")
    return tff.federated_value(tff.federated_map(exceeds_threshold_fn, [top_n_features, tff.federated_broadcast(tr)]))


# %%
# @tff.tf_computation(tf.float64, tf.float64)
def exceeds_threshold_fn(mse_test, tr):
    return mse_test > tr


# %%
# so in case of this situation, the server would need the input dimensions, some how, not important now.
# to have the model_function = lambda: tff.learning.from_keras_model(create_model) as my model line.
# i need to be able to create it on the server before sending the model to the clients.......
# yes....i know what this takes.....yep......???????
def create_model(input_dim):
    autoencoder = Sequential([
        tf.keras.layers.Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)),
        tf.keras.layers.Dense(int(0.5 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(int(0.33 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(int(0.25 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(int(0.33 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(int(0.5 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(int(0.75 * input_dim), activation="tanh"),
        tf.keras.layers.Dense(input_dim)
    ])
    # autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh", input_shape=(input_dim,)))
    # autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    # autoencoder.add(Dense(input_dim))
    return autoencoder


# %%
def model_builder(input_dim, input_spec):
    model = create_model(input_dim)
    return tff.learning.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.MeanSquaredError(),
        input_spec=input_spec,
        metrics=[tf.keras.metrics.Accuracy()],
    )

def _broadcast_encoder_fn(value):
    """Function for building encoded broadcast.

    This method decides, based on the tensor size, whether to use lossy
    compression or keep it as is (use identity encoder). The motivation for this
    pattern is due to the fact that compression of small model weights can provide
    only negligible benefit, while at the same time, lossy compression of small
    weights usually results in larger impact on model's accuracy.

    Args:
      value: A tensor or variable to be encoded in server to client communication.

    Returns:
      A `te.core.SimpleEncoder`.
    """
    # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
    # currently support Variables.
    spec = tf.TensorSpec(value.shape, value.dtype)
    if value.shape.num_elements() > 10000:
        return te.encoders.as_simple_encoder(
            te.encoders.uniform_quantization(FLAGS.broadcast_quantization_bits),
            spec)
    else:
        return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def _mean_encoder_fn(value):
    """Function for building encoded mean.

    This method decides, based on the tensor size, whether to use lossy
    compression or keep it as is (use identity encoder). The motivation for this
    pattern is due to the fact that compression of small model weights can provide
    only negligible benefit, while at the same time, lossy compression of small
    weights usually results in larger impact on model's accuracy.

    Args:
      value: A tensor or variable to be encoded in client to server communication.

    Returns:
      A `te.core.GatherEncoder`.
    """
    # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
    # currently support Variables.
    spec = tf.TensorSpec(value.shape, value.dtype)
    if value.shape.num_elements() > 10000:
        if FLAGS.use_sparsity_in_aggregation:
            return te.encoders.as_gather_encoder(
                sparsity.sparse_quantizing_encoder(
                    FLAGS.aggregation_quantization_bits), spec)
        else:
            return te.encoders.as_gather_encoder(
                te.encoders.uniform_quantization(FLAGS.aggregation_quantization_bits),
                spec)
    else:
        return te.encoders.as_gather_encoder(te.encoders.identity(), spec)



# %%
def train_main(sysarg=10):
    emnist_train, emnist_test = dataset.get_emnist_datasets(
        FLAGS.client_batch_size,
        FLAGS.client_epochs_per_round,
        only_digits=FLAGS.only_digits)

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])
    input_spec = example_dataset.element_spec

    #
    # These are the attempts to use the custom data set for this experiment
    # begin
    # df, y_train = get_train_data(sysarg)
    # x_train, x_opt, x_test = np.split(df.sample(frac=1,random_state=17),[int(1 / 3 * len(df)), int(2 / 3 * len(df))])

    # x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)
    # will look into moving this into its own method if successful
    # end
    #

    # defining the input spec
    # input_spec = tf.nest.map_structure(tf.RaggedTensor.from_tensor,
    # (tf.ragged.constant(x_train), tf.ragged.constant(y_train)))
    # an assign weight function
    assign_weights_fn = compression_process_adapter.CompressionServerState.assign_weights_to_keras_model

    #
    # the client/server optimizer functions
    #
    client_optimizer_fn = functools.partial(
        utils_impl.create_optimizer_from_flags, 'client')
    server_optimizer_fn = functools.partial(
        utils_impl.create_optimizer_from_flags, 'server')
    #
    if FLAGS.use_compression:
        # We create a `StatefulBroadcastFn` and `StatefulAggregateFn` by providing
        # the `_broadcast_encoder_fn` and `_mean_encoder_fn` to corresponding
        # utilities. The fns are called once for each of the model weights created
        # by tff_model_fn, and return instances of appropriate encoders.
        encoded_broadcast_fn = (
            tff.learning.framework.build_encoded_broadcast_from_model(
                functools.partial(model_builder,
                                  input_dim=sysarg,
                                  input_spec=input_spec), _broadcast_encoder_fn))
        encoded_mean_fn = tff.learning.framework.build_encoded_mean_from_model(
            functools.partial(model_builder,
                              input_dim=sysarg,
                              input_spec=input_spec), _mean_encoder_fn)

    # defines the iterative process, takes a model function, a client optimizer,
    # server optimizer delta aggregate and model broadcast function
    # iterative process
    # will need to look into stateful_delta_agg func and stateful model broadcast fn going forward.
    # begin
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=functools.partial(model_builder,
                                   input_dim=sysarg,
                                   input_spec=input_spec),
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        stateful_delta_aggregate_fn=encoded_mean_fn,
        stateful_model_broadcast_fn=encoded_broadcast_fn
    )
    iterative_process = compression_process_adapter.CompressionProcessAdapter(iterative_process)
    #
    # end
    #

    # client dataset function
    client_db_fn = training_utils.build_client_datasets_fn(
        train_dataset=emnist_train,
        train_clients_per_round=FLAGS.clients_per_round)

    # evaluation function
    eval_fn = training_utils.build_evaluate_fn(
        eval_dataset=emnist_test,
        model_builder=create_model(sysarg),
        loss_builder=tf.keras.losses.MeanSquaredError(),
        metrics_builder=[tf.keras.metrics.Accuracy()],
        assign_weights_to_keras_model=assign_weights_fn
    )

    # training loop
    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_db_fn,
        validation_fn=eval_fn
    )

    # %%
    def main(argv):
        if len(argv) > 2:
            raise app.UsageError('Expected one command-line argument(s), '
                                 'got: {}'.format(argv))

        tff.framework.set_default_executor(
            tff.framework.local_executor_factory(max_fanout=25))
        train_main()
        # train_fn(*sys.argv[1:])

    # %%
    # train = tff.learning.build_federated_averaging_process(train_fn(*sys.argv[1:]))
    if __name__ == '__main__':
        app.run(main)
