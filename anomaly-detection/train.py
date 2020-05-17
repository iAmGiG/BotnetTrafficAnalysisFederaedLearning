#!/usr/bin/python
# %%
import sys
# import os
from typing import Callable

import pandas as pd
from glob import iglob
import numpy as np
# from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.optimizers import SGD
import tensorflow_federated as tff
import tensorflow as tf
from tensorflow_federated.python.learning.model_utils import EnhancedModel

reading_type = tff.FederatedType(tf.float64, tff.CLIENTS)
server_type = tff.FederatedType(tf.float64, tff.SERVER)


# %%
# the body of the py function was traced once, disposed of, and replaced by serialized abstract representation in
# tff lang
#
@tff.federated_computation(reading_type)
def get_positives_on_dataset_without_attacks(inner_put):
    print(tff.federated_mean(inner_put))
    return tff.federated_mean(inner_put)


# %%
def get_train_data(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    return df


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
    df = get_train_data(top_n_features)
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

    # fed part
    # training is represent as a par of computations
    # one the initialize state
    # two the single round execution
    # both can be executed like functions in python.
    # and when we do they by default execute in a local simulation <------------------important for paper writing
    # and perform small simulation groups..
    # the state includes the model and the train data (both of which are above have above)
    # *at the time of this writing debugging as not begun fully mid May2020*
    # %%
    """input_spec: (Optional) a value convertible to `tff.Type` specifying the type
      of arguments the model expects. Notice this must be a compound structure
      of two elements, specifying both the data fed into the model to generate
      predictions, as its first element, as well as the expected type of the
      ground truth as its second. This argument will become required when we
      remove `dummy_batch`; currently, exactly one of these two must be
      specified."""
    model_fn = model_function(model)
    client_optimizer_fn = tf.keras.optimizers.SGD()
    train = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn)
    state = train.initialize()
    for _ in range(5):
        state, metrics = train.next(state, x_train)
        print(metrics.loss)
    # %%
    #
    #

    # threshold calculation
    tr = calculating_threshold(model, top_n_features, x_opt)
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    metrics = evaluation(state.model, x_opt)

    # prediction time, then the comparision of over the threshold, any false_positives,
    x_test_predictions = model.predict(x_test)
    print("Calculating MSE on test set...")

    #  Returns the average of the array elements. The average is taken over
    #  the flattened array by default, otherwise over the specified axis.
    #  `float64` intermediate and return values are used for integer inputs.
    mse_test = np.mean(np.power(x_test - x_test_predictions, 2), axis=1)
    over_tr = mse_test > tr
    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")
    return mse_test


def model_function(keras_model):
    model_fn = tff.learning.from_keras_model(keras_model,
                                             loss=keras_model.loss,
                                             input_spec=keras_model.input_spec,
                                             loss_weights=keras_model.loss_weights,
                                             metrics=keras_model.metrics)
    return model_fn


# %%
# so in case of this situation, the server would need the input dimensions, some how, not important now.
# to have the model_function = lambda: tff.learning.from_keras_model(create_model) as my model line.
# i need to be able to create it on the server before sending the model to the clients.......
# yes....i know what this takes.....yep......???????
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
# train = tff.learning.build_federated_averaging_process(train_fn(*sys.argv[1:]))
if __name__ == '__main__':
    jput = train_fn(*sys.argv[1:])
    print(jput)
