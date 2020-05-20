#!/usr/bin/python
# %%
import sys
import functools
import absl
from glob import iglob
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
FLAGS = absl.flags.FLAGS

reading_type = tff.FederatedType(tf.float32, tff.CLIENTS)
threshold_type = tff.FederatedType(tf.float32, tff.SERVER)

# %%
@tff.tf_computation(reading_type)
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
@tff.federated_computation(reading_type, threshold_type)
def main_fn(reading, threshold):
    return tff.federated_value(train, [reading_type, tff.federated_broadcast(threshold_type)])

if __name__ == '__main__':
    absl.app.run(main_fn(reading_type, threshold_type))
    #train(*sys.argv[1:])

