#!/usr/bin/python
# %%
import sys
import os
import pandas as pd
from glob import iglob
import numpy as np
# FIX Issue #15: Standardize to tensorflow.keras imports
from tensorflow.keras.models import load_model, Model, Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD


# %%
def train(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob(
        '../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)

    fisher = pd.read_csv('../data/fisher/fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    # split randomly shuffled data into 3 equal parts
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [
                                      int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    scaler = StandardScaler()
    # FIX Issue #13: Only fit scaler on training data to prevent data leakage
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_opt = scaler.transform(x_opt)
    x_test = scaler.transform(x_test)

    model = create_model(top_n_features)
    model.compile(loss="mean_squared_error",
                  optimizer="sgd")
    # Create output directories if they don't exist
    os.makedirs("models-fixed", exist_ok=True)
    os.makedirs("logs-fixed", exist_ok=True)

    cp = ModelCheckpoint(filepath=f"models-fixed/model_{top_n_features}.h5",
                         save_best_only=True,
                         verbose=0)
    tb = TensorBoard(log_dir=f"./logs-fixed",
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
    NAME = "//trainData"
    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME), histogram_freq=1, profile_batch=100000000)
    tensorboard = TensorBoard(log_dir=f"./logs-fixed",
                              histogram_freq=1,
                              profile_batch=100000000)
    print(f"Training model for all data combined")
    # FIX Bug #21: Add ModelCheckpoint callback to actually save models
    model.fit(x_train, x_train,
              epochs=5,
              batch_size=64,
              validation_data=(x_opt, x_opt),
              verbose=1,
              callbacks=[cp, tensorboard]
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
    autoencoder.add(Dense(int(0.75 * input_dim),
                    activation="tanh", input_shape=(input_dim,)))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.25 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.33 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.5 * input_dim), activation="tanh"))
    autoencoder.add(Dense(int(0.75 * input_dim), activation="tanh"))
    autoencoder.add(Dense(input_dim))
    return autoencoder


# %%
if __name__ == '__main__':
    # FIX Bug #20: Convert command-line args to int
    args = [int(arg) if arg.isdigit() else arg for arg in sys.argv[1:]]
    train(*args)
