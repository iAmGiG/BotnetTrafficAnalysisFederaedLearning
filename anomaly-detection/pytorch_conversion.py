# %%
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
from glob import iglob

# %%
REBUILD_DATA = False  # set to true to one once, then back to false unless you want to change something in your training data.


# %%
def get_train_data(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    # y_train = []
    # with open("../data/labels.txt", 'r') as labels:
    #    for lines in labels:
    #        y_train.append(lines.rstrip())
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    # return df, y_train
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
def train(net, x_train, x_opt, BATCH_SIZE, EPOCHS):
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(x_train),
                            BATCH_SIZE)):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            # print(f"{i}:{i+BATCH_SIZE}")
            batch_X = x_train[i:i + BATCH_SIZE]
            batch_y = x_opt[i:i + BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")


# %%
def test(net, x_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(x_test))):
            real_class = torch.argmax(x_test[i])
            net_out = net(x_test[i])
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))


# %%
class Net(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, int(0.75 * input_dim))
        self.fc2 = nn.Linear(int(0.75 * input_dim), int(0.5 * input_dim))
        self.fc3 = nn.Linear(int(0.5 * input_dim), int(0.33 * input_dim))
        self.fc4 = nn.Linear(int(0.33 * input_dim), int(0.25 * input_dim))
        self.fc5 = nn.Linear(int(0.25 * input_dim), int(0.33 * input_dim))
        self.fc6 = nn.Linear(int(0.33 * input_dim), int(0.5 * input_dim))
        self.fc7 = nn.Linear(int(0.5 * input_dim), int(0.75 * input_dim))
        self.fc8 = nn.Linear(int(0.75 * input_dim), input_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = self.fc8(x)


# %%
net = Net(*sys.argv[1:])

# %%
training_data = get_train_data(*sys.argv[1:])
x_train, x_opt, x_test = np.split(training_data.sample(frac=1, random_state=17),
                                  [int(1 / 3 * len(training_data)),
                                   int(2 / 3 * len(training_data))])

scaler = StandardScaler()
scaler = scaler.fit(x_train.append(x_opt))
x_train = scaler.transform(x_train)
x_opt = scaler.transform(x_opt)
x_test = scaler.transform(x_test)

# %%
# X = torch.Tensor([i[0] for in i in training_data])
# y = torch.Tensor([i[1] for in i in training_data])

# %%
x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)

# %%
BATCH_SIZE = 100
EPOCHS = 12

# %%
train(net, x_train, x_opt, BATCH_SIZE, EPOCHS)
# %%
test(net, x_test)
