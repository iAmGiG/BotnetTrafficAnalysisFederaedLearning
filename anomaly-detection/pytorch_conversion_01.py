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
import syft as sy
from syft.federated.floptimizer import Optims

# %%
REBUILD_DATA = False  # set to true to one once, then back to false unless you want to change something in your training data.

hook = sy.TorchHook(torch)
v_hook = sy.VirtualWorker(hook=hook, id="v")
# x_hook = sy.VirtualWorker(hook=hook, id="x")

workers = ['v']


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
    return df, top_n_features


# %%
def create_scalar(x_opt, x_test, x_train):
    scalar = StandardScaler()
    scalar.fit(x_train.append(x_opt))
    x_train = scalar.transform(x_train)
    x_opt = scalar.transform(x_opt)
    x_test = scalar.transform(x_test)
    return x_train, x_opt, x_test


# %%
def train(net, x_train, x_opt, BATCH_SIZE, EPOCHS, input_dim):
    outputs = 0
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    loss = 0
    batch_y = 0
    optims = Optims(workers, optim=optim.SGD(params=net.parameters(), lr=0.001))
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(x_train), BATCH_SIZE)):
            # batch_x = x_train[i:i + BATCH_SIZE]
            # batch_x = x_train.view(i, BATCH_SIZE)
            # print("bx", batch_x.size())

            batch_y = x_opt[i:i + BATCH_SIZE]
            batch_y = batch_y.send('v')
            net.send(batch_y.location)
            opt = optims.get_optim(batch_y.location.id)
            # batch_y = x_train.view(-1, 784)
            # print("by", batch_y.size())
            opt.zero_grad()
            # batch_x.view(batch_y.shape[0])
            outputs = net(batch_y)
            # print('out', outputs)

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  # Does the update
            net.get()

        print(f"Epoch: {epoch}. Loss: {loss.get()}")
        # print("opt", x_opt.size(), "output", outputs.__sizeof__())

    return np.mean(np.power(batch_y.data.numpy() - outputs.data.numpy(), 2), axis=1)


# %%
def cal_threshold(mse):
    # mse = np.mean(np.power(loss_val.real, 2), axis=1)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())

    tr = mse.mean() + mse.std()
    with open(f'threshold_{input_dim}', 'w') as t:
        t.write(str(tr))
    print(f"Calculated threshold is {tr}")
    return tr


# %%
def test(net, x_test, tr):
    correct = 0
    total = 0
    x_test_predictions = net(x_test)
    print("Calculating MSE on test set...")
    mse_test = np.mean(np.power(x_test.data.numpy() - x_test_predictions.data.numpy(), 2), axis=1)
    over_tr = mse_test > tr
    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")

    # with torch.no_grad():
    #    for i in tqdm(range(len(x_test))):
    #        real_class = torch.argmax(x_test[i])
    #        net_out = net(x_test[i])
    #        predicted_class = torch.argmax(net_out)
    #        if predicted_class == real_class:
    #            correct += 1
    #        total += 1
    # print("Accuracy: ", round(correct / total, 3))


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
        return torch.softmax(x, dim=1)


# %%
net = Net(*sys.argv[1:])

# %%
training_data, input_dim = get_train_data(*sys.argv[1:])
x_train, x_opt, x_test = np.split(training_data.sample(frac=1, random_state=1),
                                  [int(1 / 3 * len(training_data)),
                                   int(2 / 3 * len(training_data))])

# %%
x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)

# %%
BATCH_SIZE = 64
EPOCHS = 1

# %%
mse = train(net,
            torch.from_numpy(x_train).float(),
            torch.from_numpy(x_opt).float(),
            BATCH_SIZE,
            EPOCHS,
            input_dim=input_dim)

tr = cal_threshold(mse=mse)
print(tr)

# %%
test(net,
     torch.from_numpy(x_test).float(), tr=1)
