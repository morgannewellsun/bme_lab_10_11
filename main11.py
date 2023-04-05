# Solutions code by Matt Mender, W-2021

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio  # allows for importing of .mat files
import numpy

from torch.utils.data import DataLoader, sampler, TensorDataset
import torch.nn.functional as F


class myNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_states):
        super().__init__()

        #  Input layer
        self.bn1 = nn.BatchNorm1d(input_size)  # batch normalize inputs to fc1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.do1 = nn.Dropout(0.5)

        #  Hidden layer
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_states)
        self.do2 = nn.Dropout(0.5)

        #  Output layer
        # self.bn3 = nn.BatchNorm1d(num_states)
        # self.fc3 = nn.Linear(num_states, num_states)

        # Initialization
        nn.init.uniform_(self.fc1.weight)  # , nonlinearity="relu")
        nn.init.uniform_(self.fc2.weight)  # , nonlinearity="relu")
        # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")

    def forward(self, x):

        # # input layer
        # x = self.bn1(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.do1(x)
        #
        # # hidden layer
        # x = self.bn2(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.do2(x)
        #
        # # output layer
        # x = self.bn3(x)
        # x = self.fc3(x)

        x = self.do1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.do2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x


def myfit(epochs, nntofit, loss_fn, opt, train_dl, val_dl, print_every=1):
    train_loss = torch.zeros(epochs * len(train_dl), dtype=torch.float)  # train error for every iteration
    validation_loss = torch.zeros(epochs, dtype=torch.float)  # validation is only once per epoch
    i = -1  # iteration number
    for epoch in range(epochs):

        for x, y in train_dl:  # batch of training points
            i += 1
            # Set model in train mode (for batch normalization and dropout)
            nntofit.train()

            # 1. Generate your predictions by running x through the network
            yh = nntofit(x)

            # 2. Find Loss by comparing predicted and actual using loss function
            loss = loss_fn(yh, y)

            # 3. Calculate gradients with respect to weights/biases
            loss.backward()
            train_loss[i] = loss.item()  # Keep track of loss on training data

            # 4. Adjust your weights by taking a step forward on the optimizer
            opt.step()

            # 5. Reset the gradients to zero on the optimizer
            opt.zero_grad()

        # Validation accuracy
        for xval, yval in val_dl:
            with torch.no_grad():  # disable gradient calculation
                nntofit.eval()  # set model to evaluation mode (matters for batch normalization and dropout)
                loss2 = loss_fn(nntofit(xval), yval)
                validation_loss[epoch] = loss2.item()
    return train_loss, validation_loss


def main():

    # =========================================================================

    # Import your data here
    rootDir = r"D:\Documents\Academics\BME517\bme_lab_7_8_9\data\\"
    fn = 'contdata95.mat'

    dtype = torch.float
    conv_size = 3  # size of time history

    # load the mat file
    mat = sio.loadmat(rootDir + fn)

    # Get each variable from the mat file
    X = torch.tensor(mat['Y'])
    y = torch.tensor(mat['X'])[:, 0:4]

    nsamp = X.shape[0]
    ntrain = int(numpy.round(nsamp * 0.8))  # using 80% of data for training

    X_train = X[0:ntrain, :].to(dtype)
    X_test = X[ntrain + 1:, :].to(dtype)
    y_train = y[0:ntrain, :].to(dtype)
    y_test = y[ntrain + 1:, :].to(dtype)

    # Initialize tensor with conv_size*nfeatures features
    X_ctrain = torch.zeros((int(X_train.shape[0]), int(X_train.shape[1] * conv_size)), dtype=dtype)
    X_ctest = torch.zeros((int(X_test.shape[0]), int(X_test.shape[1] * conv_size)), dtype=dtype)
    X_ctrain[:, 0:X_train.shape[1]] = X_train
    X_ctest[:, 0:X_test.shape[1]] = X_test

    # Add the previous 3 time bins features as a feature in the current time bin
    for k1 in range(conv_size - 1):
        k = k1 + 1
        X_ctrain[k:, int(X_train.shape[1] * k):int(X_train.shape[1] * (k + 1))] = X_train[0:-k, :]
        X_ctest[k:, int(X_test.shape[1] * k):int(X_test.shape[1] * (k + 1))] = X_test[0:-k, :]

    # Create Dataset and dataloader
    test_ds = TensorDataset(X_ctest, y_test)
    train_ds = TensorDataset(X_ctrain, y_train)

    # If a batch in BatchNorm only has 1 sample it wont work, so dropping the last in case that happens
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, drop_last=True)

    # =========================================================================

    # Specify a model, loss function, and optimizer
    weight_decay = 1e-2
    learning_rate = 1e-5

    # Your code here for creating the network
    theNetwork = myNetwork(285, 256, 4)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    opt = torch.optim.Adam(theNetwork.parameters())  # , weight_decay=weight_decay, lr=learning_rate)

    # =========================================================================

    # Train the network
    n_epochs = 30
    train_loss, validation_loss = myfit(n_epochs, theNetwork, loss_fn, opt, train_dl, test_dl)

    # =========================================================================

    # Plot training and validation losses
    plot_epochs = n_epochs

    val_iters = numpy.arange(0, n_epochs) * len(train_dl)
    train_iters = numpy.arange(0, len(train_dl) * n_epochs)
    n_iter = len(train_dl) * n_epochs  # number of batches per epoch * number of epochs

    plt.plot(train_iters[0:plot_epochs * len(train_dl)], train_loss[0:plot_epochs * len(train_dl)], 'b')
    plt.plot(val_iters[0:plot_epochs], validation_loss[0:plot_epochs], 'r')
    plt.xlabel('Number of iterations')
    plt.ylabel('MSE')
    plt.show()

    # =========================================================================

    # Plot some example decodes
    for x, y in test_dl:
        with torch.no_grad():
            yh = theNetwork(x)
            # looking at select channel
        th = numpy.arange(0, x.shape[0]) * 50e-3

        plt.subplot(2, 1, 1)
        plt.plot(th[1000:1500], y[1000:1500, 0], 'b')
        plt.plot(th[1000:1500], yh[1000:1500, 0].detach().numpy(), 'r')
        # plt.xlabel('sec')
        plt.ylabel('X Position')

        plt.subplot(2, 1, 2)
        plt.plot(th[1000:1500], y[1000:1500, 1], 'b')
        plt.plot(th[1000:1500], yh[1000:1500, 1].detach().numpy(), 'r')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y Position')

        plt.show()

        r = numpy.corrcoef(yh.detach().numpy().T, y.T)
        r = numpy.diag(r[4:, 0:4])
        print('Correlation for X position is %g' % r[0])
        print('Correlation for Y position is %g' % r[1])
        print('Correlation for X velocity is %g' % r[2])
        print('Correlation for Y velocity is %g' % r[3])
        print('Average Correlation: %g' % numpy.mean(r))


if __name__ == "__main__":
    main()
