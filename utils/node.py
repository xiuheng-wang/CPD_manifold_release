from __future__ import division
import torch
from torch import nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
from utils.model import Net, init_weights
from torch.autograd import Variable

def node(Y, swl=64, in_and_hi_layer=[32]*3, lr=0.075, single_test=False):
    T, N = np.shape(Y)
    stat = np.zeros(T)
    model = Net(N, in_and_hi_layer).to(device = device)
    loss_function = torch.nn.BCELoss() # torch.nn.BCELoss()
    train_label = np.concatenate((np.zeros([swl, 1]), np.ones([swl, 1])), axis = 0)
    # The SGD optimizer in PyTorch is just gradient descent.
    optimizer = torch.optim.SGD(model.parameters(), lr)
    for t in range(2*swl, T):
        train_data = Y[t - 2*swl : t]
        x_train = Variable(torch.from_numpy(train_data).float().to(device = device))
        y_train = Variable(torch.from_numpy(train_label).float().to(device = device))
        if t == 2*swl:
            model.apply(init_weights)
        model.train()
        output = model(x_train)
        optimizer.zero_grad()
        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()
        # single test sample
        if single_test == True:
            test_data = Y[t]
            test_data = torch.from_numpy(test_data).float().unsqueeze(0).to(device = device)
        # multi test samples
        else:
            test_data = Y[t - swl : t]
            test_data = torch.from_numpy(test_data).float().to(device = device) 
        model.eval()
        pred = model(test_data)
        stat[t] = (pred / (1-pred) - 1).mean().abs()
    return stat