import numpy as np
import torch
import torch.nn as nn
from numpy.array_api import float32


def supervised_training(agent, data, epochs = 100, lr = 0.01, train_decoder = False):
    model = agent.s2m if train_decoder else agent.m2s

    optimizer = torch.optim.Adam(model.parameters())
    epoch = 30
    # MSE loss will calculate Mean Squared Error between the inputs
    loss_function = nn.MSELoss()

    print('====Training start====')
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            # prepare input data
            # data = data.cuda()
            inputs = torch.reshape(data, (-1, 784))  # -1 can be any value. So when reshape, it will satisfy 784 first

            # set gradient to zero
            optimizer.zero_grad()

            # feed inputs into model
            recon_x = model(inputs)

            # calculating loss
            loss = loss_function(recon_x, inputs)

            # calculate gradient of each parameter
            loss.backward()
            train_loss += loss.item()

            # update the weight based on the gradient calculated
            optimizer.step()
