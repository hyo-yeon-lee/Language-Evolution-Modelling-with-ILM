import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def imshow(img):
    npimg = img.numpy()
    # transpose: change array axis to correspond to the plt.imshow() function
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)

print(train_dataset)

batchSize = 128

# only after packed in DataLoader, can we feed the data into the neural network iteratively
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)


# Building the encoder
class Autoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2):
        super(Autoencoder, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # decoder part
        self.fc3 = nn.Linear(h_dim2, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def decoder(self, x):
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder(num_input, num_hidden_1, num_hidden_2)

optimizer = optim.Adam(model.parameters())
epoch = 30
# MSE loss will calculate Mean Squared Error between the inputs
loss_function = nn.MSELoss()

print('====Training start====')
for i in range(epoch):
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

    if i % 10 == 0:
        print('====> Epoch: {} Average loss: {:.9f}'.format(i, train_loss))
print('====Training finish====')

inputs, _ = next(iter(test_loader))
inputs_example = make_grid(inputs[:16, :, :, :], 4)
imshow(inputs_example)

# convert from image to tensor
inputs = torch.reshape(inputs, (-1, 784))

# get the outputs from the trained model
outputs = model(inputs)

# convert from tensor to image
outputs = torch.reshape(outputs, (-1, 1, 28, 28))

# show the output images
outputs_example = make_grid(outputs[:16, :, :, :], 4)
imshow(outputs_example)