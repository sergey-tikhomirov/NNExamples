import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the network
class SimpleNN(nn.Module):
    def __init__(self, in_features = 2, hidden_size=7, out_features=1):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(in_features, hidden_size)   # input: 2 → hidden layer
        #self.activation = nn.ReLU()              # nonlinearity
        self.activation = nn.Sigmoid()              # nonlinearity
        self.output = nn.Linear(hidden_size, out_features)  # hidden → output: 1 real number

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def TrainingStep(self, x, y):
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()




