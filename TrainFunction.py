import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import SimpleNN

nIn = 3
nOut = 3
nNodes = 8

vMin = -5
vMax = 5

model = SimpleNN.SimpleNN(in_features = nIn, hidden_size=nNodes, out_features=nOut)

optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
loss_fn = nn.MSELoss()

def Lorenz(xyz):
    sigma = 10
    rho = 28
    beta = 8.0/3.0
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return [sigma*(y-x), x*(rho - z) - y, x*y - beta*z]

def GenerateTrainingData():
    valIn = np.random.uniform(vMin, vMax, nIn)
    x = torch.tensor([valIn], dtype=torch.float32)  # input: 2 real numbers
    y = torch.tensor([Lorenz(valIn)], dtype=torch.float32)  # target: 1 real number
    model.TrainingStep(x, y)

def DoTrainingFunc(numTrain):
    for i in range(numTrain):
        GenerateTrainingData()

def CalculateQualityFunc(numTests):
    inputs = vMin + (vMax - vMin) * torch.rand(numTests, 3)
    outputs = model(inputs)
    trueOutput = Lorenz(inputs)
    distances = torch.norm(outputs - trueOutput, dim=1)
    print('L2 metric:', distances.mean())
    print('Linfty metric:', distances.max())


DoTrainingFunc(10)
CalculateQualityFunc(1000)

DoTrainingFunc(100)
CalculateQualityFunc(1000)

DoTrainingFunc(1000)
CalculateQualityFunc(1000)

DoTrainingFunc(10000)
CalculateQualityFunc(1000)









