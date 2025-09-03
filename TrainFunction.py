import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import SimpleNN

nIn = 3
nOut = 3
nNodes = 80

vMin = -5
vMax = 5

model = SimpleNN.SimpleNN(in_features = nIn, hidden_size=nNodes, out_features=nOut)

SimpleNN.optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
SimpleNN.loss_fn = nn.MSELoss()

def LorenzClassic(xyz):
    sigma = 10
    rho = 28
    beta = 8.0/3.0
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return [sigma*(y-x), x*(rho - z) - y, x*y - beta*z]

def lorenz(xyz, sigma=10.0, rho=28.0, beta=8.0/3.0):
    # xyz: (..., 3)
    x, y, z = xyz.unbind(dim=-1)          # each is shape (...)
    dx = sigma*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
    return torch.stack((dx, dy, dz), dim=-1)  # shape (..., 3)


def GenerateTrainingData():
    #valIn = np.random.uniform(vMin, vMax, nIn)
    x = vMin + (vMax - vMin) * torch.rand(1, 3)
    #x = torch.tensor([valIn], dtype=torch.float32)  # input: 2 real numbers
    #y = torch.tensor([Lorenz(valIn)], dtype=torch.float32)  # target: 1 real number
    y = lorenz(x)
    model.TrainingStep(x, y)

def DoTrainingFunc(numTrain):
    for i in range(numTrain):
        GenerateTrainingData()

def CalculateTrueOutput(tensor, numTests):
    result = np.array([])
    for i in range(numTests):
        result = numpy.append(result, LorenzClassic(tensor[i]), axis=0)
    return torch.tensor(result)

def CalculateQualityFunc(numTests):
    inputs = vMin + (vMax - vMin) * torch.rand(numTests, 3)
    outputs = model(inputs)
    trueOutput = lorenz(inputs)
    #trueOutput = CalculateTrueOutput(inputs, numTests)
    #print(inputs)
    #print(outputs)
    #print(trueOutput)
    distances = torch.norm(outputs - trueOutput, dim=1)
    #print(distances)
    print('L2 metric:', distances.square().mean().sqrt().item())
    print('Linfty metric:', distances.max().item())
    print('-------------')

def TrainDemonstration():
    DoTrainingFunc(10)
    CalculateQualityFunc(1000)

    DoTrainingFunc(100)
    CalculateQualityFunc(1000)

    DoTrainingFunc(1000)
    CalculateQualityFunc(1000)

    DoTrainingFunc(10000)
    CalculateQualityFunc(1000)

    DoTrainingFunc(100000)
    CalculateQualityFunc(1000)

TrainDemonstration()







