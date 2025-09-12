from array import array

import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

import SimpleNN
import DeepNN
import TrainFunction
import HeunStepNN

import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy


delta = 0.1
dim = 2

#########################
vMin = -7
vMax = 7
gamma = 0.4
omega2 = 8.91

def dampedPendulum(xin):
    x1, x2 = xin.unbind(dim=-1)          # each is shape (...)
    dx1 = x2
    dx2 = -gamma*x2 - omega2 * torch.sin(x1)
    return torch.stack((dx1, dx2), dim=-1)  # shape (..., 3)

A = torch.tensor([[0.0, -1.0],
                  [1.0, 0.0]])
b = torch.tensor([0, 0])

def linearSystem(xin):
    return xin @ A.T + b

def RHS(xin):
    return dampedPendulum(xin)
    #return linearSystem(xin)

#####################################

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

def trainModel(modelnn, func, nBlocks, blockSize):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = func(x)
        modelnn.TrainingStep(x, y)

def trainModelConservative(modelnn, func, nBlocks, blockSize, al):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = func(x)
        modelnn.TrainingStepConservation(x, y, al)



def EulerStep(xin):
    return delta * RHS(xin)

def Euler100Step(xin):
    N = 2
    cur = xin.clone()
    for i in range(N):
        cur += delta * RHS(cur)/N
    return cur-xin

def NextEuler(xin):
    return xin + EulerStep(xin)

def NextEuler100(xin):
    return xin + Euler100Step(xin)

def TrajectoryFromStep(xin, nSteps, func):
    res = xin
    xcurr = xin.clone()
    for i in range(nSteps):
        xcurr += func(xcurr)
        res = torch.cat([res, xcurr], dim = 0)
    return res

def TrajectoryFromNext(xin, nSteps, func):
    res = xin
    xcurr = xin.clone()
    for i in range(nSteps):
        xcurr = func(xcurr)
        res = torch.cat([res, xcurr], dim = 0)
    return res

def plotPoints(pts, lbl):
    pts = pts.detach().numpy()
    if dim == 2:
        # 2D scatter
        plt.plot(pts[:, 0], pts[:, 1], label=lbl)
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D points')
        ax = plt.gca()
        ax.set_xlim(2*vMin, 2*vMax)
        ax.set_ylim(2*vMin, 2*vMax)

def DrawTrajectories3(nSteps, nBlocks, blockSize):
    nnStep = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    DeepNN.optimizer = optim.Adam(nnStep.parameters(), lr=0.01)
    DeepNN.loss_fn = nn.MSELoss()
    trainModel(nnStep, Euler100Step, nBlocks, blockSize)

    nnNext = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    DeepNN.optimizer = optim.Adam(nnNext.parameters(), lr=0.01)
    DeepNN.loss_fn = nn.MSELoss()
    trainModel(nnNext, NextEuler100, nBlocks, blockSize)

    nnHeunInternal = DeepNN.DeepNN(in_features = dim, hidden_layer = 2, hidden_size=40, out_features=dim)
    nnNNHeun = HeunStepNN.HeunStep(nnHeunInternal)
    trainModel(nnNNHeun, NextEuler100, nBlocks, blockSize)

    nnConservation1 = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    DeepNN.optimizer = optim.Adam(nnConservation1.parameters(), lr=0.01)
    trainModelConservative(nnConservation1, Euler100Step, nBlocks, blockSize, 0.1)

    nnConservation10 = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    DeepNN.optimizer = optim.Adam(nnConservation10.parameters(), lr=0.01)
    trainModelConservative(nnConservation10, Euler100Step, nBlocks, blockSize, 1)

    x1 = RandomPoint()
    x2 = x1.clone()
    x3 = x1.clone()
    x4 = x1.clone()
    x5 = x1.clone()
    x6 = x1.clone()

    print(x1)

    trajEulerN = TrajectoryFromStep(x1, nSteps, Euler100Step)
    plotPoints(trajEulerN, 'EulerN')

    trajNNStep = TrajectoryFromStep(x2, nSteps, nnStep.forward)
    plotPoints(trajNNStep, 'NN-Step')

    trajNNNext = TrajectoryFromNext(x3, nSteps, nnNext.forward)
    plotPoints(trajNNNext, 'NN-Next')

    #trajHeunStep = TrajectoryFromStep(x4, nSteps, nnStep.forward)
    #plotPoints(trajHeunStep, 'NN-Heun')

    #trajNNNext = TrajectoryFromStep(x5, nSteps, nnConservation1.forward)
    #plotPoints(trajNNNext, 'NN-Conservation1')

    #trajNNNext = TrajectoryFromStep(x6, nSteps, nnConservation10.forward)
    #plotPoints(trajNNNext, 'NN-Conservation10')

    plt.legend()
    plt.show()


DrawTrajectories3(100, 1000, 10)


