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
from mpl_toolkits.mplot3d import Axes3D
from sympy.stats.sampling.sample_numpy import numpy

dim = 3
delta = 0.1
vMax = 50.0
vMin = -50.0

sigma = 10
rho = 28
beta = 8.0/3.0

def lorenzMap(xin):
    x1, x2, x3 = xin.unbind(dim=-1)          # each is shape (...)
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1*x2 - beta*x3
    return torch.stack((dx1, dx2, dx3), dim=-1)  # shape (..., 3)

def RHS(xin):
    return lorenzMap(xin)

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

def trainModel(modelnn, func, nBlocks, blockSize):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = func(x)
        modelnn.TrainingStep(x, y)

def EulerStep(xin):
    return delta * RHS(xin)

def Euler100Step(xin):
    N = 2
    cur = xin.clone()
    for i in range(N):
        cur += delta * RHS(cur)/N
    return cur-xin

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
    if dim == 3:
        fig = plt.figure()
        ax = next((a for a in fig.axes if getattr(a, "name", "") == "3d"), None)
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")  # create it if missing
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(lbl)

        # make axes roughly equal scale
        ax.set_xlim(vMin, vMax)
        ax.set_ylim(vMin, vMax)
        ax.set_zlim(vMin, vMax)

def trainAndRun(modelnn, isStep, func, nBlocks, blockSize, point, nSteps = 100, plotName = 'Plot'):
    trainModel(modelnn, func, nBlocks, blockSize)
    if isStep:
        traj = TrajectoryFromStep(point, nSteps, func)
    else:
        traj = TrajectoryFromNext(point, nSteps, func)
    plotPoints(traj, plotName)

def DrawTrajectories(nSteps, nBlocks, blockSize):
    x1 = RandomPoint()
    x2 = x1.clone()
    x3 = x1.clone()
    x4 = x1.clone()
    x5 = x1.clone()
    x6 = x1.clone()

    trajEulerN = TrajectoryFromStep(x1, nSteps, Euler100Step)
    plotPoints(trajEulerN, 'EulerN')

    nnStep = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    DeepNN.optimizer = optim.Adam(nnStep.parameters(), lr=0.01)
    DeepNN.loss_fn = nn.MSELoss()
    trainAndRun(nnStep, point=x2, isStep=True, func = nnStep.forward, nBlocks = nBlocks, blockSize = blockSize, plotName='NN EulerN')

    plt.show()

DrawTrajectories(nSteps = 100, nBlocks=1000, blockSize = 100)