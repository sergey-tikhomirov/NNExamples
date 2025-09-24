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
delta = 0.01
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

def trainModel(modelnn, func, nBlocks, blockSize, nSteps = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        for j in range(nSteps):
            y = func(x)
            modelnn.TrainingStep(x, y)
            x += func(x)

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

_ax = None

def get_or_create_3d(fig, position=111, **kwargs):
    global _ax
    if _ax is None:
        print('New Subplot')
        _ax = fig.add_subplot(position, projection="3d", **kwargs)
    return _ax

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
        #ax = next((a for a in fig.axes if getattr(a, "name", "") == "3d"), None)
        #if ax is None:
        #    print('New SubPlot')
        #    ax = fig.add_subplot(111, projection="3d")  # create it if missing
        ax = get_or_create_3d(fig)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, label = lbl)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        #ax.set_title(lbl)
        # make axes roughly equal scale
        ax.set_xlim(1*vMin, 1*vMax)
        ax.set_ylim(1*vMin, 1*vMax)
        ax.set_zlim(1*vMin, 1*vMax)

def trainAndRun(modelnn, isStep, func, nBlocks, blockSize, point, nSteps = 100, plotName = 'Plot'):
    trainModel(modelnn, func, nBlocks, blockSize)
    if isStep:
        print('From Step')
        traj = TrajectoryFromStep(point, nSteps, modelnn.forward)
    else:
        print('From Next')
        traj = TrajectoryFromNext(point, nSteps, modelnn.forward)
    plotPoints(traj, plotName)



def DrawTrajectories(nSteps, nBlocks, blockSize):
    x1 = RandomPoint()
    x2 = x1.clone()
    x3 = x1.clone()
    x4 = x1.clone()
    x5 = x1.clone()
    x6 = x1.clone()

    #trajEuler = TrajectoryFromStep(x1, nSteps, EulerStep)
    #plotPoints(trajEuler, 'Euler')

    trajEulerN = TrajectoryFromStep(x2, nSteps, Euler100Step)
    plotPoints(trajEulerN, 'EulerN')

    nnStep2_40 = DeepNN.DeepNN(in_features = dim, hidden_layer = 2, hidden_size=40, out_features=dim)
    #nnStep2_40.activation = nn.ReLU()
    nnStep2_40.activation = nn.Sigmoid()
    DeepNN.optimizer = optim.Adam(nnStep2_40.parameters(), lr=0.1)
    DeepNN.loss_fn = nn.MSELoss()
    trainAndRun(nnStep2_40, point=x3, isStep=True, func = Euler100Step, nBlocks = nBlocks, blockSize = blockSize, plotName='NN EulerN 2-40')

    #nnStep2_400 = DeepNN.DeepNN(in_features = dim, hidden_layer = 2, hidden_size=400, out_features=dim)
    #nnStep2_400.activation = nn.ReLU()
    #nnStep2_400.activation = nn.Sigmoid()
    #DeepNN.optimizer = optim.Adam(nnStep2_400.parameters(), lr=0.1)
    #DeepNN.loss_fn = nn.MSELoss()
    #trainAndRun(nnStep2_400, point=x4, isStep=True, func = Euler100Step, nBlocks = nBlocks, blockSize = blockSize, plotName='NN EulerN 2-400')

    #nnStep4_400 = DeepNN.DeepNN(in_features = dim, hidden_layer = 4, hidden_size=40, out_features=dim)
    #nnStep4_400.activation = nn.ReLU()
    #nnStep4_400.activation = nn.Sigmoid()
    #DeepNN.optimizer = optim.Adam(nnStep4_400.parameters(), lr=0.1)
    #DeepNN.loss_fn = nn.MSELoss()
    #trainAndRun(nnStep4_400, point=x5, isStep=True, func = Euler100Step, nBlocks = nBlocks, blockSize = blockSize, plotName='NN EulerN 4-400')

    #nnStep2_40_nSteps = DeepNN.DeepNN(in_features = dim, hidden_layer = 2, hidden_size=40, out_features=dim)
    ##nnStep2_40_nSteps.activation = nn.ReLU()
    #nnStep2_40_nSteps.activation = nn.Sigmoid()
    #DeepNN.optimizer = optim.Adam(nnStep2_40_nSteps.parameters(), lr=0.1)
    #DeepNN.loss_fn = nn.MSELoss()
    #trainAndRun(nnStep2_40_nSteps, point=x6, isStep=True, func = Euler100Step, nBlocks = nBlocks, blockSize = blockSize, nSteps = 1000, plotName='NN EulerN 2-40-steps')

    plt.show()

DrawTrajectories(nSteps = 1000, nBlocks=1000, blockSize = 1000)
#max allowed nBlocks=10000, blockSize = 10000, nSetp = 1