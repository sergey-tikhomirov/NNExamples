import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

import SimpleNN
import DeepNN
import TrainFunction
#import HeunStepNN
import ODE_PINN

import matplotlib.pyplot as plt

from RunODE import nHidden

delta = 0.1
dim = 2
nHidden = 40

def RHS(xin):
    return dampedPendulum(xin)


modelNN = SimpleNN.SimpleNN(in_features = dim, hidden_size=nHidden, out_features=dim)
modelODEPINN = ODE_PINN.SimpleODEPINN(modelNN, RHS)

vMin = -7
vMax = 7
tMax = 10
gamma = 0.4
omega2 = 8.91


def dampedPendulum(xin):
    x1, x2 = xin.unbind(dim=-1)          # each is shape (...)
    dx1 = x2
    dx2 = -gamma*x2 - omega2 * torch.sin(x1)
    return torch.stack((dx1, dx2), dim=-1)  # shape (..., 3)

##########################################

def Trajectory(xin, nSteps, func):
    res = xin
    xcurr = xin.clone()
    for i in range(nSteps):
        xcurr += func(xcurr)
        res = torch.cat([res, xcurr], dim = 0)
    return res

#####################################################

def Euler100Step(xin):
    N = 2
    cur = xin.clone()
    for i in range(N):
        cur += delta * RHS(cur)/N
    return cur-xin

def TrajectoryEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, Euler100Step)

######################################################

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

def RandomTime(blockSize = 1):
    return 0 + (tMax - 0) * torch.rand(blockSize, 1)

def TrainODEPINN(nBlocks, blockSize = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        t = RandomTime(blockSize)
        modelODEPINN.TrainingStep(x, t)


def TrajectoryODEPINN(xin, nSteps):
    res = xin
    for i in range(nSteps):
        xcurr = modelODEPINN(xin, i*delta)
        res = torch.cat([res, xcurr], dim = 0)
    return res

##############################################################

def plotPoints(pts, lbl):
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

def DrawTrajectories():
    x1 = RandomPoint()
    x2 = x1.clone()
    x3 = x1.clone()
    x4 = x1.clone()
    x5 = x1.clone()
    x6 = x1.clone()
    x7 = x1.clone()

    print(x1)

    trajEuler = TrajectoryEuler100(x1, 100)
    plotPoints(trajEuler, 'EulerN')

    TrainODEPINN(1000, 100)
    trajHeunNNEuler = TrajectoryODEPINN(x2, 100)
    plotPoints(trajHeunNNEuler, 'ODEPINN')