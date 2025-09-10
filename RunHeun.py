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

from RunODE import nHidden

delta = 0.1
dim = 2
nHidden = 40

modelNN = SimpleNN.SimpleNN(in_features = dim, hidden_size=nHidden, out_features=dim)
modelNNHeun = HeunStepNN.HeunStep(modelNN)

vMin = -7
vMax = 7
gamma = 0.4
omega2 = 8.91

def RHS(xin):
    return dampedPendulum(xin)

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

def RandomPoint2():
    al = 1/1.42
    return (vMin+vMax)/2 - al*(vMax-vMin)/2 + (vMax - vMin) * torch.rand(1, dim) * al

def TrainHeunNNEuler100(nBlocks, blockSize = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = Euler100Step(x)
        modelNNHeun.TrainingStep(x, y)

def HeunNNEuler100Step(xin):
    return modelNNHeun(xin).detach()

def TrajectoryHeunNNEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, HeunNNEuler100Step)

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
    x1 = RandomPoint2()
    x2 = x1.clone()
    x3 = x1.clone()
    x4 = x1.clone()
    x5 = x1.clone()
    x6 = x1.clone()
    x7 = x1.clone()

    print(x1)

    # trajEuler = TrajectoryEuler(x1, 50)
    # plotPoints(trajEuler, 'Euler')

    trajEuler = TrajectoryEuler100(x1, 100)
    plotPoints(trajEuler, 'EulerN')

    TrainHeunNNEuler100(1000, 1000)
    trajHeunNNEuler = TrajectoryHeunNNEuler100(x2, 100)
    plotPoints(trajHeunNNEuler, 'HeunNNEulerN')