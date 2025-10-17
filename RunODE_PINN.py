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

#from RunODE import nHidden

delta = 0.1
nSteps = 50
dim = 2
nLayers = 6
nHidden = 30

def RHS(xin):
    return dampedPendulum(xin)
    #return linearSystem(xin)


testNN = DeepNN.DeepNN(in_features = dim, hidden_layer = 6, hidden_size=30, out_features=dim)
DeepNN.optimizer = optim.Adam(testNN.parameters(), lr=0.01)
DeepNN.loss_fn = nn.MSELoss()

modelNN = DeepNN.DeepNN(in_features = dim + 1, hidden_layer = nLayers, hidden_size=nHidden, out_features=dim)
modelODEPINN = ODE_PINN.SimpleODEPINN(dim, modelNN, RHS, isStrong=True, icLambda = 0.1)

modelNNStrong = DeepNN.DeepNN(in_features = dim + 1, hidden_layer = nLayers, hidden_size=nHidden, out_features=dim)
modelODEPINNStrong = ODE_PINN.SimpleODEPINN(dim, modelNNStrong, RHS, True)

vMin = -7
vMax = 7
tMax = nSteps*delta
gamma = 0.6
omega2 = 8.91

def dampedPendulum(xin):
    x1, x2 = xin.unbind(dim=-1)
    dx1 = x2
    dx2 = -gamma*x2 - omega2 * torch.sin(x1)
    return torch.stack((dx1, dx2), dim=-1)

#A = torch.tensor([[1.0, 1.0],
#                  [1.0, -1.0]])
#b = torch.tensor([-2, 1])
#vMin = 0
#vMax = 2

#A = torch.tensor([[-2.0, 1.0],
#                  [1.0, -2.0]])
#b = torch.tensor([-2, 1])
#vMin = -2
#vMax = 1

#A = torch.tensor([[0.0, -1.0],
#                  [1.0, 0.0]])
#b = torch.tensor([0, 0])
#vMin = -2
#vMax = 2

def linearSystem(xin):
    return xin @ A.T + b

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
        #print(t)
        modelODEPINN.Train(x, t)
        #modelODEPINN.TrainingStep(x, t)


def TrainODEPINNStrong(nBlocks, blockSize = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        t = RandomTime(blockSize)
        modelODEPINNStrong.Train(x, t)


def TrainDeepNNEuler100(nBlocks, blockSize = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = Euler100Step(x)
        testNN.TrainingStep(x, y)


def TrajectoryODEPINN(xin, nSteps, model):
    #print(xin)
    res = xin.clone()
    for i in range(nSteps):
        xcurr = model.forward(xin, torch.tensor([[i*delta]]))
        res = torch.cat([res, xcurr], dim = 0)
    print(res)
    return res.detach().numpy()

def TestNNStep(xin):
    return testNN(xin).detach()

def TrajectoryDeepNNEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, TestNNStep)

##############################################################

def plotPoints(pts, lbl):
    #print(pts)
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

    trajEuler = TrajectoryEuler100(x1, nSteps)
    plotPoints(trajEuler, 'EulerN')

    #TrainDeepNNEuler100(1000, 1000)
    #trajDeepNN = TrajectoryDeepNNEuler100(x2, nSteps)
    #plotPoints(trajDeepNN, 'ResDeepNN')

    TrainODEPINN(1000, 1000)
    trajODEPINN = TrajectoryODEPINN(x3, nSteps, modelODEPINN)
    plotPoints(trajODEPINN, 'ODEPINN')

    TrainODEPINNStrong(100000, 1000)
    trajODEPINNStrong = TrajectoryODEPINN(x4, nSteps, modelODEPINNStrong)
    plotPoints(trajODEPINNStrong, 'ODEPINNStrong')

    plt.legend()
    plt.show()

DrawTrajectories()