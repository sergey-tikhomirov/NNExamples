from array import array

import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

import SimpleNN
import DeepNN
import TrainFunction

import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy



delta = 0.1
dim = 2

nHidden = 20
nHiddenDeep = 6

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

A = torch.tensor([[0.0, -1.0],
                  [1.0, 0.0]])
b = torch.tensor([0, 0])
vMin = -2
vMax = 2

def RHS(xin):
    return dampedPendulum(xin)
    #return linearSystem(xin)

modelNNEuler = SimpleNN.SimpleNN(in_features = dim, hidden_size=nHidden, out_features=dim)
#SimpleNN.optimizer = optim.SGD(modelNNEuler.parameters(), lr=0.01, weight_decay=0.0001)
SimpleNN.optimizer = optim.Adam(modelNNEuler.parameters(), lr=0.01)
SimpleNN.loss_fn = nn.MSELoss()

modelNNEuler100 = SimpleNN.SimpleNN(in_features = dim, hidden_size=nHidden, out_features=dim)

def linearSystem(xin):
    return xin @ A.T + b

#################################

#modelNNEuler = DeepNN.DeepNN(in_features = dim, hidden_size=nHiddenDeep, hidden_layer=3, out_features=dim)
##SimpleNN.optimizer = optim.SGD(modelNNEuler.parameters(), lr=0.01, weight_decay=0.0001)
#DeepNN.optimizer = optim.Adam(modelNNEuler.parameters(), lr=1.3)
#DeepNN.loss_fn = nn.MSELoss()

#modelNNEuler100 = DeepNN.DeepNN(in_features = dim, hidden_size=nHidden, hidden_layer=6, out_features=dim)

vMin = -7
vMax = 7
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

def EulerStep(xin):
    return delta * RHS(xin)

def TrajectoryEuler(xin, nSteps):
    return Trajectory(xin, nSteps, EulerStep)

#####################################################

def Euler100Step(xin):
    N = 2
    cur = xin.clone()
    for i in range(N):
        cur += delta * RHS(cur)/N
    return cur-xin

def TrajectoryEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, Euler100Step)

#######################################

def TrainNNEuler(nPoints):
    for i in range(nPoints):
        x = RandomPoint()
        y = EulerStep(x)
        modelNNEuler.TrainingStep(x, y)

def NNEulerStep(xin):
    return modelNNEuler(xin).detach()

def TrajectoryNNEuler(xin, nSteps):
    return Trajectory(xin, nSteps, NNEulerStep)

######################################################

#######################################

def TrainNNEuler100(nPoints):
    for i in range(nPoints):
        x = RandomPoint()
        y = Euler100Step(x)
        modelNNEuler.TrainingStep(x, y)

def NNEuler100Step(xin):
    return modelNNEuler(xin).detach()

def TrajectoryNNEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, NNEuler100Step)

#######################################

deepNNEuler100 = DeepNN.DeepNN(in_features = dim, hidden_layer = 6, hidden_size=30, out_features=dim)
#SimpleNN.optimizer = optim.SGD(modelNNEuler.parameters(), lr=0.01, weight_decay=0.0001)
DeepNN.optimizer = optim.Adam(deepNNEuler100.parameters(), lr=0.01)
DeepNN.loss_fn = nn.MSELoss()

def TrainDeepNNEuler100(nBlocks, blockSize = 1):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = Euler100Step(x)
        deepNNEuler100.TrainingStep(x, y)

def DeepNNEuler100Step(xin):
    return deepNNEuler100(xin).detach()

def TrajectoryDeepNNEuler100(xin, nSteps):
    return Trajectory(xin, nSteps, DeepNNEuler100Step)

######################################################

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

def RandomPoint2():
    al = 1/1.42
    return (vMin+vMax)/2 - al*(vMax-vMin)/2 + (vMax - vMin) * torch.rand(1, dim) * al

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

    #trajEuler = TrajectoryEuler(x1, 50)
    #plotPoints(trajEuler, 'Euler')

    trajEuler = TrajectoryEuler100(x1, 100)
    plotPoints(trajEuler, 'EulerN')

    #TrainNNEuler(100)
    #trajNNEuler = TrajectoryNNEuler(x2, 50)
    #plotPoints(trajNNEuler, 'NNEuler:100')

    #TrainNNEuler(1000)
    #trajNNEuler = TrajectoryNNEuler(x3, 100)
    #plotPoints(trajNNEuler, 'NNEuler:1000')

    #TrainNNEuler(10000)
    #trajNNEuler = TrajectoryNNEuler(x4, 50)
    #plotPoints(trajNNEuler, 'NNEuler10000')

    #TrainNNEuler100(1000)
    #trajNNEuler = TrajectoryNNEuler100(x5, 100)
    #plotPoints(trajNNEuler, 'NNEulerN:1000')

    #TrainNNEuler100(10000)
    #trajNNEuler = TrajectoryNNEuler100(x6, 50)
    #plotPoints(trajNNEuler, 'NNEulerN:10000')

    TrainDeepNNEuler100(1000, 1000)
    trajDeepNNEuler = TrajectoryDeepNNEuler100(x7, 100)
    plotPoints(trajDeepNNEuler, 'DeepNNEulerN:10000')

#circle
#good Sigmoid, layers = 2, size = 6, nBlocks = 1000, blockSize = 10
#good ReLU, layers = 2, size = 6, nBlocks = 1000, blockSize = 10
#good ReLU, layers = 3, size = 6, nBlocks = 1000, blockSize = 10
#good ReLU, layers = 4, size = 6, nBlocks = 1000, blockSize = 10

# pendulum
# good Sigmoid, layers = 2, size = 40, nBlocks = 1000, blockSize = 100
# bad Sigmoid, layers = 2, size = 20, nBlocks = 1000, blockSize = 100
# bad Sigmoid, layers = 2, size = 20, nBlocks = 1000, blockSize = 1000
# bad Sigmoid, layers = 2, size = 20, nBlocks = 10000, blockSize = 1000
# good Sigmoid, layers = 2, size = 30, nBlocks = 10000, blockSize = 1000
# good Sigmoid, layers = 2, size = 30, nBlocks = 1000, blockSize = 1000
# bad Sigmoid, layers = 2, size = 30, nBlocks = 1000, blockSize = 100
# good Sigmoid, layers = 3, size = 30, nBlocks = 1000, blockSize = 1000
# good ReLU, layers = 3, size = 30, nBlocks = 1000, blockSize = 1000
# good ReLU, layers = 6, size = 30, nBlocks = 1000, blockSize = 1000

    plt.legend()
    plt.show()

def CalculateQualityFunc(numTests, model, func):
    inputs = vMin + (vMax - vMin) * torch.rand(numTests, dim)
    inputs1 = inputs.clone()
    outputs = model(inputs)
    trueOutput = func(inputs1)

    print(outputs)
    print(trueOutput)

    distances = torch.norm(outputs - trueOutput, dim=1)
    #print(distances)
    print('L2 metric:', distances.square().mean().sqrt().item())
    print('Linfty metric:', distances.max().item())
    print('-------------')

def TestNNEuler():
    TrainNNEuler(1000)
    CalculateQualityFunc(1000, modelNNEuler, EulerStep)
    #trajNNEuler = TrajectoryNNEuler(x3, 10)
    #plotPoints(trajNNEuler, 'NNEuler1000')

#TestNNEuler()

DrawTrajectories()








