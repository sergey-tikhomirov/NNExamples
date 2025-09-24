'''
Cubic Power ODE system with unit disk Barrier
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import ResNN
import DeepNN


####
#Parâmetros Caso particular Cubic Power
vMin = -2
vMax = 2
#####

####
#Parâmetros Codigo antigo
delta = 0.1 #É utilizado para Euler
dim = 2 #Dimensão do tensor de entrada
####

#Parametros de Treinamento
nBlocks=1000
blockSize=100
############


#####
## Funções específicas do meu problema
def cubic_power(xin):
    x1, x2 = xin.unbind(dim=-1)
    dx1 = x2 -x1*(x1*x1 +x2*x2 -1)
    dx2 = -x1-x2*(x1*x1 +x2*x2 -1) 
    return torch.stack((dx1, dx2), dim=-1)  # shape (..., 2)

def RHS(xin):
    return cubic_power(xin)

##################################
### Para graficar ###############
def Trajectory(xin, nSteps, func):
    '''
    #Salva a Trajetória, func pode ser a rede treinada ou a função exata
    '''
    res = xin
    xcurr = xin.clone()
    for i in range(nSteps):
        xcurr += func(xcurr)
        res = torch.cat([res, xcurr], dim = 0)
    return res

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

def plotPoints(pts, lbl):
    if dim == 2:
        pts = pts.detach().numpy()  #para não salvar o gradiente
        # 2D scatter
        plt.plot(pts[:, 0], pts[:, 1], label=lbl)
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D points')
        ax = plt.gca()
        ax.set_xlim(vMin, vMax)
        ax.set_ylim(vMin, vMax)
##########################################################




###################
#Funções da solução "Exata" (Euler) 
def EulerStep(xin, N=10):
    """
    Euler com subdivisão: divide o passo delta em N subpassos menores.
    """
    h = delta / N
    cur = xin.clone()
    for _ in range(N):
        cur = cur + h * RHS(cur)
    return cur - xin #return delta * RHS(xin)
################################################



#############
#Treinamento
def trainModel(modelnn, func, nBlocks, blockSize):
    for i in range(nBlocks):
        x = RandomPoint(blockSize)
        y = func(x)
        modelnn.TrainingStep(x, y)

def DrawTrajectories():
    x1 = RandomPoint()
    x2 = x1.clone()
    x3= x1.clone()

    ###############
    ##Referência
    print('Ponto Inicial:',x1)
    trajEuler=Trajectory(x1,100,EulerStep)
    plotPoints(trajEuler, 'Euler')
    ###############

    ###################
    ##  DeepNNEuler
    DeepNNEuler = DeepNN.DeepNN(in_features = dim, hidden_layer = 3, hidden_size=80, out_features=dim)
    DeepNN.optimizer = optim.Adam(DeepNNEuler.parameters(), lr=0.01)
    DeepNN.loss_fn = nn.MSELoss()
    trainModel(DeepNNEuler, EulerStep, nBlocks,blockSize ) #nBlocks,Blocksize #con 10000,100 treina quase perfeito, mas demora
    trajDeepNNEuler =Trajectory(x2, 100, DeepNNEuler.forward)
    plotPoints(trajDeepNNEuler, 'DeepNNEuler')
    ################################
    

    #################################
    ### ResNet
    ResNNEuler = ResNN.ResNN(in_features = dim, hidden_layer = 3, hidden_size=80, out_features=dim)
    ResNN.optimizer = optim.Adam(ResNNEuler.parameters(), lr=0.01)
    ResNN.loss_fn = nn.MSELoss()
    trainModel(ResNNEuler, EulerStep, nBlocks, blockSize)
    trajResNNEuler =Trajectory(x2, 100, ResNNEuler.forward)
    plotPoints(trajResNNEuler, 'ResNNEuler')
    #################################


    plt.legend()
    plt.show()

#Main
DrawTrajectories()
print('ok')