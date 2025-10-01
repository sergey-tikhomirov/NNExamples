import torch
import matplotlib . pyplot as plt
from torch . autograd import grad
from matplotlib . gridspec import GridSpec
from matplotlib .cm import ScalarMappable

#from RunODE_PINN import nSteps

dev = torch . device (" cuda :0" if torch . cuda . is_available () else "cpu")

#torch.manual_seed (0)

print("ODE_PINN_Book")

dim = 2

vMin = -7
vMax = 7
gamma = 0.4
omega2 = 8.91
delta = 0.1
nSteps = 50

def dampedPendulum(xin):
    x1, x2 = xin.unbind(dim=-1)
    dx1 = x2
    dx2 = -gamma*x2 - omega2 * torch.sin(x1)
    return torch.stack((dx1, dx2), dim=-1)


T = delta*nSteps # the time horizom
M = 20000 # the number of training samples



x_data = vMin + (vMax - vMin) * torch.randn(M, 2).to(dev ) * 2
t_data = torch.rand(M, 1).to(dev) * T

# The initial value
def phi(x):
    return x
    #return x.square().sum( axis =1, keepdims = True ).sin()

# We use a network with 4 hidden layers of 50 neurons each and the
# Swish activation function ( called SiLU in PyTorch )
N = torch .nn. Sequential (
    torch .nn. Linear (3, 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 2) ,
).to(dev)

optimizer = torch . optim . Adam (N. parameters () , lr =3e-4)

J = 256 # the batch size

for i in range (20000) :
# Choose a random batch of training samples
    indices = torch . randint (0, M, (J ,))
    x = x_data [ indices , :]
    t = t_data [ indices , :]

    x1 , x2 = x[:, 0:1] , x[:, 1:2]

    x1.requires_grad_()
    x2.requires_grad_()
    t.requires_grad_()

    optimizer.zero_grad ()

    # Denoting by u the realization function of the ANN , compute
    # u(0, x) for each x in the batch
    u0 = N(torch.hstack((torch.zeros_like(t), x)))
    # Compute the loss for the initial condition
    initial_loss = (u0 - phi(x)).square().mean()

    # Compute the partial derivatives using automatic
    # differentiation
    u = N(torch.hstack((t, x1 , x2)))
    ones = torch.ones_like (u)
    u_t = grad (u, t, ones , create_graph = True ) [0]
    #u_x1 = grad (u, x1 , ones , create_graph = True )[0]
    #u_x2 = grad (u, x2 , ones , create_graph = True )[0]
    #ones = torch . ones_like ( u_x1 )
    #u_x1x1 = grad (u_x1 , x1 , ones , create_graph = True )[0]
    #u_x2x2 = grad (u_x2 , x2 , ones , create_graph = True )[0]
    rhs = dampedPendulum(u)

    # Compute the loss for the PDE
    #Laplace = u_x1x1 + u_x2x2
    #pde_loss = ( u_t - (0.005 * Laplace + u - u **3) ). square (). mean ()
    ode_loss = (u_t - rhs).square().mean()

    # Compute the total loss and perform a gradient step
    loss = initial_loss + ode_loss
    loss.backward ()
    optimizer.step ()


def TrajectoryODEPINN(xin, nSteps, model):
    #print(xin)
    res = xin.clone()
    for i in range(nSteps):
        print(xin)
        z = torch.cat([torch.tensor([[i*delta]]), xin], dim=1)
        print(z)
        xcurr = model.forward(z)
        res = torch.cat([res, xcurr], dim = 0)
    print(res)
    return res.detach().numpy()

def RandomPoint(blockSize = 1):
    return vMin + (vMax - vMin) * torch.rand(blockSize, dim)

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

x3 = RandomPoint()
trajODEPINN = TrajectoryODEPINN(x3, nSteps, N)
plotPoints(trajODEPINN, 'ODEPINN')

plt.show()
