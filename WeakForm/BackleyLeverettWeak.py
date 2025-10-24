import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad

dev = torch . device (" cuda :0" if torch . cuda . is_available () else "cpu")

hx = 0.004
ht = 0.004
GL = 2
xMin = -1.0
xMax = 1.0
T = 2.0

leftVal = 1.0
rightVal = 0.0


def flux(u: torch.Tensor) -> torch.Tensor:
    # Buckley–Leverett with Corey exponents 2, equal viscosities
    u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
    return (u**2) / (u**2 + (1 - u)**2)

# φ(ξ,τ)=(1-ξ^2)(1-τ^2) on [-1,1]^2; we need ∂x φ and ∂t φ via chain rule
# ξ=2(x-xc)/hx, τ=2(t-tc)/ht  ⇒  ∂x φ = -4 ξ (1-τ^2)/hx,  ∂t φ = -4 τ (1-ξ^2)/ht
def dphi_dx_ref(xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return -4.0 * xi * (1.0 - tau**2) / hx

def dphi_dt_ref(xi: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return -4.0 * tau * (1.0 - xi**2) / ht


def initData(x):
    return torch.zeros_like(x)



def leggauss_2d(n, ax, bx, ay, by):
    # 1-D nodes/weights on [-1,1]
    x1, w1 = np.polynomial.legendre.leggauss(n)
    x2, w2 = np.polynomial.legendre.leggauss(n)

    # map to [ax,bx] and [ay,by]
    X = 0.5*(bx-ax)*x1 + 0.5*(bx+ax)
    Y = 0.5*(by-ay)*x2 + 0.5*(by+ay)

    # tensor-product grid and weights
    XX, YY = np.meshgrid(X, Y, indexing='ij')                 # shape (n, n)
    W = (0.25*(bx-ax)*(by-ay)) * np.outer(w1, w2)             # shape (n, n)
    return XX, YY, W

def weak_residual_loss_singleVal(NN, t, x, dt, dx):
    evalt = t + float(dt*ht)
    evalx = x + float(dt*hx)
    s = NN(torch.hstack((evalt, evalx)))
    term1 = s*dphi_dt_ref(torch.full_like(x, dx), torch.full_like(t, dt))
    term2 = flux(s)*dphi_dx_ref(torch.full_like(x, dx), torch.full_like(t, dt))
    return (-(term1 + term2)*16*hx*ht).square().mean()

def weak_residual_loss(NN, x, t):
    tt, xx, ww = leggauss_2d(GL, -1, 1, -1, 1)
    #t = torch.from_numpy(tt)
    #x = torch.from_numpy(xx)
    #w = torch.from_numpy(ww)
    res = 0
    for itt in range(GL):
        for ixx in range(GL):
            dt = tt[itt, ixx]
            dx = xx[itt, ixx]
            res+= weak_residual_loss_singleVal(NN, t, x, dt, dx)

    return res


NWeak = torch .nn. Sequential (
    torch .nn. Linear (2, 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 1) ,
).to(dev)

optimizerWeak = torch.optim.Adam(NWeak.parameters(), lr =3e-4)
J = 256 # the batch size
nBatches = 5000

def TrainNWeak():
    for i in range(nBatches):
        x = (xMin + (xMax-xMin)*torch.rand(J, 1)).to(dev)
        t = torch.rand(J, 1).to(dev) * T

        x.requires_grad_()
        t.requires_grad_()

        optimizerWeak.zero_grad()

        s0 = NWeak(torch.hstack((torch.zeros_like(t), x)))
        initial_loss = (s0 - initData(x)).square().mean()

        sl = NWeak(torch.hstack((t, torch.full_like(x, xMin))))
        bcleft_loss = (sl - torch.full_like(x, leftVal)).square().mean()

        sr = NWeak(torch.hstack((t, torch.full_like(x, xMax))))
        bcright_loss = (sr - torch.full_like(x, rightVal)).square().mean()

        residual_loss = weak_residual_loss(NWeak, x, t)

        loss = initial_loss + 10* residual_loss + bcleft_loss + bcright_loss

        #print(initial_loss, residual_loss, bcleft_loss, bcright_loss)

        loss.backward()
        optimizerWeak.step()

def plotWeak():
    N = 1000
    x = torch.linspace(xMin, xMax, N+1).unsqueeze(1)
    t = torch.full((N+1, ), T).unsqueeze(1)
    print(torch.hstack((t, x)))
    res = NWeak(torch.hstack((t, x)))

    #fig, ax = plt.subplots()
    plt.plot(x.detach().numpy(), res.detach().numpy())
    plt.ylim(0, 1)
    #ax.set_ylim(0, 1)
    plt.show()

TrainNWeak()
plotWeak()
