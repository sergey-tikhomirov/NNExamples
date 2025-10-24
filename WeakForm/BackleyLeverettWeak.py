import torch
import matplotlib.pyplot as plt
from torch.autograd import grad

dev = torch . device (" cuda :0" if torch . cuda . is_available () else "cpu")

hx = 0.004
ht = 0.004
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

def weak_residual_loss(NN, x, t):
    term1 = -NN(torch.hstack((t, x)))*dphi_dt_ref(torch.zeros_like(x), torch.zeros_like(t))
    term2 = flux(x)*dphi_dx_ref(torch.zeros_like(x), torch.zeros_like(t))
    return (-(term1 + term2)*16*hx*ht).square().mean()

NWeak = torch .nn. Sequential (
    torch .nn. Linear (2, 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 50) , torch .nn. SiLU () ,
    torch .nn. Linear (50 , 1) ,
).to(dev)

optimizerWeak = torch.optim.Adam(NWeak.parameters(), lr =3e-4)
J = 256 # the batch size
nBatches = 100

def TrainNWeak():
    for i in range(500):
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

        loss = initial_loss + residual_loss + bcleft_loss + bcright_loss

        loss.backward()
        optimizerWeak.step()

def plotWeak():
    x = torch.linspace(xMin, xMax, 1001).to(dev)
    t = torch.full((N+1, ), T)



TrainNWeak()
