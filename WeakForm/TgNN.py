"""
Minimal PyTorch weak-form PINN for 1D Buckley–Leverett
- Local test functions + Gauss–Legendre quadrature on spacetime patches
- Weak residual: ∫∫ [ u * ∂_t φ + f(u) * ∂_x φ ] dx dt ≈ 0
- Soft IC/BC (u(x,0)=0, u(0,t)=1, relaxed outflow at x=1)

Run (example):
    python bl_weakform_min.py --steps 10000 --width 64 --depth 4 --qx 4 --qt 4 --patches 128
"""

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# ---------------------- Config ----------------------
@dataclass
class Config:
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.5
    width: int = 48
    depth: int = 3
    lr: float = 2e-3
    steps: int = 4000
    qx: int = 3
    qt: int = 3
    n_patches: int = 64
    hx: float = 0.25          # patch size in x
    ht: float = 0.12          # patch size in t
    n_bc: int = 256
    out_npz: str = "bl_weakform_solution.npz"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Flux ----------------------
def flux(u: torch.Tensor) -> torch.Tensor:
    # Buckley–Leverett with Corey exponents 2, equal viscosities
    u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
    return (u**2) / (u**2 + (1 - u)**2)

# ---------------------- Model ----------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, width=64, depth=4, x_bounds=(0.,1.), t_bounds=(0.,0.5)):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.x_min, self.x_max = x_bounds
        self.t_min, self.t_max = t_bounds

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # scale to [-1,1] for stability
        xs = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        ts = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1
        z = torch.stack([xs, ts], dim=-1)
        return torch.sigmoid(self.net(z))  # saturation in [0,1]

# ---------------------- Quadrature & test functions ----------------------
def gauss_legendre(n: int, device: torch.device):
    nodes, weights = np.polynomial.legendre.leggauss(n)   # on [-1,1]
    return (
        torch.tensor(nodes, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )

# φ(ξ,τ)=(1-ξ^2)(1-τ^2) on [-1,1]^2; we need ∂x φ and ∂t φ via chain rule
# ξ=2(x-xc)/hx, τ=2(t-tc)/ht  ⇒  ∂x φ = -4 ξ (1-τ^2)/hx,  ∂t φ = -4 τ (1-ξ^2)/ht
def dphi_dx_ref(xi: torch.Tensor, tau: torch.Tensor, hx: float) -> torch.Tensor:
    return -4.0 * xi * (1.0 - tau**2) / hx

def dphi_dt_ref(xi: torch.Tensor, tau: torch.Tensor, ht: float) -> torch.Tensor:
    return -4.0 * tau * (1.0 - xi**2) / ht

def sample_patches(n_patches: int, hx: float, ht: float, device: torch.device, x_min, x_max, t_min, t_max):
    margin_x, margin_t = hx/2, ht/2
    xc = torch.rand(n_patches, device=device) * (x_max - x_min - 2*margin_x) + (x_min + margin_x)
    tc = torch.rand(n_patches, device=device) * (t_max - t_min - 2*margin_t) + (t_min + margin_t)
    return xc, tc

# ---------------------- Losses ----------------------
def weak_residual_loss(model, cfg):
    xi_nodes, xi_w = gauss_legendre(cfg.qx, cfg.device)
    tau_nodes, tau_w = gauss_legendre(cfg.qt, cfg.device)
    XI, TAU = torch.meshgrid(xi_nodes, tau_nodes, indexing='ij')
    WX, WT = torch.meshgrid(xi_w, tau_w, indexing='ij')
    W2 = (WX * WT).reshape(-1)             # tensor-product weights
    XI = XI.reshape(-1); TAU = TAU.reshape(-1)

    xc, tc = sample_patches(cfg.n_patches, cfg.hx, cfg.ht, cfg.device,
                            cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max)
    # map reference nodes to each patch
    x = xc[:, None] + (cfg.hx/2.0) * XI[None, :]
    t = tc[:, None] + (cfg.ht/2.0) * TAU[None, :]

    u = model(x, t)
    f = flux(u)

    dphix = dphi_dx_ref(XI[None, :], TAU[None, :], cfg.hx)
    dphit = dphi_dt_ref(XI[None, :], TAU[None, :], cfg.ht)

    g = u * dphit + f * dphix                        # integrand
    J = (cfg.hx/2.0) * (cfg.ht/2.0)                  # Jacobian of the map
    patch_int = (g * W2).sum(dim=1) * J              # ∫∫ g φ' dx dt per patch
    return (patch_int**2).mean()                     # mean-squared residual over patches

def boundary_losses(model, cfg):
    # Initial condition: u(x,0)=0
    x0 = torch.rand(cfg.n_bc, device=cfg.device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t0 = torch.zeros_like(x0) + cfg.t_min
    L_ic = (model(x0[:, None], t0[:, None]).squeeze(-1)**2).mean()

    # Injection boundary: u(0,t)=1
    tL = torch.rand(cfg.n_bc, device=cfg.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    xL = torch.zeros_like(tL) + cfg.x_min
    L_L = ((model(xL[:, None], tL[:, None]).squeeze(-1) - 1.0)**2).mean()

    # Outflow boundary (relaxed Neumann): u(1,t) ≈ u(1-ε,t)
    eps = 1e-2
    tR = torch.rand(cfg.n_bc, device=cfg.device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    xR1 = torch.zeros_like(tR) + (cfg.x_max - eps)
    xR2 = torch.zeros_like(tR) + cfg.x_max
    u_R1 = model(xR1[:, None], tR[:, None]).squeeze(-1)
    u_R2 = model(xR2[:, None], tR[:, None]).squeeze(-1)
    L_R = ((u_R2 - u_R1)**2).mean()
    return L_ic, L_L, L_R

# ---------------------- Train & evaluate ----------------------
def train(cfg: Config):
    model = MLP(width=cfg.width, depth=cfg.depth,
                x_bounds=(cfg.x_min, cfg.x_max),
                t_bounds=(cfg.t_min, cfg.t_max)).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        opt.zero_grad()
        L_wf = weak_residual_loss(model, cfg)
        L_ic, L_L, L_R = boundary_losses(model, cfg)
        loss = L_wf + 1.0*L_ic + 1.0*L_L + 0.2*L_R
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"[{step:6d}] total={loss.item():.4e}  wf={L_wf.item():.4e}  "
                  f"ic={L_ic.item():.4e}  L={L_L.item():.4e}  R={L_R.item():.4e}")

    return model

def evaluate_grid(model: nn.Module, cfg: Config, NX=201, NT=101):
    xs = torch.linspace(cfg.x_min, cfg.x_max, NX, device=cfg.device)
    ts = torch.linspace(cfg.t_min, cfg.t_max, NT, device=cfg.device)
    X, T = torch.meshgrid(xs, ts, indexing='ij')
    with torch.no_grad():
        U = model(X.reshape(-1,1), T.reshape(-1,1)).reshape(NX, NT).cpu().numpy()
    return X.cpu().numpy(), T.cpu().numpy(), U

def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=cfg.steps)
    p.add_argument("--width", type=int, default=cfg.width)
    p.add_argument("--depth", type=int, default=cfg.depth)
    p.add_argument("--qx", type=int, default=cfg.qx)
    p.add_argument("--qt", type=int, default=cfg.qt)
    p.add_argument("--patches", type=int, default=cfg.n_patches)
    p.add_argument("--hx", type=float, default=cfg.hx)
    p.add_argument("--ht", type=float, default=cfg.ht)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--out", type=str, default=cfg.out_npz)
    args = p.parse_args()

    # override from CLI
    cfg.steps    = args.steps
    cfg.width    = args.width
    cfg.depth    = args.depth
    cfg.qx       = args.qx
    cfg.qt       = args.qt
    cfg.n_patches= args.patches
    cfg.hx       = args.hx
    cfg.ht       = args.ht
    cfg.lr       = args.lr
    cfg.out_npz  = args.out

    model = train(cfg)
    X, T, U = evaluate_grid(model, cfg, NX=161, NT=81)
    np.savez(cfg.out_npz, X=X, T=T, U=U)
    print(f"Saved: {cfg.out_npz}")

if __name__ == "__main__":
    main()
