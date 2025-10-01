# FitzHugh–Nagumo (Example 4.8) — ResNet ODE one-step learner
# Requirements: python>=3.9, pytorch, numpy, matplotlib

import math, random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt

# -------------------
# Problem definition
# -------------------
@dataclass
class FHNConfig:
    k: float = 0.5
    dt: float = 0.05
    domain_min: float = -5.0
    domain_max: float =  5.0

CFG = FHNConfig()

def fhn_rhs(x, k=CFG.k):
    """
    x: (..., 2) tensor -> dx/dt: (..., 2)
    dx1/dt = 3 * (x1 + x2 - (1/3) x1^3 - k)
    dx2/dt = - (1/3) * (x1 + 0.8 x2 - 0.7)
    """
    x1, x2 = x[..., 0], x[..., 1]
    dx1 = 3.0 * (x1 + x2 - (x1**3)/3.0 - k)
    dx2 = - (x1 + 0.8*x2 - 0.7) / 3.0
    return torch.stack([dx1, dx2], dim=-1)

import numpy.linalg as LA

def fhn_equilibria(k=CFG.k):
    """
    Solve equilibrium conditions analytically reduced to a cubic in x1:
      4 x1^3 + 3 x1 + 12k - 10.5 = 0,
    then x2 = 0.875 - 1.25 x1 (from x1 + 0.8 x2 = 0.7).
    Returns a list of (x1, x2) for real roots.
    """
    coeffs = [4.0, 0.0, 3.0, 12.0*k - 10.5]
    roots = np.roots(coeffs)
    reals = roots[np.isreal(roots)].real
    points = [(float(x1), float(0.875 - 1.25*x1)) for x1 in reals]
    return points

def fhn_jacobian_at(x):
    """
    Jacobian of RHS at x=(x1,x2):
      J = [[ 3*(1 - x1^2),    3.0     ],
           [    -1/3      ,  -0.8/3  ]]
    """
    x1 = float(x[0])
    return np.array([
        [3.0*(1.0 - x1**2), 3.0],
        [-1.0/3.0,         -0.8/3.0]
    ], dtype=np.float64)

def classify_linear(J):
    """
    Classify by eigenvalues: spiral/focus vs node/saddle and stability.
    """
    eig = LA.eigvals(J)
    tr = np.trace(J)
    det = LA.det(J)
    disc = tr*tr - 4*det

    kind = None
    if det < 0:
        kind = "saddle"
    else:
        if np.iscomplex(eig[0]) or np.iscomplex(eig[1]) or disc < 0:
            kind = "spiral"
        else:
            kind = "node"

    if kind == "saddle":
        stability = "unstable"
    else:
        if tr < 0:
            stability = "stable"
        elif tr > 0:
            stability = "unstable"
        else:
            stability = "center/neutral"

    return kind, stability, eig, tr, det, disc


# -------------------
# One-step targets
# -------------------
def euler_step(x, dt, rhs):
    return x + dt * rhs(x)

def rk2_step(x, dt, rhs):
    k1 = rhs(x)
    k2 = rhs(x + dt * k1)
    return x + dt * 0.5 * (k1 + k2)

def rk4_step(x, dt, rhs):
    k1 = rhs(x)
    k2 = rhs(x + 0.5*dt*k1)
    k3 = rhs(x + 0.5*dt*k2)
    k4 = rhs(x + dt*k3)
    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

STEP_METHODS = {
    "euler": euler_step,
    "rk2": rk2_step,
    "rk4": rk4_step,
}

# -------------------
# Data generation
# -------------------
def sample_initials(n, lo=CFG.domain_min, hi=CFG.domain_max, device="cpu"):
    arr = np.random.uniform(lo, hi, size=(n,2)).astype(np.float32)
    return torch.tensor(arr, device=device)

def make_pairs(n_pairs, stepper="rk4", dt=CFG.dt, device="cpu"):
    """
    Returns (x0, x1) tensors of shape (n_pairs, 2),
    where x1 is one-step target from chosen method.
    """
    x0 = sample_initials(n_pairs, device=device)
    rhs = lambda z: fhn_rhs(z)
    x1 = STEP_METHODS[stepper](x0, dt, rhs)
    return x0, x1

# -------------------
# ResNet model (predicts increment, then adds skip connection)
# -------------------
class IncrementMLP(nn.Module):
    def __init__(self, dim=2, hidden=128, depth=4, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), activation()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]  # output increment
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        inc = self.net(x)
        return inc

class OneStepResNet(nn.Module):
    def __init__(self, inc_net: nn.Module):
        super().__init__()
        self.f = inc_net

    def forward(self, x):
        # pout = x + N(x)  (ResNet-style skip)
        return x + self.f(x)

# -------------------
# Training utilities
# -------------------
def train(model, x0, x1, *, epochs=3000, lr=1e-3, batch=256, verbose=True):
    device = next(model.parameters()).device
    x0, x1 = x0.to(device), x1.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = x0.shape[0]
    for ep in range(1, epochs+1):
        # mini-batch
        idx = torch.randint(0, n, (batch,), device=device)
        xb, yb = x0[idx], x1[idx]
        yhat = model(xb)
        loss = loss_fn(yhat, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (ep % max(1, epochs//10) == 0):
            with torch.no_grad():
                mse = loss_fn(model(x0), x1).item()
            print(f"epoch {ep:5d}  train MSE={mse: .6e}")
    return model

@torch.no_grad()
def evaluate(model, x0_test, x1_test):
    loss_fn = nn.MSELoss()
    device = next(model.parameters()).device
    x0_test, x1_test = x0_test.to(device), x1_test.to(device)
    mse = loss_fn(model(x0_test), x1_test).item()
    linf = (model(x0_test) - x1_test).abs().max().item()
    return {"mse": mse, "linf": linf}

eqs = fhn_equilibria(k=CFG.k)
print("\nEquilibria (real):", eqs)

for p in eqs:
    J = fhn_jacobian_at(p)
    kind, stab, eig, tr, det, disc = classify_linear(J)
    print(f"Equilibrium {p}: {kind}, {stab}")
    print(f"  eigenvalues = {eig}")
    print(f"  trace = {tr:.6f}, det = {det:.6f}, disc = {disc:.6f}")

# -------------------
# Rollout (simulate trajectory by repeated one-step NN)
# -------------------
@torch.no_grad()
def rollout(model, x0, n_steps, dt=CFG.dt):
    # Ensure x is on the same device as the model
    dev = next(model.parameters()).device
    x = x0.to(dev).clone()
    xs = [x.clone()]
    for _ in range(n_steps):
        x = model(x)
        xs.append(x.clone())
    return torch.stack(xs, dim=0)  # (n_steps+1, 2)

# -------------------
# Plot: vector field once + overlay rollouts for all models
# -------------------
def plot_all_phase_portraits(resnets, n_grid=20, x0=(0.8, -0.1), T=10.0):
    """
    Plot vector field and overlay both reference integrator rollouts
    and NN rollouts up to total time T from initial x0.
    """
    # Vector field grid on CPU (numpy-friendly for quiver)
    xs = np.linspace(CFG.domain_min, CFG.domain_max, n_grid)
    ys = np.linspace(CFG.domain_min, CFG.domain_max, n_grid)
    X, Y = np.meshgrid(xs, ys)
    pts_cpu = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)

    with torch.no_grad():
        rhs_vals = fhn_rhs(pts_cpu.reshape(-1, 2)).cpu().numpy()
    U = rhs_vals[:, 0].reshape(n_grid, n_grid)
    V = rhs_vals[:, 1].reshape(n_grid, n_grid)

    # Initial condition and rollout length
    x0_cpu = torch.tensor(x0, dtype=torch.float32)
    rhs = lambda z: fhn_rhs(z)
    n_steps = int(T / CFG.dt)

    plt.figure(figsize=(7, 7))
    plt.quiver(X, Y, U, V, color="lightgray", angles="xy")

    # Consistent colors for each method (reference + NN share a color, solid vs dashed)
    colors = {"euler": "tab:green", "rk2": "tab:orange", "rk4": "tab:red"}

    for stepper, model in resnets.items():
        # NN rollout (device-safe)
        traj_nn = rollout(model, x0_cpu, n_steps).cpu().numpy()

        # Reference rollout on CPU
        x = x0_cpu.clone()
        traj_ref = [x.numpy()]
        for _ in range(n_steps):
            x = STEP_METHODS[stepper](x, CFG.dt, rhs)
            traj_ref.append(x.numpy())
        traj_ref = np.stack(traj_ref, axis=0)

        c = colors.get(stepper, None)
        plt.plot(traj_ref[:, 0], traj_ref[:, 1], "-",  label=f"{stepper.upper()} ref", color=c)
        plt.plot(traj_nn[:, 0],  traj_nn[:, 1],  "--", label=f"{stepper.upper()} NN",  color=c)

        # --- NEW: overlay equilibria with labels ---
    eqs = fhn_equilibria(k=CFG.k)
    for p in eqs:
        J = fhn_jacobian_at(p)
        kind, stab, eig, tr, det, disc = classify_linear(J)

        # plot the point
        plt.scatter([p[0]], [p[1]], s=50, marker="X", label=f"eq: {kind}, {stab}")

        # small text annotation with type
        plt.text(p[0], p[1], f"  {kind}, {stab}", fontsize=9)

    # start marker as you already had

    plt.scatter([x0[0]], [x0[1]], marker="o", label="start")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(f"FitzHugh–Nagumo: Phase Portraits up to T={T}")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# -------------------
# Main: train three models on different targets (Euler/RK2/RK4)
# -------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    n_train = 2000
    n_test  = 2000

    # choose architecture per Example 4.8: 2 layers × 64 neurons
    inc_net = lambda: IncrementMLP(dim=2, hidden=64, depth=2, activation=nn.ReLU)
    resnets = {}
    metrics  = {}

    for stepper in ["euler", "rk2", "rk4"]:
        print(f"\n=== Training ResNet on {stepper.upper()} targets ===")
        x0_tr, x1_tr = make_pairs(n_train, stepper=stepper, device=device)
        x0_te, x1_te = make_pairs(n_test,  stepper=stepper, device=device)

        model = OneStepResNet(inc_net()).to(device)
        train(model, x0_tr, x1_tr, epochs=5000, lr=1e-3, batch=256, verbose=True)
        resnets[stepper] = model
        metrics[stepper] = evaluate(model, x0_te, x1_te)
        print(f"Test metrics {stepper}: {metrics[stepper]}")

    # Example rollout from initial point (-0.5, 1.0) up to a longer T
    T = 20.0  # <--- increase this for longer curves
    n_steps = int(T / CFG.dt)

    x0 = torch.tensor([0.8, -0.1], device=device, dtype=torch.float32)
    for stepper, model in resnets.items():
        traj = rollout(model, x0, n_steps)
        print(f"{stepper} rollout final state at T={T}:", traj[-1].cpu().numpy())

    # One-call summary plot for all trained models (change x0, T as desired)
    plot_all_phase_portraits(resnets, x0=(0.8, -0.1), T=T)
