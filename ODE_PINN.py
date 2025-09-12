import torch
import torch.nn as nn

class SimpleODEPINN(nn.Module):
    """
    Minimal NN(x0, t) with SOFT initial-condition penalty.
    Constructor only requires `dim` (state dimension).
    """
    def __init__(self, dim: int, fnn: nn.Module):
        super().__init__()
        self.dim = dim
        self.net = fnn
            # nn.Sequential(
            #nn.Linear(dim + 1, 64),
            #nn.Tanh(),
            #nn.Linear(64, dim),)


    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x0: (B, dim), t: (B, 1)
        z = torch.cat([x0, t], dim=1)
        return self.net(z)

    @staticmethod
    def _dt(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # dy/dt for batched y wrt per-sample scalar t
        grads = []
        for k in range(y.shape[1]):
            gk = torch.autograd.grad(
                outputs=y[:, k].sum(), inputs=t,
                create_graph=True, retain_graph=True
            )[0]  # (B, 1)
            grads.append(gk)
        return torch.cat(grads, dim=1)  # (B, dim)

    def loss(self,
             x0: torch.Tensor,
             t: torch.Tensor,
             f,
             lambda_ic: float = 1.0):
        """
        total = mean(|| d/dt NN(x0,t) - f(NN(x0,t)) ||^2) + lambda_ic * mean(|| NN(x0,0) - x0 ||^2)
        f: callable (B, dim) -> (B, dim)
        """
        t = t.clone().requires_grad_(True)
        x_hat = self.forward(x0, t)                 # (B, dim)
        dxt   = self._dt(x_hat, t)                  # (B, dim)
        resid = dxt - f(x_hat)
        residual_loss = (resid ** 2).mean()

        t0 = torch.zeros_like(t)
        ic_loss = ((self.forward(x0, t0) - x0) ** 2).mean()

        total = residual_loss + lambda_ic * ic_loss
        return total, {"residual_loss": residual_loss.detach(),
                       "ic_loss": ic_loss.detach(),
                       "total_loss": total.detach()}

    def Train(self, x0: torch.Tensor, t: torch.Tensor):
        lr = 1e-3
        lambda_ic = 1.0

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train(True)
        history = []
        opt.zero_grad(set_to_none=True)
        L = self.loss(x0, t, lambda_ic=lambda_ic)
        L.backward()
        opt.step()
        history.append(float(L.detach()))
        return history
