import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

class SimpleODEPINN(nn.Module):
    """
    Minimal NN(x0, t) with SOFT initial-condition penalty.
    Constructor only requires `dim` (state dimension).
    """
    def __init__(self, dim: int, fnn: nn.Module, rhs, isStrong: bool = False, icLambda = 1.0):
        super().__init__()
        self.dim = dim
        self.net = fnn
        self.rhs = rhs
        self.isStrong = isStrong
        self.icLambda = icLambda
            # nn.Sequential(
            #nn.Linear(dim + 1, 64),
            #nn.Tanh(),
            #nn.Linear(64, dim),)


    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x0: (B, dim), t: (B, 1)
        z = torch.cat([x0, t], dim=1)
        if self.isStrong:
            return x0 + t * self.net(z)
        else:
            return self.net(z)

#    @staticmethod
#    def _dt(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#        # dy/dt for batched y wrt per-sample scalar t
#        grads = []
#        for k in range(y.shape[1]):
#            gk = torch.autograd.grad(
#                outputs=y[:, k].sum(), inputs=t,
#                create_graph=True, retain_graph=True
#            )[0]  # (B, 1)
#            grads.append(gk)
#        return torch.cat(grads, dim=1)  # (B, dim)

    def _dt(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # y: (B, dim), t: (B, 1) with requires_grad=True
        ones = torch.ones_like(y)
        dt = torch.autograd.grad(y, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]  # (B,1)
        return dt.expand(-1, y.shape[1])  # (B, dim) if you want a per-dim copy

    def loss(self,
             x0: torch.Tensor,
             t: torch.Tensor,
             f):
        """
        total = mean(|| d/dt NN(x0,t) - f(NN(x0,t)) ||^2) + lambda_ic * mean(|| NN(x0,0) - x0 ||^2)
        f: callable (B, dim) -> (B, dim)
        """
        t = t.clone().requires_grad_(True)
        x_hat = self.forward(x0, t)                 # (B, dim)
        dxt   = self._dt(x_hat, t)                  # (B, dim)
        resid = dxt - f(x_hat)
        residual_loss = (resid ** 2).mean()
        #print(dxt, f(x_hat), residual_loss)

        t0 = torch.zeros_like(t)
        ic_loss = ((self.forward(x0, t0) - x0) ** 2).mean()
        total = residual_loss
        if not self.isStrong:
            total += self.icLambda * ic_loss

        #vec = parameters_to_vector(self.parameters())
        #reg_loss = vec.pow(2).sum()
        #total += reg_loss

        return total
        #,{"residual_loss": residual_loss.detach(), "ic_loss": ic_loss.detach(), "total_loss": total.detach()}

    def Train(self, x0: torch.Tensor, t: torch.Tensor):
        lr = 0.01

        #opt = torch.optim.Adam(self.parameters(), lr=lr)

        decay, no_decay = [], []
        for n, p in self.named_parameters():
            #print(n, p)
            if not p.requires_grad: continue
            (no_decay if (p.ndim == 1 or n.endswith("bias")) else decay).append(p)

        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-4},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=1e-3
        )

        self.train(True)
        history = []
        opt.zero_grad(set_to_none=True)
        L = self.loss(x0, t, self.rhs)
        L.backward()
        opt.step()
        history.append(float(L.detach()))
        return history
