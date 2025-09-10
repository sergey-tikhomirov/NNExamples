import torch
import torch.nn as nn

class HeunStep(nn.Module):
    """
    Minimal Heun (RK2) block that returns the increment:
        res = 0.5 * ( f(x) + f(x + f(x)) )

    Train with fit_pairs(x, y) where y has same shape as forward(x)
    (e.g., y = x_next - x if you're learning increments).
    """

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k1 = self.f(x)
        x_tilde = x + k1
        k2 = self.f(x_tilde)
        res = 0.5 * (k1 + k2)
        return res

    def TrainingStep(self, x: torch.Tensor, y: torch.Tensor, lr: float = 0.1) -> float:
        """
        Single full-batch MSE update so that forward(x) ≈ y.
        Returns the loss value after this update.
        """
        assert x.shape == y.shape, f"Shapes must match: got {x.shape} vs {y.shape}"
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)

        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        opt.zero_grad(set_to_none=True)
        pred = self(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        return loss.item()

