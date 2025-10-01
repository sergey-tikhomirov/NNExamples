import torch
import torch.nn as nn
from torch.optim import optimizer


class DeepNN(nn.Module):
    def __init__(self, in_features=2, hidden_layer =3 , hidden_size=80, out_features=2):
        """
        Input shape:  (batch, in_features)
        Output shape: (batch, out_features)
        """
        super(DeepNN, self).__init__()
        # Hidden stack: first goes from in_features -> hidden_size,
        # then four more hidden_size -> hidden_size layers (total 5 hidden).
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(in_features, hidden_size))
        for _ in range(hidden_layer - 2):                      # now 1 + 4 = 5 hidden layers total
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        #self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x

    def loss_fn(self, output, target):
        term1 = ((target - output) ** 2).sum(dim=-1)
        return term1.mean()

    def TrainingStep(self, x, y):
        # Uses global optimizer and loss_fn, same as your SimpleNN.
        optimizer.zero_grad()
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        # (No return to keep interface identical to your SimpleNN)

    # def loss_fn_conservative(self, output, target, xin, al):
    #     term1 = ((target - output) ** 2).sum(dim=-1)
    #     norm_x_sq = (xin ** 2).sum(dim=-1)                       # (B,) or scalar
    #     norm_x_plus_ypred_sq = ((xin + output) ** 2).sum(dim=-1) # (B,) or scalar
    #     term2 = (norm_x_sq - norm_x_plus_ypred_sq) ** 2        # (B,) or scalar
    #     return (term1 + al*term2).mean()

    # def TrainingStepConservation(self, x, y, al = 0):
    #     """
    #     One optimization step for the loss:
    #         L = ||y - y_pred||^2 + (||x||^2 - ||x + y_pred||^2)^2
    #     Assumes `optimizer` is a global (like in your SimpleNN example).
    #     """
    #     self.train()
    #     optimizer.zero_grad(set_to_none=True)
    #     y_pred = self(x)
    #     """
    #     # Term 1: ||y - y_pred||^2 (sum over feature dim)
    #     term1 = ((y - y_pred) ** 2).sum(dim=-1)  # shape: (B,) or scalar
    #     # Term 2: (||x||^2 - ||x + y_pred||^2)^2
    #     norm_x_sq = (x ** 2).sum(dim=-1)                       # (B,) or scalar
    #     norm_x_plus_ypred_sq = ((x + y_pred) ** 2).sum(dim=-1) # (B,) or scalar
    #     term2 = (norm_x_sq - norm_x_plus_ypred_sq) ** 2        # (B,) or scalar
    #     # Reduce over batch (if present)
    #     loss = (term1 + al * term2).mean()
    #     """
    #     loss = self.loss_fn_conservative(y_pred, y, x, al)
    #     loss.backward()
    #     optimizer.step()
