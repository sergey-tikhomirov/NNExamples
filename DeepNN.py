import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self, in_features=2, hidden_layer = 6, hidden_size=7, out_features=2):
        """
        6-layer MLP: 5 hidden linear layers (size = hidden_size) + 1 output layer.
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

    def TrainingStep(self, x, y):
        # Uses global optimizer and loss_fn, same as your SimpleNN.
        optimizer.zero_grad()
        y_pred = self(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        # (No return to keep interface identical to your SimpleNN)
