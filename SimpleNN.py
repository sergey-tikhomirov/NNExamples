import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the network
class SimpleNN(nn.Module):
    def __init__(self, in_features = 2, hidden_size=8, out_features=1):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(in_features, hidden_size)   # input: 2 → hidden layer
        #self.activation = nn.ReLU()              # nonlinearity
        self.activation = nn.Sigmoid()              # nonlinearity
        self.output = nn.Linear(hidden_size, out_features)  # hidden → output: 1 real number

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x


def TrainingStep(x, y):
    global y_pred
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

# Training Logic functions

def funcAnd(x1, x2):
    return x1*x2


def funcOr(x1, x2):
    return 1-(1-x1)*(1-x2)


def funcXOr(x1, x2):
    return 1- (x1+x2-1)*(x1+x2-1)


def GenerateTrainingData():
    x1 = np.random.randint(0, 2)
    x2 = np.random.randint(0, 2)
    x = torch.tensor([[x1, x2]], dtype=torch.float32)  # input: 2 real numbers
    y = torch.tensor([[funcXOr(x1, x2)]], dtype=torch.float32)  # target: 1 real number
    TrainingStep(x, y)


def DoTraining(n):
    for i in range(n):
        GenerateTrainingData()
    allInput = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    print("Iteration", n)
    print("Prediction:", model(allInput).detach().numpy().flatten())
    for name, param in model.named_parameters():
        print(name, param.data)
    print("--------")

# Example usage
model = SimpleNN(hidden_size=8)

# Example optimizer and loss
#optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
loss_fn = nn.MSELoss()

#Parametres to vary
#AND/OR/XOR
#ReLU/sigmoid
#hidden_size
#lr 0.01-0.1


DoTraining(1)
DoTraining(10)
DoTraining(100)
DoTraining(1000)
DoTraining(10000)
#DoTraining(100000)

#Stable convergence
# AND, Sigmoid, hidden_size = 1, lr = 0.1
# AND, Sigmoid, hidden_size = 2, lr = 0.1
# AND, Sigmoid, hidden_size = 3, lr = 0.1
# AND, ReLU, hidden_size = 8, lr = 0.01
# XOR, Sigmoid, hidden_size = 2, lr = 0.1
# XOR, Sigmoid, hidden_size = 3, lr = 0.1
# XOR, Sigmoid, hidden_size = 3, lr = 0.05
# XOR, Sigmoid, hidden_size = 8, lr = 0.05

#Not Stable Convergence
# AND, ReLU, hidden_size = 1, lr = 0.1
# AND, ReLU, hidden_size = 1, lr = 0.01
# AND, ReLU, hidden_size = 2, lr = 0.01
# AND, ReLU, hidden_size = 3, lr = 0.01
# XOR, hidden_size = 1
# XOR, Sigmoid, hidden_size = 2, lr = 0.01
# XOR, Sigmoid, hidden_size = 8, lr = 0.01 Need Length 100000