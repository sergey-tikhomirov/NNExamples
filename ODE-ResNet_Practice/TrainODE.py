import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

from DeepNN   import DeepNN
from ResNN    import ResNN

from ODEModel import ODEModel


# Parameters

# Final time in trajectory
FINAL_TIME = 20.

# Domain of interest
## Vertical
V_MAX = 5
V_MIN = 0
## Horizontal
H_MAX = 5
H_MIN = 0

# Integration parameters
NUM_STEPS = 1
DELTA = 0.1 # aka time lag

"""A hyperparameter is a parameter that can be set in order to
define any configurable part of a model's learning process."""
# Hyperparameters
## Model
NUM_HIDDEN = 6
HIDDEN_SIZE = 40

## Learning
NUM_EPOCHS = 10000 # aka iterations
NUM_BLOCKS = 1
LR = 0.01

# My optimal
"""
## Model
NUM_HIDDEN = 6
HIDDEN_SIZE = 40

## Learning
num_epochs = 1000
num_blocks = 100
learning_rate = 0.01
"""

# Define the ODEs

def lotka_volterra(x):
    x1, x2 = x.unbind(dim=-1)

    dx1 = x1 * (1-0.2*x1-2*x2/(x1+6))
    dx2 = x2 * (-0.25 + x1/(x1+6))
    return torch.stack((dx1, dx2), dim=-1)

def lotka_volterra2(x):
    x1, x2 = x.unbind(dim=-1)

    dx1 = 1.1 * x1 - 0.4 * x1 * x2
    dx2 = -0.4 * x2 + 0.1 * x1 * x2
    return torch.stack((dx1, dx2), dim=-1)

def cubic_power(xin):
    x1, x2 = xin.unbind(dim=-1)

    dx1 = x2 -x1*(x1*x1 +x2*x2 -1)
    dx2 = -x1-x2*(x1*x1 +x2*x2 -1) 
    return torch.stack((dx1, dx2), dim=-1)  # shape (..., 2)

# Define the linear system
A = torch.tensor([[0, -1.],
                  [1., 0.]])
B = torch.tensor([0, 0])

def linear_system(x):
    return x @ A.T + B

# Trajectory calculation

def trajectory(x, num_steps, func):
    res = x
    x_curr = x.clone()
    for i in range(num_steps):
        x_curr += func(x_curr)
        res = torch.cat([res, x_curr], dim = 0)
    return res


# Integration methods

RHS = lotka_volterra

def euler_step(x, num_steps=NUM_STEPS, delta=DELTA):
    x_cur = x.clone()
    for i in range(num_steps):
        x_cur += delta/num_steps * RHS(x_cur)
    return x_cur - x


def runge_kutta2(x, delta=DELTA):
    k1 = RHS(x)
    k2 = RHS(x+delta*k1)
    return delta * (k1+k2)/2


# Generate random 2D points
def random_2d_point(size=1):
    #return V_MIN + (V_MAX - V_MIN) * torch.rand(size, 2)
    
    x1, x2 = torch.rand(size, 2).unbind(dim=-1)
    x1 = H_MIN + (H_MAX - H_MIN) * x1
    x2 = V_MIN + (V_MAX - V_MIN) * x2
    
    return torch.stack((x1, x2), dim=-1)


# Plotting routine
def plot_points(pts, lbl, clr):
    plt.plot(pts[:, 0], pts[:, 1], label=lbl, color=clr, alpha=0.8)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('2D points')
    ax = plt.gca()
    ax.set_xlim(2*H_MIN, 2*H_MAX)
    ax.set_ylim(2*V_MIN, 2*V_MAX)


# Save losses routine

def save_losses(model_name, l2, linf):
    try:
        with open("losses.txt", "a") as file:
            file.write(f"Name: {model_name}\n")
            file.write(f"# Epochs: {NUM_EPOCHS}\n")
            file.write(f"# Blocks: {NUM_BLOCKS}\n")
            file.write(f"L2 error: {l2:.10f}\n")
            file.write(f"Linf error: {linf:.10f}\n")
            file.write(f"# Layers: {NUM_HIDDEN}\n")
            file.write(f"# Neurons: {HIDDEN_SIZE}\n")
            file.write('-' * 50 + '\n')
    except Exception as e:
        print(f"Error saving losses: {e}")


### Starting point

# Initialize networks, loss, optimizers
model_list = [ODEModel(DeepNN(2, NUM_HIDDEN, HIDDEN_SIZE, 2), euler_step,
                   'Delta Euler Step DeepNN', 'blue', vectorial=True),
              #ODEModel(DeepNN(2, NUM_HIDDEN, HIDDEN_SIZE, 2), (lambda x: euler_step(x)+x),
              #     'Full Euler Step DeepNN', 'purple'),
              ODEModel(ResNN(2, NUM_HIDDEN, HIDDEN_SIZE, 2), (lambda x: euler_step(x)+x),
                   'Euler Step ResNN', 'red')]#,
              #ODEModel(ResNN(2, NUM_HIDDEN, HIDDEN_SIZE, 2), (lambda x: runge_kutta2(x)+x),
              #     'Runge-Kutta2 ResNN', 'green')]


# Train models
for model in model_list:
   model.train(random_2d_point, LR, NUM_EPOCHS, NUM_BLOCKS)


# Present data
# Create a single figure with subplots
fig = plt.figure(figsize=(15, 10)) # 15 8

# Plot loss curves with logarithmic scale
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1

for model in model_list:
    plt.loglog(model.loss_curve, label=f'{model.name} Loss', alpha=0.8, color=model.color)
    
plt.xlabel('Epoch (log scale)')
plt.ylabel('Loss (log scale)')
plt.title('Training Loss (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2D trajectories
plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2 (spans both columns)

## Trajectory graph

#x = random_2d_point()
x = torch.tensor([[3.,3.]])
print('Initial point:', x)

num_traj_steps = int(FINAL_TIME/DELTA)

true_traj = trajectory(x, num_traj_steps, euler_step)
plot_points(true_traj, 'Euler Step', 'black')

l2_loss = nn.MSELoss()

for model in model_list:
    traj = model.calc_trajectory(x, num_traj_steps)
    plot_points(traj, model.name, model.color)
    #print(l2_loss(true_traj, traj), torch.amax(true_traj-traj))
    #save_losses(model.name, l2_loss(true_traj, traj), torch.amax(true_traj-traj))

plt.legend()
plt.title('2D Trajectories Comparison')

plt.tight_layout()
plt.show()


