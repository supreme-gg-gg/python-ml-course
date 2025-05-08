# This is based on the popular PINN workshop on the harmonic oscillator
# https://github.com/benmoseley/harmonic-oscillator-pinn-workshop/blob/main/PINN_intro_workshop_student.ipynb

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# define the exact solution of the harmonic oscillator
# u'' + mu * u' + k * u = 0
# u(0) = 1, u'(0) = 0
# u(t) = exp(-d*t) * (A * cos(w*t + phi))
def exact_solution(d, w0, t):
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos
    return u

# define a very simple MLP as the PINN
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(FCN, self).__init__()
        self.input_layer = nn.Linear(N_INPUT, N_HIDDEN)
        self.hidden_layers = nn.ModuleList([nn.Linear(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS-2)])
        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x
    

torch.manual_seed(123)
    
def train_forward_simulation():
        # time is the only input here
        pinn = FCN(1, 1, 32, 4)

        # values in time we want to evaluate boundary loss
        # only at the beginning when t=0 because that's the boundary condition
        t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

        # training points over the entire domain for physics loss
        t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

        # train PINN
        d, w0 = 2, 20
        mu, k = 2*d, w0**2

        t_test = torch.linspace(0, 1, 300).view(-1, 1)
        u_exact = exact_solution(d, w0, t_test)

        optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

        for i in range(15001):
            optimizer.zero_grad()

            # compute each term of the PINN loss using lambda parameters
            lambda1, lambda2 = 1e-1, 1e-4

            # boundary loss
            u = pinn(t_boundary) # shape (1, 1)
            loss1 = (torch.squeeze(u) - 1)**2 # boundary condition u(0) = 1

            # find u' with autograd, still part of the boundary loss
            du_dt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
            loss2 = (torch.squeeze(du_dt) - 0)**2 # boundary condition u'(0) = 0

            # physics loss, run the network again on the physics points
            # all these points are unsupervised
            u = pinn(t_physics) # shape (30, 1)
            # then get the first and second derivatives
            du_dt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
            d2u_dt2 = torch.autograd.grad(du_dt, t_physics, torch.ones_like(du_dt), create_graph=True)[0]
            
            # compute the physics loss (plugging in the ODE)
            loss3 = torch.mean((d2u_dt2 + mu * du_dt + k * u)**2)

            # backprop joint loss
            loss = loss1 + lambda1 * loss2 + lambda2 * loss3
            loss.backward()
            optimizer.step()

            # copy and pasted visualization code
            if i % 5000 == 0:
                #print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
                u = (pinn(t_test)*torch.sin(a*t_test+b)).detach()
                plt.figure(figsize=(6,2.5))
                plt.scatter(t_physics.detach()[:,0],
                            torch.zeros_like(t_physics)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)
                plt.scatter(t_boundary.detach()[:,0],
                            torch.zeros_like(t_boundary)[:,0], s=20, lw=0, color="tab:red", alpha=0.6)
                plt.plot(t_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
                plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
                plt.title(f"Training step {i}")
                plt.legend()
                plt.show()

def train_inverse_task():
    # make my also a trainable parameter, this time we will remove boundary loss but add in data loss
    # data loss assumes we have some noisy data points gathered experimentally, we will keep the physics loss

    mus = []
    pinn = FCN(1, 1, 32, 4)

    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

    d, w0 = 2, 20
    _, k = 2*d, w0**2

    # generate and visualize some noisy data
    print(f"True value of mu: {2*d}")
    t_obs = torch.rand(40).view(-1,1)
    u_obs = exact_solution(d, w0, t_obs) + 0.04*torch.randn_like(t_obs)

    plt.figure()
    plt.title("Noisy observational data")
    plt.scatter(t_obs[:,0], u_obs[:,0])
    t_test, u_exact = torch.linspace(0,1,300).view(-1,1), exact_solution(d, w0, t_test)
    plt.plot(t_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
    plt.show()

    # treat mu as a trainable parameter
    mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    # add mu to the optimizer
    optimizer = torch.optim.Adam(list(pinn.parameters() + [mu]), lr=1e-3)

    for i in range(15001):
        optimizer.zero_grad()

        lambda1 = 1e-4

        # compute physics loss as above
        u = pinn(t_physics)
        du_dt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t_physics, torch.ones_like(du_dt), create_graph=True)[0]
        loss1 = torch.mean((d2u_dt2 + mu * du_dt + k * u)**2)

        # compute data loss
        u = pinn(t_obs)
        loss2 = torch.mean((u - u_obs)**2)

        # backprop joint loss
        loss = loss1 + lambda1 * loss2
        loss.backward()
        optimizer.step()

        # record mu value
        mus.append(mu.item())

        # copy and pasted visualization code
        if i % 5000 == 0:
            u = pinn(t_test).detach()
            plt.figure(figsize=(6,2.5))
            plt.scatter(t_obs[:,0], u_obs[:,0], label="Noisy observations", alpha=0.6)
            plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {i}")
            plt.legend()
            plt.show()

        plt.figure()
        plt.title("$\mu$")
        plt.plot(mus, label="PINN estimate")
        plt.hlines(2*d, 0, len(mus), label="True value", color="tab:green")
        plt.legend()
        plt.xlabel("Training step")
        plt.show()

# you will see that at higher frequencies (w0) the PINN struggles to converge
# to improve, we can assume something about the solution, e.g. intuitively u(t; theta, alpha, beta) = u_PINN(t; theta) * sin(alpha * t + beta) ~ u(t)
# then we can just make the PINN learn the parameters alpha and beta, what's left are just the exponential function to learn which is much easier

def train_ansatz_formulation():
    pinn = FCN(1, 1, 32, 4)

    # define alpha and beta as trainable parameters
    a = torch.nn.Parameter(70*torch.ones(1, requires_grad=True))
    b = torch.nn.Parameter(0*torch.ones(1, requires_grad=True))

    # define boundary point for boundary loss
    t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

    # training points for physics loss
    t_physics = torch.linspace(0, 1, 60).view(-1, 1).requires_grad_(True)

    d, w0 = 2, 80 # NOTE: w0 is a much higher frequency
    mu, k = 2*d, w0**2
    t_test = torch.linspace(0, 1, 300).view(-1, 1)
    u_exact = exact_solution(d, w0, t_test)
    optimizer = torch.optim.Adam(list(pinn.parameters() + [a, b]), lr=1e-3)

    for i in range(15001):
        optimizer.zero_grad()

        lambda1, lambda2 = 1e-1, 1e-4

        # boundary loss, using the same boundary conditions as above
        u = pinn(t_boundary) * torch.sin(a*t_boundary + b)
        loss1 = (torch.squeeze(u) -1) **2
        du_dt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
        loss2 = (torch.squeeze(du_dt) - 0)**2

        # physics loss
        u = pinn(t_physics) * torch.sin(a*t_physics + b)
        du_dt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t_physics, torch.ones_like(du_dt), create_graph=True)[0]
        loss3 = torch.mean((d2u_dt2 + mu * du_dt + k * u)**2)

        loss = loss1 + lambda1 * loss2 + lambda2 * loss3
        loss.backward()
        optimizer.step()

        if i % 5000 == 0:
            #print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
            u = (pinn(t_test)*torch.sin(a*t_test+b)).detach()
            plt.figure(figsize=(6,2.5))
            plt.scatter(t_physics.detach()[:,0],
                        torch.zeros_like(t_physics)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)
            plt.scatter(t_boundary.detach()[:,0],
                        torch.zeros_like(t_boundary)[:,0], s=20, lw=0, color="tab:red", alpha=0.6)
            plt.plot(t_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
            plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {i}")
            plt.legend()
            plt.show()

        # this can converge successfully within the 15k iterations
