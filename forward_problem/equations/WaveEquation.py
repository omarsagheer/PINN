import numpy as np
import torch

from forward_problem.ForwardPINN import ForwardFPINN
torch.set_default_dtype(torch.float64)

class WavePDE(ForwardFPINN):
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):
        self.c = 10.0
        time_domain = [0, 1/self.c]
        space_domain = [0, self.c]
        super().__init__(n_int, n_sb, n_tb, time_domain, space_domain, lambda_u, n_hidden_layers, neurons,
                         regularization_param, regularization_exp, retrain_seed)


    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]
        u = torch.sin(np.pi * x) * torch.cos(np.pi * self.c * t)
        return u

    def initial_condition(self, x):
        return torch.sin(np.pi * x)

    def left_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)


    def apply_initial_derivative_condition(self, input_tb):
        # Additional condition for du/dt at t=0 velocity
        input_tb.requires_grad = True
        u = self.approximate_solution(input_tb)
        grad_u = torch.autograd.grad(u.sum(), input_tb, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        return self.ms(grad_u_t - torch.zeros(input_tb.shape[0], 1))

    def compute_pde_residual(self, input_int):
        # utt - c^2 * uxx = 0
        grad_u = super().compute_pde_residual(input_int)
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]
        grad_u_tt = torch.autograd.grad(grad_u_t.sum(), input_int, create_graph=True)[0][:, 0]

        residual = (grad_u_tt - self.c ** 2 * grad_u_xx)/self.c**2
        return residual.reshape(-1, )

    def compute_loss(self, train_points, verbose=True, new_loss=None, no_right_boundary=False):
        inp_train_tb = train_points[4]
        initial_derivative_loss = self.apply_initial_derivative_condition(inp_train_tb)
        loss = super().compute_loss(train_points, verbose, initial_derivative_loss, no_right_boundary)
        return loss


