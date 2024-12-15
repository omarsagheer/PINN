import numpy as np
import torch


from forward_problem.ForwardPINN import ForwardPINN


class HeatFPINN(ForwardPINN):
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):
        super().__init__(n_int, n_sb, n_tb, time_domain, space_domain, lambda_u, n_hidden_layers, neurons,
                         regularization_param, regularization_exp, retrain_seed)


    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]
        u = -torch.exp(-np.pi ** 2 * t) * torch.sin(np.pi * x)
        return u

    def initial_condition(self, x):
        return -torch.sin(np.pi * x)

    def left_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        residual = grad_u_t - grad_u_xx
        return residual.reshape(-1, )

