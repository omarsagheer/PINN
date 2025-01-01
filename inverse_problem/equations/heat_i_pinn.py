import numpy as np
import torch

from inverse_problem.InversePINN import InversePINN


class HeatIPINN(InversePINN):
    def __init__(self, n_int, n_sb, n_tb, n_sensors, **kwargs):
        super().__init__(n_int, n_sb, n_tb, n_sensors,[0, 1], [0, 1], **kwargs)


    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]
        u = -torch.exp(-np.pi ** 2 * t) * torch.sin(np.pi * x)
        return u

    def initial_condition(self, x):
        return -torch.sin(np.pi * x)

    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def left_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def exact_coefficient(self, inputs):
        x = inputs[:, 1]
        k = (torch.sin(np.pi * x) + 1.1)

        return k

    def source(self, inputs):

        s = -np.pi**2*self.exact_solution(inputs)*(1 - self.exact_coefficient(inputs))
        return s
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        k = self.approximate_coefficient(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        residual = grad_u_t - k*grad_u_xx - self.source(input_int)
        return residual.reshape(-1, )

