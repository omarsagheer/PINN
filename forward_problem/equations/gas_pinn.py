import numpy as np
import pandas as pd
import torch

from forward_problem.ForwardPINN import ForwardPINN


class GasFPINN(ForwardPINN):
    def __init__(self, n_int, n_sb, n_tb, **kwargs):
        # initial conditions
        self.f = 0.2
        M_alpha, g = 0.04, 9.8
        R, T = 8.314, 260
        self.M = (M_alpha * g) / (R * T)
        self.G = 10+0.03
        self.F = 200+485
        space_domain, time_domain = [0, 1], [0, 1]
        self.Te = time_domain[1]
        self.zf = space_domain[1]
        super().__init__(n_int, n_sb, n_tb, space_domain, time_domain, **kwargs)


    @staticmethod
    def D_alpha(x):
        return 200 - 199.98 * x

    def initial_condition(self, x):
        return torch.zeros(x.shape[0], 1)

    def left_boundary_condition(self, t):
        return 2* (t **0.25)

    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def exact_solution(self, inputs):
        return 0

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        # get the derivative of D_alpha
        D_alpha = self.D_alpha(input_int[:, 1])
        D_alpha_x = torch.autograd.grad(D_alpha.sum(), input_int, create_graph=True)[0][:, 1]

        left_side = (grad_u_t * self.f)/self.Te + (grad_u_x*self.f*self.F)/self.zf + u*self.G

        # get the ride side of the equation
        right_side_1 = D_alpha_x*grad_u_x/self.zf**2 - D_alpha_x*u*self.M/self.zf
        right_side_2 = D_alpha*grad_u_xx/self.zf**2 - D_alpha*grad_u_x*self.M/self.zf
        right_side = right_side_1 + right_side_2

        residual = (left_side - right_side)

        return residual.reshape(-1, )

    def apply_right_boundary_derivative(self, inp_train_sb_right):
        inp_train_sb_right.requires_grad = True
        u = self.approximate_solution(inp_train_sb_right)
        grad_u = torch.autograd.grad(u.sum(), inp_train_sb_right, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]
        x_right = inp_train_sb_right[:, 1]
        D_alpha = self.D_alpha(x_right)
        return self.ms(D_alpha*(grad_u_x/self.zf - self.M*u) - self.right_boundary_condition(inp_train_sb_right[:, 0]))
        # return self.ms(grad_u_x - self.right_boundary_condition(inp_train_sb_right[:, 0]))

    def compute_loss(self, train_points, verbose=True, new_loss=None, no_right_boundary=None):
        inp_train_sb_right = train_points[2]
        right_boundary_loss = self.apply_right_boundary_derivative(inp_train_sb_right)
        loss = super().compute_loss(train_points, verbose, new_loss=right_boundary_loss, no_right_boundary=True)
        return loss

    def get_points(self, n_points):
        path = r'/Users/omar/Desktop/Thesis/thesis_code/PINN/firn_exact_data.xlsx'
        data = pd.read_excel(path, header=None)
        x = data[0].values
        t = data[1].values
        inputs = torch.tensor(np.stack((t, x), axis=1), dtype=self.dtype)
        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = data[2].values.reshape(-1, )
        exact_output = torch.tensor(exact_output, dtype=output.dtype)
        return inputs, output, exact_output
