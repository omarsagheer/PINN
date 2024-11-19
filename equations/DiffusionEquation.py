import numpy as np
import pandas as pd
import torch

from F_PINN import F_PINN


class DiffusionPDE(F_PINN):
    def __init__(self, n_int_, n_sb_, n_tb_, time_domain_=None, space_domain_=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):
        # initial conditions
        self.f = 0.2
        self.M = 1.8076e-4
        self.G = 10.03
        self.F = 685.0
        # self.L = self.zf
        # self.D = 200
        # space_domain_ = [0, self.zf/self.L]
        # time_domain_ = [0, self.Te*self.D/self.L**2]
        super().__init__(n_int_, n_sb_, n_tb_, time_domain_, space_domain_, lambda_u, n_hidden_layers, neurons,
                         regularization_param, regularization_exp, retrain_seed, rescale_to_0_1=True)
        self.zf = space_domain_[1]
        self.Te = time_domain_[1]


    @staticmethod
    def D_alpha(x):
        return 200 - 199.98 * x
        # return 1 - 0.9999 * x
    def initial_condition(self, x):
        return torch.zeros(x.shape[0], 1)

    def left_boundary_condition(self, t):
        # return 2* t **0.25
        return 2* (self.Te*t) **0.25
    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1)

    def exact_solution(self, inputs):
        # t = inputs[:, 0]
        # x = inputs[:, 1]
        # u = torch.square(x-t)
        return 0

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]
        # grad_u_tt = torch.autograd.grad(grad_u_t.sum(), input_int, create_graph=True)[0][:, 0]

        D_alpha = self.D_alpha(input_int[:, 1])
        # D_alpha_x = torch.autograd.grad(D_alpha.sum(), input_int, create_graph=True)[0][:, 1]
        D_alpha_x = -199.98
        left_side = (grad_u_t * self.f)/self.Te + (grad_u_x*self.f*self.F)/self.zf + u*self.G
        right_side = (D_alpha_x*(grad_u_x/self.zf - u*self.M) + D_alpha*(grad_u_xx/self.zf - grad_u_x*self.M))/self.zf
        # left_side = (grad_u_t * self.f) + (grad_u_x*self.f*self.F) + u*self.G
        # right_side = (D_alpha_x*(grad_u_x - u*self.M) + D_alpha*(grad_u_xx - grad_u_x*self.M))
        residual = (left_side - right_side)/200
        # Pe = self.f * self.F * self.zf / self.D
        # Da = self.G * self.zf**2 / self.D
        # Gr = self.M * self.zf
        # left_side = grad_u_t + (Pe * grad_u_x) + Da * u
        # right_side = (D_alpha_x * (grad_u_x - Gr * u) + D_alpha * (grad_u_xx - Gr * grad_u_x))
        # residual = left_side - right_side
        return residual.reshape(-1, )

    def apply_right_boundary_derivative(self, inp_train_sb_right):
        inp_train_sb_right.requires_grad = True
        u = self.approximate_solution(inp_train_sb_right)
        grad_u = torch.autograd.grad(u.sum(), inp_train_sb_right, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]
        x_right = inp_train_sb_right[:, 1]
        D_alpha = self.D_alpha(x_right)
        return self.ms(D_alpha*(grad_u_x - self.M*u) - self.right_boundary_condition(inp_train_sb_right[:, 0]))
        # return self.ms(grad_u_x - self.right_boundary_condition(inp_train_sb_right[:, 0]))

    def compute_loss(self, train_points, verbose=True, new_loss=None, right_boundary_loss=None):
        inp_train_sb_right = train_points[2]
        right_boundary_loss = self.apply_right_boundary_derivative(inp_train_sb_right)
        loss = super().compute_loss(train_points, verbose, right_boundary_loss, True)
        return loss

    def relative_L2_error(self, n_points=10000):
        path = r'/Users/omar/Desktop/PINN/exact_data.xlsx'
        data = pd.read_excel(path, header=None)
        x = data[0].values
        t = data[1].values
        inputs = torch.tensor(np.stack((t, x), axis=1), dtype=torch.float64)
        output = self.approximate_solution(inputs).reshape(-1, )
        # exact_output = self.exact_solution(inputs).reshape(-1, )
        exact_output = data[2].values.reshape(-1, )
        exact_output = torch.tensor(exact_output, dtype=output.dtype)

        err = (torch.mean((output.detach() - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")
        return inputs, output, exact_output
