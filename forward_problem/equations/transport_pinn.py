import torch

from forward_problem.ForwardPINN import ForwardPINN


class TransportFPINN(ForwardPINN):
    def __init__(self, n_int, n_sb, n_tb, **kwargs):

        super().__init__(n_int, n_sb, n_tb, [0, 20], [0, 5], **kwargs)
        self.c = 1.0

    def initial_condition(self, x):
        return torch.square(x)

    def left_boundary_condition(self, t):
        return torch.square(t)

    def right_boundary_condition(self, t):
        return 2 * (5- t)

    def exact_solution(self, inputs):
        t = inputs[:, 0]
        x = inputs[:, 1]
        u = torch.square(x-t)
        return u

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]

        residual = grad_u_t + self.c * grad_u_x
        return residual.reshape(-1, )

    def apply_right_boundary_derivative(self, inp_train_sb_right):
        inp_train_sb_right.requires_grad = True
        u = self.approximate_solution(inp_train_sb_right)
        grad_u = torch.autograd.grad(u.sum(), inp_train_sb_right, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]
        return self.ms(grad_u_x - self.right_boundary_condition(inp_train_sb_right[:, 0]))

    def compute_loss(self, train_points, verbose=True, new_loss=None, no_right_boundary=None):
        inp_train_sb_right = train_points[2]
        right_boundary_loss = self.apply_right_boundary_derivative(inp_train_sb_right)
        loss = super().compute_loss(train_points, verbose, right_boundary_loss, no_right_boundary=True)
        return loss
