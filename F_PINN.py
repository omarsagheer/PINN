from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from new_trial.Common import NeuralNet
torch.set_default_dtype(torch.float64)

class F_PINN(ABC):
    def __init__(self, n_int_, n_sb_, n_tb_, time_domain_=None, space_domain_=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):

        if time_domain_ is None:
            time_domain_ = [0, 1]
        if space_domain_ is None:
            space_domain_ = [0, 1]
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Extrema of the solution domain (t,x)
        self.domain_extrema = torch.tensor([time_domain_, space_domain_])

        # Parameter to balance the role of data and PDE
        self.lambda_u = lambda_u
        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                  n_hidden_layers=n_hidden_layers, neurons=neurons, regularization_param=regularization_param,
                  regularization_exp=regularization_exp, retrain_seed=retrain_seed)

        self.ms = lambda x: torch.mean(torch.square(x))

    # Function to linearly transform a tensor whose value is between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    @abstractmethod
    def initial_condition(self, x):
        pass

    @abstractmethod
    def left_boundary_condition(self, t):
        pass

    @abstractmethod
    def right_boundary_condition(self, t):
        pass

    @abstractmethod
    def exact_solution(self, inputs):
        pass

    # add points
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb = input_tb.to(torch.float64)
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)

        return input_tb, output_tb

    # def add_spatial_boundary_points(self):
    #     x_left = self.domain_extrema[1, 0]
    #     x_right = self.domain_extrema[1, 1]
    #
    #     input_sb = self.convert(self.soboleng.draw(self.n_sb))
    #     input_sb = input_sb.to(torch.float64)
    #
    #     input_sb_left = torch.clone(input_sb)
    #     input_sb_left[:, 1] = torch.full(input_sb_left[:, 1].shape, x_left)
    #
    #     input_sb_right = torch.clone(input_sb)
    #     input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right)
    #
    #     output_sb_left = self.left_boundary_condition(input_sb_left[:, 0]).reshape(-1, 1)
    #     output_sb_right = self.right_boundary_condition(input_sb_right[:, 0]).reshape(-1, 1)
    #     return torch.cat((input_sb_left, input_sb_right), 0), torch.cat((output_sb_left, output_sb_right), 0)

    def add_spatial_boundary_points_left(self):
        x_left = self.domain_extrema[1, 0]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb = input_sb.to(torch.float64)

        input_sb_left = torch.clone(input_sb)
        input_sb_left[:, 1] = torch.full(input_sb_left[:, 1].shape, x_left)

        output_sb_left = self.left_boundary_condition(input_sb_left[:, 0]).reshape(-1, 1)

        return input_sb_left, output_sb_left

    def add_spatial_boundary_points_right(self):
        x_right = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb = input_sb.to(torch.float64)

        input_sb_right = torch.clone(input_sb)
        input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right)

        output_sb_right = self.right_boundary_condition(input_sb_right[:, 0]).reshape(-1, 1)
        return input_sb_right, output_sb_right


    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        input_int = input_int.to(torch.float64)
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int


    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        # input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_sb_left, output_sb_left = self.add_spatial_boundary_points_left()
        input_sb_right, output_sb_right = self.add_spatial_boundary_points_right()
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int
        # training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.n_sb, shuffle=False)
        training_set_sb_left = DataLoader(torch.utils.data.TensorDataset(input_sb_left, output_sb_left), batch_size=self.n_sb, shuffle=False)
        training_set_sb_right = DataLoader(torch.utils.data.TensorDataset(input_sb_right, output_sb_right), batch_size=self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        return training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int


    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions_left(self, input_sb):
        u_pred_sb = self.approximate_solution(input_sb)
        return u_pred_sb

    def apply_boundary_conditions_right(self, input_sb):
        u_pred_sb = self.approximate_solution(input_sb)
        return u_pred_sb

    @abstractmethod
    def compute_pde_residual(self, input_int):
        pass

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, train_points, verbose=True, new_loss=None, no_right_boundary=False):
        (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right, inp_train_tb, u_train_tb,
         inp_train_int) = train_points
        u_pred_sb_left = self.apply_boundary_conditions_left(inp_train_sb_left)
        u_pred_sb_right = self.apply_boundary_conditions_right(inp_train_sb_right)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        assert (u_pred_sb_left.shape[1] == u_train_sb_left.shape[1])
        assert (u_pred_sb_right.shape[1] == u_train_sb_right.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb_left = u_train_sb_left - u_pred_sb_left
        r_sb_right = u_train_sb_right - u_pred_sb_right
        # r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb

        # loss_sb = self.ms(r_sb)
        loss_sb_left = self.ms(r_sb_left)
        loss_sb_right = self.ms(r_sb_right)
        loss_tb = self.ms(r_tb)
        loss_int = self.ms(r_int)

        loss_u = loss_sb_left + loss_sb_right + loss_tb

        if no_right_boundary:
            loss_u = loss_sb_left + loss_tb

        if new_loss is not None:
            loss_u += new_loss

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose: print('Total loss: ', round(loss.item(), 4), '| PDE Loss: ', round(torch.log10(loss_u).item(), 4),
                          '| Function Loss: ', round(torch.log10(loss_int).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int = self.assemble_datasets()
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print('################################ ', epoch, ' ################################')

            for j, ((inp_train_sb_left, u_train_sb_left), (inp_train_sb_right, u_train_sb_right),
                    (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) \
                    in enumerate(zip(training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    train_points = (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right, inp_train_tb, u_train_tb, inp_train_int)
                    loss = self.compute_loss(train_points, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

        ################################################################################################

    def relative_L2_error(self, n_points=10000):
        inputs = self.soboleng.draw(n_points)
        inputs = self.convert(inputs)
        inputs = inputs.to(torch.float64)

        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print('L2 Relative Error Norm: ', err.item(), '%')
        return inputs, output, exact_output

    def plotting(self, n_points=10000):
        inputs, output, exact_output = self.relative_L2_error(n_points)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=exact_output.detach(), cmap='jet')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('t')
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which='both', ls=':')
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output.detach(), cmap='jet')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('t')
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which='both', ls=':')
        axs[0].set_title('Exact Solution')
        axs[1].set_title('Approximate Solution')

        plt.show()
        plt.close()

    def plot_training_points(self):
        # Plot the input training points
        # input_sb_, output_sb_ = self.add_spatial_boundary_points()
        input_sb_left_, output_sb_left_ = self.add_spatial_boundary_points_left()
        input_sb_right_, output_sb_right_ = self.add_spatial_boundary_points_right()
        input_tb_, output_tb_ = self.add_temporal_boundary_points()
        input_int_, output_int_ = self.add_interior_points()

        plt.figure(figsize=(16, 8), dpi=150)
        # plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label='Boundary Points')
        plt.scatter(input_sb_left_[:, 1].detach().numpy(), input_sb_left_[:, 0].detach().numpy(), label='Left Boundary Points')
        plt.scatter(input_sb_right_[:, 1].detach().numpy(), input_sb_right_[:, 0].detach().numpy(), label='Right Boundary Points')
        plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label='Interior Points')
        plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label='Initial Points')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_loss(hist):
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.show()


    def optimizer_LBFGS(self, lr=0.5, max_iter=1000, max_eval=10000, history_size=150, line_search_fn="strong_wolfe",
                        tolerance_change=1.0 * np.finfo(float).eps):
        return optim.LBFGS(self.approximate_solution.parameters(), lr=float(lr), max_iter=max_iter, max_eval=max_eval,
                           history_size=history_size, line_search_fn=line_search_fn, tolerance_change=tolerance_change)


    def optimizer_ADAM(self, lr=1e-5):
        return optim.Adam(self.approximate_solution.parameters(), lr=float(lr))


