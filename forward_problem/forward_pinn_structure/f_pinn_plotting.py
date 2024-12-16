from abc import ABC

import numpy as np
import torch
from matplotlib import pyplot as plt

from forward_problem.forward_pinn_structure.adding_f_pinn_points import AddingFPINNPoints


class FPINNPlotting(AddingFPINNPoints, ABC):
    pass

    def relative_L2_error(self, n_points=10000):
        inputs = self.soboleng.draw(n_points)
        inputs = self.convert(inputs)
        inputs = inputs.to(self.dtype)
        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
        print('L2 Relative Error Norm: {:.6e}'.format(err.item()))
        return inputs, output, exact_output

    def plotting_solution(self, n_points=50000):
        inputs, output, exact_output = self.relative_L2_error(n_points)
        # inputs = inputs.cpu()
        # output = output.cpu()
        # exact_output = exact_output.cpu()
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
        input_sb_left_, _ = self.add_spatial_boundary_points_left()
        input_sb_right_, _= self.add_spatial_boundary_points_right()
        input_tb_, _ = self.add_temporal_boundary_points()
        input_int_, _ = self.add_interior_points()
        #
        # input_sb_left_ = copy.deepcopy(input_sb_left_).cpu()
        # input_sb_right_ = copy.deepcopy(input_sb_right_).cpu()
        # input_tb_ = copy.deepcopy(input_tb_).cpu()
        # input_int_ = copy.deepcopy(input_int_).cpu()

        plt.figure(figsize=(16, 8), dpi=150)
        plt.scatter(input_sb_left_[:, 1].detach().numpy(), input_sb_left_[:, 0].detach().numpy(), label='Left Boundary Points')
        plt.scatter(input_sb_right_[:, 1].detach().numpy(), input_sb_right_[:, 0].detach().numpy(), label='Right Boundary Points')
        plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label='Interior Points')
        plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label='Initial Points')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_loss(history):
        hist = history['total_loss']
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_training_history(history):
        """Plot training history including losses and learning rate."""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot losses
        ax1.plot(history['total_loss'], label='Total Loss')
        ax1.plot(history['pde_loss'], label='PDE Loss')
        ax1.plot(history['boundary_loss'], label='Boundary Loss')
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        ax2.plot(history['learning_rate'], label='Learning Rate')
        ax2.set_yscale('log')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
