from abc import ABC

import numpy as np
import torch
from matplotlib import pyplot as plt

from forward_problem.forward_pinn_structure.adding_pde_points import AddingPDEPoints
from forward_problem.forward_pinn_structure.main_pde import MainPDE


class PDEPlotting(MainPDE, AddingPDEPoints, ABC):
    pass

    def relative_L2_error(self, n_points=10000):
        inputs = self.soboleng.draw(n_points)
        inputs = self.convert(inputs)
        inputs = inputs.to(torch.dtype)

        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )

        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
        # print in scientific notation the L2 norm
        print('')
        print('L2 Relative Error Norm: ', err.item(), '%')
        return inputs, output, exact_output

    def plotting_solution(self, n_points=25000):
        inputs, output, exact_output = self.relative_L2_error(n_points)
        inputs = inputs.cpu()
        output = output.cpu()
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
        input_sb_left_, output_sb_left_ = self.add_spatial_boundary_points_left()
        input_sb_right_, output_sb_right_ = self.add_spatial_boundary_points_right()
        input_tb_, output_tb_ = self.add_temporal_boundary_points()
        input_int_, output_int_ = self.add_interior_points()

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
    def plot_train_loss(hist):
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
