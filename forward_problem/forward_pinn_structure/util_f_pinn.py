from abc import ABC

import numpy as np
import torch
from matplotlib import pyplot as plt

from forward_problem.forward_pinn_structure.base_f_pinn import BaseFPINN


class UtilFPINN(BaseFPINN, ABC):
    """Utility class for the forward PINN structure."""

    def get_points(self, n_points):
        inputs = self.generate_sobol_points(n_points)
        output = self.approximate_solution(inputs).reshape(-1, )
        exact_output = self.exact_solution(inputs).reshape(-1, )
        return inputs, output, exact_output

    def relative_L2_error(self, n_points=10000):
        inputs, output, exact_output = self.get_points(n_points)
        err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
        print('L2 Relative Error Norm: {:.6e}'.format(err.item()))
        return inputs, output, exact_output

    def plotting_solution(self, n_points=100000):
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
        input_sb_left_, _ = self.add_spatial_boundary_points_left()
        input_sb_right_, _= self.add_spatial_boundary_points_right()
        input_tb_, _ = self.add_temporal_boundary_points()
        input_int_, _ = self.add_interior_points()

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
        plt.xlabel("Epoch")
        plt.ylabel('Log10 Loss')
        plt.legend()
        plt.show()