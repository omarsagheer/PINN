from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

from Common import PINN


class BaseFPINN(ABC):
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        if time_domain is None:
            time_domain = [0, 1]
        if space_domain is None:
            space_domain = [0, 1]

        self.n_int = n_int
        self.n_sb = n_sb
        self.n_tb = n_tb

        # Extrema of the solution domain (t,x)
        self.domain_extrema = torch.tensor([time_domain, space_domain])

        # Parameter to balance the role of data and PDE
        self.lambda_u = lambda_u
        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = PINN(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                  n_hidden_layers=n_hidden_layers, neurons=neurons, regularization_param=regularization_param,
                  regularization_exp=regularization_exp, retrain_seed=retrain_seed).to(device)

        self.ms = lambda x: torch.mean(torch.square(x))

    # Function to linearly transform a tensor whose value is between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def generate_sobol_points(self, n_points):
        return self.convert(self.soboleng.draw(n_points)).to(self.device) # noqa

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

    @abstractmethod
    def compute_pde_residual(self, input_int):
        pass


    # add points
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.generate_sobol_points(self.n_tb)
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)
        output_tb = output_tb.to(self.device)
        return input_tb, output_tb

    def add_spatial_boundary_points_left(self):
        x_left = self.domain_extrema[1, 0]
        input_sb = self.generate_sobol_points(self.n_sb)
        input_sb_left = torch.clone(input_sb)
        input_sb_left[:, 1] = torch.full(input_sb_left[:, 1].shape, x_left)

        output_sb_left = self.left_boundary_condition(input_sb_left[:, 0]).reshape(-1, 1)
        output_sb_left = output_sb_left.to(self.device)
        return input_sb_left, output_sb_left

    def add_spatial_boundary_points_right(self):
        x_right = self.domain_extrema[1, 1]
        input_sb = self.generate_sobol_points(self.n_sb)
        input_sb_right = torch.clone(input_sb)

        input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right)

        output_sb_right = self.right_boundary_condition(input_sb_right[:, 0]).reshape(-1, 1)
        output_sb_right = output_sb_right.to(self.device)
        return input_sb_right, output_sb_right


    # Function returning the input-output tensor required to assemble the training set S_int corresponding to the
    # interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.generate_sobol_points(self.n_int)
        output_int = torch.zeros((input_int.shape[0], 1), device=self.device)
        return input_int, output_int


    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb_left, output_sb_left = self.add_spatial_boundary_points_left()
        input_sb_right, output_sb_right = self.add_spatial_boundary_points_right()
        input_tb, output_tb = self.add_temporal_boundary_points()
        input_int, output_int = self.add_interior_points()
        training_set_sb_left = DataLoader(torch.utils.data.TensorDataset(input_sb_left, output_sb_left), batch_size=self.n_sb, shuffle=False)
        training_set_sb_right = DataLoader(torch.utils.data.TensorDataset(input_sb_right, output_sb_right), batch_size=self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        return training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int