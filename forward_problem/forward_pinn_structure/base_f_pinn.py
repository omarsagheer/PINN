from abc import ABC, abstractmethod
import torch

from Common import NeuralNet


class BaseFPINN(ABC):
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.dtype = torch.float32
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
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                  n_hidden_layers=n_hidden_layers, neurons=neurons, regularization_param=regularization_param,
                  regularization_exp=regularization_exp, retrain_seed=retrain_seed, device=device).to(device)

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

    @abstractmethod
    def compute_pde_residual(self, input_int):
        pass