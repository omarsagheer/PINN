from abc import ABC

from forward_pinn_structure.f_pinn_loss import FPINNLoss
from forward_problem.forward_pinn_structure.util_f_pinn import UtilFPINN


class ForwardPINN(FPINNLoss, UtilFPINN, ABC):
    # This class is a "funnel" class that inherits from all the other classes
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10,
                    n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):
            super().__init__(n_int, n_sb, n_tb, time_domain, space_domain, lambda_u, n_hidden_layers, neurons,
                            regularization_param, regularization_exp, retrain_seed)
