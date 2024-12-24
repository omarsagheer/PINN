from abc import ABC

from inverse_problem.inverse_pinn_structure.i_pinn_loss import IPINNLoss
from inverse_problem.inverse_pinn_structure.i_pinn_optimizers import IPINNOptimizers
from inverse_problem.inverse_pinn_structure.i_pinn_plotting import IPINNPlotting


class InversePINN(IPINNOptimizers, IPINNLoss, IPINNPlotting, ABC):
    # This class is a "funnel" class that inherits from all the other classes
    def __init__(self, n_int, n_sb, n_tb, n_sensors, time_domain=None, space_domain=None, lambda_u=10,
                    n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42):
            super().__init__(n_int, n_sb, n_tb, n_sensors, time_domain, space_domain, lambda_u, n_hidden_layers, neurons,
                            regularization_param, regularization_exp, retrain_seed)
