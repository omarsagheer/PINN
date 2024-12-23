from abc import ABC
import torch
from torch.utils.data import DataLoader

from inverse_problem.inverse_pinn_structure.base_i_pinn import BaseIPINN


class AddingIPINNPoints(BaseIPINN, ABC):
    pass



    # add points
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)
        return input_tb, output_tb

    def add_spatial_boundary_points_left(self):
        x_left = self.domain_extrema[1, 0]
        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_left = torch.clone(input_sb)
        input_sb_left[:, 1] = torch.full(input_sb_left[:, 1].shape, x_left)

        output_sb_left = self.left_boundary_condition(input_sb_left[:, 0]).reshape(-1, 1)
        return input_sb_left, output_sb_left

    def add_spatial_boundary_points_right(self):
        x_right = self.domain_extrema[1, 1]
        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_right = torch.clone(input_sb)

        input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right)

        output_sb_right = self.right_boundary_condition(input_sb_right[:, 0]).reshape(-1, 1)
        return input_sb_right, output_sb_right


    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    def get_measurement_data(self):
        torch.random.manual_seed(42)
        # take measurements every 0.001 sec on a set of randomly placed (in space) sensors
        t = torch.linspace(0, self.domain_extrema[0, 1], 25)
        x = torch.linspace(self.domain_extrema[1, 0], self.domain_extrema[1, 1], self.n_sensors)


        input_meas = torch.cartesian_prod(t, x)

        output_meas = self.exact_solution(input_meas).reshape(-1,1)
        noise = 0.01*torch.randn_like(output_meas)
        output_meas = output_meas + noise

        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb_left, output_sb_left = self.add_spatial_boundary_points_left()
        input_sb_right, output_sb_right = self.add_spatial_boundary_points_right()
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int
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