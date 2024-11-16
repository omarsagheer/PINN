# # %% init
# import torch
#
# from new_trial.equations.DiffusionEquation import DiffusionPDE
# from new_trial.equations.HeatEquation import HeatPDE
# from new_trial.equations.TransportEquation import TransportPDE
# from new_trial.equations.WaveEquation import WavePDE
#
# n_sb = 64
# n_tb = 64
# n_int = 256
#
#
# # Define the PDE
# # pde = HeatPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# pde = WavePDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# # pde = TransportPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# # pde = DiffusionPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# # %% plot the training points
# pde.plot_training_points()
#
#
# # %% Train the model
# n_epochs = 1
# # Train the model
# hist = pde.fit(num_epochs=n_epochs,
#                 optimizer=pde.optimizer_LBFGS(),
#                 verbose=True)
#
# # %% Plot the loss
# # Plot the loss
# pde.plot_train_loss(hist)
#
#
# # %% Plot the solution
# # pde.relative_L2_error()
# pde.plotting()
# # %% play with output
# # def test_L2_error(PDE, n):
# #     # inputs = PDE.soboleng.draw(n_points)
# #     # inputs = PDE.convert(inputs)
# #     t = torch.linspace(PDE.domain_extrema[0, 0], PDE.domain_extrema[0, 1], n)
# #     x = torch.linspace(PDE.domain_extrema[1, 0], PDE.domain_extrema[1, 1], n)
# #     print('t')
# #     print(t)
# #     print()
# #     print('x')
# #     print(x)
# #     print()
# #     # i want inputs to be a tensor of shape (n_points, 2) where the first column is t and the second column is x
# #     inputs = torch.zeros(n, 2)
# #     inputs[:, 0] = t
# #     inputs[:, 1] = x
# #     inputs = inputs.to(torch.float64)
# #
# #     print('inputs')
# #     print(inputs)
# #     print()
# #     output = PDE.approximate_solution(inputs).reshape(-1, )
# #     print('output')
# #     print(output)
# #     print()
# #     exact_output = PDE.exact_solution(inputs).reshape(-1, )
# #     print('exact_output')
# #     print(exact_output)
# #     print()
# #     err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
# #     print('L2 Relative Error Norm: ', err.item(), '%')
# #     return err
# #
# # test_L2_error(pde, 1000)
from new_trial.equations.WaveEquation import WavePDE

# Create the model with more points
wave_pde = WavePDE(
    n_int_=2000,    # More interior points
    n_sb_=200,      # More boundary points
    n_tb_=200,      # More initial points
    time_domain_=[0, 1],
    space_domain_=[0, 1],
    n_hidden_layers=6,
    neurons=64,
    lambda_u=100
)

# Two-phase training
# Phase 1: Train with Adam
optimizer_adam = wave_pde.optimizer_ADAM(lr=1e-3)
history_adam = wave_pde.fit(num_epochs=5000, optimizer=optimizer_adam, verbose=True)

# Phase 2: Fine-tune with L-BFGS
optimizer_lbfgs = wave_pde.optimizer_LBFGS(lr=0.5)
history_lbfgs = wave_pde.fit(num_epochs=100, optimizer=optimizer_lbfgs, verbose=True)

# Evaluate
# wave_pde.relative_L2_error()
wave_pde.plotting()