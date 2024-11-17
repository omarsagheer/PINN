# %% init
import torch

from Common import TrainingConfig
from equations.HeatEquation import HeatPDE
from equations.WaveEquation import WavePDE

n_sb = 64
n_tb = 64
n_int = 256


# Define the PDE
# pde = HeatPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
pde = WavePDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# pde = TransportPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)
# pde = DiffusionPDE(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb)

# %% config
config = TrainingConfig(
    num_epochs=10000,
    early_stopping_patience=20,
    scheduler_patience=10,
    scheduler_factor=0.5,
    scheduler_min_lr=1e-6
)

# for LBFGS
# config = TrainingConfig(
#     num_epochs=300,
#     early_stopping_patience=20,
#     max_iter=100,  # Maximum iterations per optimization step
#     history_size=50,
#     line_search_fn="strong_wolfe"
# )
# %% plot the training points
pde.plot_training_points()


# %% Train the model
# Train the model
optimizer = pde.optimizer_ADAM(lr=1e-3)
# optimizer = pde.optimizer_LBFGS(config)
history = pde.enhanced_fit(
    num_epochs=config.num_epochs,
    optimizer=optimizer,
    config=config,
    verbose=True
)

# %% Plot the loss
# Plot the loss
# pde.plot_train_loss(hist)
pde.plot_training_history(history)

# %% Plot the solution
# pde.relative_L2_error()
pde.plotting()
# %% play with output
# def test_L2_error(PDE, n):
#     # inputs = PDE.soboleng.draw(n_points)
#     # inputs = PDE.convert(inputs)
#     t = torch.linspace(PDE.domain_extrema[0, 0], PDE.domain_extrema[0, 1], n)
#     x = torch.linspace(PDE.domain_extrema[1, 0], PDE.domain_extrema[1, 1], n)
#     print('t')
#     print(t)
#     print()
#     print('x')
#     print(x)
#     print()
#     # i want inputs to be a tensor of shape (n_points, 2) where the first column is t and the second column is x
#     inputs = torch.zeros(n, 2)
#     inputs[:, 0] = t
#     inputs[:, 1] = x
#     inputs = inputs.to(torch.float64)
#
#     print('inputs')
#     print(inputs)
#     print()
#     output = PDE.approximate_solution(inputs).reshape(-1, )
#     print('output')
#     print(output)
#     print()
#     exact_output = PDE.exact_solution(inputs).reshape(-1, )
#     print('exact_output')
#     print(exact_output)
#     print()
#     err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
#     print('L2 Relative Error Norm: ', err.item(), '%')
#     return err
#
# test_L2_error(pde, 1000)