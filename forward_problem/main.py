import time

from Common import TrainingConfig
from forward_problem.equations.wave_pinn import WaveFPINN
from forward_problem.equations.heat_pinn import HeatFPINN
from forward_problem.equations.transport_pinn import TransportFPINN

#
# def train_pde(pde_class, config, use_lbfgs=False, plot_points=False):
#     # Initialize PDE
#     n_sb = 64
#     n_tb = 64
#     n_int = 256
#     pde = pde_class(n_int=n_int, n_sb=n_sb, n_tb=n_tb, n_hidden_layers=10, neurons=128,
#                     time_domain=[0, 150], space_domain=[0, 200], regularization_param=1e-1, lambda_u=30)
#     if plot_points:
#         pde.plot_training_points()
#
#     # Create optimizer
#     if use_lbfgs:
#         optimizer = pde.optimizer_LBFGS(config)
#     else:
#         optimizer = pde.optimizer_ADAM(lr=1e-3)
#
#     # Train model
#     history = pde.enhanced_fit(
#         num_epochs=config.num_epochs,
#         optimizer=optimizer,
#         config=config,
#         verbose=True
#     )
#     # history = pde.fit(
#     #     num_epochs=config.num_epochs,
#     #     optimizer=optimizer,
#     #     # config=config,
#     #     verbose=True
#     # )
#
#     # Evaluate and plot results
#     if plot_points:
#         pde.relative_L2_error()
#         pde.plot_training_history(history)
#
#     pde.plotting_solution()
#
#     return pde, history
#

# if __name__ == "__main__":
    # Configuration for ADAM
# adam_config = TrainingConfig(
#     num_epochs=1000,
#     early_stopping_patience=30,
#     scheduler_patience=20,
#     scheduler_factor=0.5,
#     scheduler_min_lr=1e-7
# )

# Configuration for LBFGS
lbfgs_config = TrainingConfig(
    num_epochs=200,
    early_stopping_patience=20,
    max_iter=120,
)
start = time.time()
pde = WaveFPINN(n_int=256*2, n_sb=64*2, n_tb=64*2, n_hidden_layers=6, neurons=30)
pde.plot_training_points()

optimizer = pde.optimizer_LBFGS(lbfgs_config)
history = pde.enhanced_fit(
    num_epochs=lbfgs_config.num_epochs,
    optimizer=optimizer,
    config=lbfgs_config,
    verbose=True
)
end = time.time()
print('Time taken: ', end - start)
pde.plot_train_loss(history)
pde.plotting_solution()
