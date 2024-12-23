import time

from forward_problem.Common import TrainingConfig
from forward_problem.equations.heat_pinn import HeatFPINN

# Configuration for ADAM
# config = TrainingConfig(
#     num_epochs=5000,
#     early_stopping_patience=30,
#     scheduler_patience=20,
#     scheduler_factor=0.5,
#     scheduler_min_lr=1e-7
# )

# Configuration for LBFGS
config = TrainingConfig(
    num_epochs=100,
    early_stopping_patience=20,
    max_iter=25,
)
start = time.time()
pde = HeatFPINN(n_int=256, n_sb=64, n_tb=64)
pde.plot_training_points()

optimizer = pde.optimizer_LBFGS(config)
history = pde.approximate_solution()
end = time.time()
print('Time taken: ', end - start)
pde.plot_train_loss(history)
pde.plotting_solution()
