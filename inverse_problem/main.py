import time

from Common import TrainingConfig
from inverse_problem.equations.heat_i_pinn import HeatIPINN

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
    max_iter=30,
)
start = time.time()
pde = HeatIPINN(n_int=256, n_sb=64, n_tb=64, n_sensors=64)
pde.plot_training_points()

optimizer = pde.optimizer_LBFGS(config)
history = pde.enhanced_fit(
    num_epochs=config.num_epochs,
    optimizer=optimizer,
    config=config,
    verbose=True
)
end = time.time()
print('Time taken: ', end - start)
pde.plot_train_loss(history)
# pde
pde.plotting_solution(function='pde')

# coefficient
pde.plotting_solution(function='coefficient')
