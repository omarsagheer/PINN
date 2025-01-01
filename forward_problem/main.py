import time

from Common import TrainingConfig
from forward_problem.equations.heat_pinn import HeatFPINN
from forward_problem.equations.transport_pinn import TransportFPINN

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
    early_stopping_patience=30,
    max_iter=50,
)
pde = TransportFPINN(n_int=256, n_sb=64, n_tb=64, n_hidden_layers=6, neurons=64)
# pde = HeatFPINN(n_int=256, n_sb=64, n_tb=64)
# pde.plot_training_points()

start = time.time()
history = pde.training(
    num_epochs=config.num_epochs,
    optimizer='lbfgs',
    config=config,
    verbose=True
)
end = time.time()
print('Time taken: ', end - start)
pde.plot_train_loss(history)
pde.plotting_solution()
