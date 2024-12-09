import time

from Common import TrainingConfig
from GasPDE import GasPDE
from equations.DiffusionEquation import DiffusionPDE
from equations.HeatEquation import HeatPDE
from equations.TransportEquation import TransportPDE
from equations.WaveEquation import WavePDE


def train_pde(pde_class, config, use_lbfgs=False, plot_points=False):
    # Initialize PDE
    n_sb = 64
    n_tb = 64
    n_int = 256
    pde = pde_class(n_int=n_int, n_sb=n_sb, n_tb=n_tb, n_hidden_layers=10, neurons=128,
                    time_domain=[0, 150], space_domain=[0, 200], regularization_param=1e-1, lambda_u=30)
    if plot_points:
        pde.plot_training_points()

    # Create optimizer
    if use_lbfgs:
        optimizer = pde.optimizer_LBFGS(config)
    else:
        optimizer = pde.optimizer_ADAM(lr=1e-3)

    # Train model
    history = pde.enhanced_fit(
        num_epochs=config.num_epochs,
        optimizer=optimizer,
        config=config,
        verbose=True
    )
    # history = pde.fit(
    #     num_epochs=config.num_epochs,
    #     optimizer=optimizer,
    #     # config=config,
    #     verbose=True
    # )

    # Evaluate and plot results
    if plot_points:
        pde.relative_L2_error()
        pde.plot_training_history(history)

    pde.plotting_solution()

    return pde, history


if __name__ == "__main__":
    # Configuration for ADAM
    adam_config = TrainingConfig(
        num_epochs=1000,
        early_stopping_patience=30,
        scheduler_patience=20,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-7
    )

    # Configuration for LBFGS
    lbfgs_config = TrainingConfig(
        num_epochs=100,
        early_stopping_patience=20,
        max_iter=50,
    )
    start = time.time()
    heat_pde = TransportPDE(n_int=256, n_sb=64, n_tb=64, space_domain=[0, 5], time_domain=[0, 20])
    heat_pde.plot_training_points()

    optimizer = heat_pde.optimizer_LBFGS(lbfgs_config)
    history = heat_pde.enhanced_fit(
        num_epochs=lbfgs_config.num_epochs,
        optimizer=optimizer,
        config=lbfgs_config,
        verbose=True
    )
    # heat_pde.plot_training_history(history)
    end = time.time()
    print('Time taken: ', end - start)
    heat_pde.plotting_solution(100000)
    # Diffusion equation
    # diffusion_pde, diffusion_history = train_pde(GasPDE, adam_config, use_lbfgs=False, plot_points=False)
    # diffusion_pde, diffusion_history = train_pde(GasPDE, lbfgs_config, use_lbfgs=True, plot_points=False)

    # Train GasPDE equation with ADAM then optimize with LBFGS
    # Train Wave equation with ADAM
    # wave_pde, wave_history = train_pde(WavePDE, lbfgs_config, use_lbfgs=True)
    # wave_pde, wave_history = train_pde(WavePDE, adam_config, use_lbfgs=False)

    # Train Heat equation with LBFGS
    # heat_pde, heat_history = train_pde(HeatPDE, lbfgs_config, use_lbfgs=True)


    # save model
    # path = 'diffusion_pde.pth'
    # diffusion_pde.save_model(path)

    # Initialize PDE
    # n_sb = 64*2*2
    # n_tb = 64*2*2
    # n_int = 256*2*2
    # pde = GasPDE(n_int=n_int, n_sb=n_sb, n_tb=n_tb, n_hidden_layers=10, neurons=128,
    #                 time_domain=[0, 150], space_domain=[0, 200], regularization_param=1e-1)
    #
    # # optimize with Adam
    # optimizer = pde.optimizer_ADAM(lr=1e-3)
    # adam_history = pde.enhanced_fit(
    #     num_epochs=adam_config.num_epochs,
    #     optimizer=optimizer,
    #     config=adam_config,
    #     verbose=True
    # )
    #
    # # optimize with LBFGS
    # optimizer = pde.optimizer_LBFGS(lbfgs_config)
    # lbfgs_history = pde.enhanced_fit(
    #     num_epochs=lbfgs_config.num_epochs,
    #     optimizer=optimizer,
    #     config=lbfgs_config,
    #     verbose=True
    # )
    #
    # # plot results
    # pde.plotting()