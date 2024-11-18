from Common import TrainingConfig
from equations.HeatEquation import HeatPDE
from equations.WaveEquation import WavePDE


def train_pde(pde_class, config, use_lbfgs=False, plot_points=False):
    # Initialize PDE
    n_sb = 64*2
    n_tb = 64*2
    n_int = 256*2
    pde = pde_class(n_int_=n_int, n_sb_=n_sb, n_tb_=n_tb, lambda_u=10, n_hidden_layers=8, neurons=64)

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

    # Evaluate and plot results
    if plot_points:
        pde.relative_L2_error()
        pde.plot_training_history(history)

    pde.plotting()

    return pde, history


if __name__ == "__main__":
    # Configuration for ADAM
    adam_config = TrainingConfig(
        num_epochs=10000,
        early_stopping_patience=20,
        scheduler_patience=10,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6
    )

    # Configuration for LBFGS
    lbfgs_config = TrainingConfig(
        num_epochs=300,
        early_stopping_patience=20,
        max_iter=69,
    )

    # Train Wave equation with ADAM
    wave_pde, wave_history = train_pde(WavePDE, lbfgs_config, use_lbfgs=True)
    # wave_pde, wave_history = train_pde(WavePDE, adam_config, use_lbfgs=False)

    # Train Heat equation with LBFGS
    # heat_pde, heat_history = train_pde(HeatPDE, lbfgs_config, use_lbfgs=True)
