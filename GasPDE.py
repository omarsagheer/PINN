import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from Common import NeuralNet, TrainingConfig, EarlyStopping

torch.set_default_dtype(torch.float64)


class GasPDE:
    def __init__(self, n_int_, n_sb_, n_tb_, time_domain_=None, space_domain_=None, lambda_u=10,
                 n_hidden_layers=4, neurons=20, regularization_param=0., regularization_exp=2., retrain_seed=42,
                 rescale_to_0_1=True, device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        # torch.set_default_dtype(torch.float64)

        if time_domain_ is None:
            time_domain_ = [0, 1]
        if space_domain_ is None:
            space_domain_ = [0, 1]

        if rescale_to_0_1:
            self.Te = time_domain_[1]
            self.zf = space_domain_[1]
        else:
            self.Te = 1
            self.zf = 1

        self.rescale_to_0_1 = rescale_to_0_1
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Move domain extrema to GPU
        self.domain_extrema = torch.tensor([time_domain_, space_domain_], dtype=torch.float64).to(device)

        self.lambda_u = lambda_u
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Move neural network to GPU
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=n_hidden_layers,
            neurons=neurons,
            regularization_param=regularization_param,
            regularization_exp=regularization_exp,
            retrain_seed=retrain_seed
        ).to(device)

        self.ms = lambda x: torch.mean(torch.square(x))

        self.f = 0.2
        self.M = 1.8076e-4
        self.G = 10.03
        self.F = 685.0

    # Function to linearly transform a tensor whose value is between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        tens = tens.to(torch.float64).to(self.device)
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        if self.rescale_to_0_1:
            return tens
        else:
            return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    @staticmethod
    def D_alpha(x):
        return 200 - 199.98 * x
        # return 1 - 0.9999 * x


    def initial_condition(self, x):
        return torch.zeros(x.shape[0], 1, dtype=torch.float64, device=self.device)

    def left_boundary_condition(self, t):
        # return 2* t **0.25
        return 2* (self.Te*t) **0.25


    def right_boundary_condition(self, t):
        return torch.zeros(t.shape[0], 1, dtype=torch.float64, device=self.device)

    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0, dtype=torch.float64, device=self.device)
        output_tb = self.initial_condition(input_tb[:, 1]).reshape(-1, 1)
        return input_tb, output_tb

    def add_spatial_boundary_points_left(self):
        x_left = self.domain_extrema[1, 0]
        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_left = torch.clone(input_sb)
        input_sb_left[:, 1] = torch.full(input_sb_left[:, 1].shape, x_left, dtype=torch.float64, device=self.device)
        output_sb_left = self.left_boundary_condition(input_sb_left[:, 0]).reshape(-1, 1)
        return input_sb_left, output_sb_left

    def add_spatial_boundary_points_right(self):
        x_right = self.domain_extrema[1, 1]
        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_right = torch.clone(input_sb)

        if self.rescale_to_0_1:
            input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right / self.zf, dtype=torch.float64, device=self.device)
        else:
            input_sb_right[:, 1] = torch.full(input_sb_right[:, 1].shape, x_right, dtype=torch.float64, device=self.device)

        output_sb_right = self.right_boundary_condition(input_sb_right[:, 0]).reshape(-1, 1)
        return input_sb_right, output_sb_right


    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1), dtype=torch.float64, device=self.device)
        return input_int, output_int


    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb_left, output_sb_left = self.add_spatial_boundary_points_left() # noqa
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

    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True # noqa
        u = self.approximate_solution(input_int)
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        # input_int = input_int.cpu()
        # grad_u_x = grad_u_x.cpu()
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]
        # grad_u_tt = torch.autograd.grad(grad_u_t.sum(), input_int, create_graph=True)[0][:, 0]
        grad_u_xx = grad_u_xx.to(self.device)
        D_alpha = self.D_alpha(input_int[:, 1])
        # D_alpha_x = torch.autograd.grad(D_alpha.sum(), input_int, create_graph=True)[0][:, 1]
        D_alpha_x = -199.98
        left_side = (grad_u_t * self.f)/self.Te + (grad_u_x*self.f*self.F)/self.zf + u*self.G
        right_side = (D_alpha_x*(grad_u_x/self.zf - u*self.M) + D_alpha*(grad_u_xx/self.zf - grad_u_x*self.M))/self.zf
        # left_side = (grad_u_t * self.f) + (grad_u_x*self.f*self.F) + u*self.G
        # right_side = (D_alpha_x*(grad_u_x - u*self.M) + D_alpha*(grad_u_xx - grad_u_x*self.M))
        residual = (left_side - right_side)/200
        # Pe = self.f * self.F * self.zf / self.D
        # Da = self.G * self.zf**2 / self.D
        # Gr = self.M * self.zf
        # left_side = grad_u_t + (Pe * grad_u_x) + Da * u
        # right_side = (D_alpha_x * (grad_u_x - Gr * u) + D_alpha * (grad_u_xx - Gr * grad_u_x))
        # residual = left_side - right_side
        return residual.reshape(-1, )

    def apply_right_boundary_derivative(self, inp_train_sb_right):
        inp_train_sb_right.requires_grad = True # noqa
        u = self.approximate_solution(inp_train_sb_right)
        grad_u = torch.autograd.grad(u.sum(), inp_train_sb_right, create_graph=True)[0]
        grad_u_x = grad_u[:, 1]
        x_right = inp_train_sb_right[:, 1]
        D_alpha = self.D_alpha(x_right)
        return self.ms(D_alpha*(grad_u_x - self.M*u) - self.right_boundary_condition(inp_train_sb_right[:, 0]))

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, train_points, verbose=True, no_right_boundary=True):
        (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right, # noqa
         inp_train_tb, u_train_tb, inp_train_int) = train_points

        # Compute boundary predictions
        u_pred_sb_left = self.apply_boundary_conditions_left(inp_train_sb_left)
        u_pred_sb_right = self.apply_boundary_conditions_right(inp_train_sb_right)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        # Validate shapes
        assert (u_pred_sb_left.shape[1] == u_train_sb_left.shape[1])
        assert (u_pred_sb_right.shape[1] == u_train_sb_right.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])

        # Compute residuals
        r_int = self.compute_pde_residual(inp_train_int)
        r_sb_left = u_train_sb_left - u_pred_sb_left
        r_sb_right = u_train_sb_right - u_pred_sb_right
        r_tb = u_train_tb - u_pred_tb

        # Compute individual losses
        loss_sb_left = self.ms(r_sb_left)
        loss_sb_right = self.ms(r_sb_right)
        loss_tb = self.ms(r_tb)
        loss_int = self.ms(r_int)
        # print('loss_sb_left: ', loss_sb_left)
        # # print('loss_sb_right: ', loss_sb_right)
        # print('loss_tb: ', loss_tb)
        # print('loss_int: ', loss_int)
        # print('new_loss: ', new_loss)
        # print()
        # print()
        # Compute boundary loss
        loss_u = loss_sb_left + loss_tb + self.apply_right_boundary_derivative(inp_train_sb_right)
        # if not no_right_boundary:
        #     loss_u += loss_sb_right
        # if new_loss is not None:
        #     loss_u += new_loss

        # Total loss with log scaling
        loss = torch.log10(self.lambda_u * loss_u + loss_int)

        if verbose:
            print(f'Total Loss: {loss.item():.4f} | '
                  f'Boundary Loss: {torch.log10(loss_u).item():.4f} | '
                  f'PDE Loss: {torch.log10(loss_int).item():.4f}')

        return loss, loss_u, loss_int

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list() # noqa
        training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int = self.assemble_datasets()
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print('################################ ', epoch, ' ################################')

            for j, ((inp_train_sb_left, u_train_sb_left), (inp_train_sb_right, u_train_sb_right),
                    (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) \
                    in enumerate(zip(training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    train_points = (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right, inp_train_tb, u_train_tb, inp_train_int)
                    loss, _, _ = self.compute_loss(train_points, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    def enhanced_fit(self, num_epochs, optimizer, config=None, verbose=True):
        if config is None: # noqa
            config = TrainingConfig(num_epochs=num_epochs)

        history = {
            'total_loss': [],
            'pde_loss': [],
            'boundary_loss': [],
            'learning_rate': []
        }

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )

        # Setup scheduler for ADAM
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        scheduler = None
        if not is_lbfgs:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.scheduler_min_lr,
                verbose=verbose
            )

        # Get training datasets
        training_sets = self.assemble_datasets()
        training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int = training_sets

        for epoch in range(num_epochs):
            epoch_losses = []

            if verbose and epoch % max(1, num_epochs // 10) == 0:
                print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            for data in zip(training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int):
                inp_train_sb_left, u_train_sb_left = data[0]
                inp_train_sb_right, u_train_sb_right = data[1]
                inp_train_tb, u_train_tb = data[2]
                inp_train_int, _ = data[3]

                # Enable gradients
                inp_train_sb_left.requires_grad_(True)
                inp_train_sb_right.requires_grad_(True)
                inp_train_tb.requires_grad_(True)
                inp_train_int.requires_grad_(True)

                def closure():
                    if is_lbfgs:
                        optimizer.zero_grad()

                    train_points = (
                        inp_train_sb_left, u_train_sb_left,
                        inp_train_sb_right, u_train_sb_right,
                        inp_train_tb, u_train_tb,
                        inp_train_int
                    )

                    loss, loss_u, loss_int = self.compute_loss(train_points, verbose=False)
                    loss.backward()

                    epoch_losses.append({
                        'total': loss.item(),
                        'pde': torch.log10(loss_int).item(),
                        'boundary': torch.log10(loss_u).item()
                    })

                    return loss

                if is_lbfgs:
                    optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    loss = closure()
                    optimizer.step()

            # Calculate average losses
            avg_losses = {
                k: np.mean([loss[k] for loss in epoch_losses])
                for k in ['total', 'pde', 'boundary']
            }

            # Update history
            history['total_loss'].append(avg_losses['total'])
            history['pde_loss'].append(avg_losses['pde'])
            history['boundary_loss'].append(avg_losses['boundary'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Update scheduler
            if scheduler is not None:
                scheduler.step(avg_losses['total'])

            # Print progress
            if verbose and epoch % max(1, num_epochs // 10) == 0:
                print(f"Total Loss: {avg_losses['total']:.6f} | "
                      f"Boundary Loss: {avg_losses['boundary']:.6f} | "
                      f"PDE Loss: {avg_losses['pde']:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6e}")

            # Early stopping
            if early_stopping(self.approximate_solution, avg_losses['total']):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                early_stopping.restore_best_weights(self.approximate_solution)
                break

        if verbose:
            print(f"\nTraining completed. Final loss: {history['total_loss'][-1]:.6f}")

        return history

    def relative_L2_error(self):
        path = r'/Users/omar/Desktop/PINN/exact_data.xlsx' # noqa
        data = pd.read_excel(path, header=None)
        x = data[0].values
        t = data[1].values
        inputs = torch.tensor(np.stack((t, x), axis=1))
        output = self.approximate_solution(inputs).reshape(-1, )
        # exact_output = self.exact_solution(inputs).reshape(-1, )
        exact_output = data[2].values.reshape(-1, )
        exact_output = torch.tensor(exact_output, dtype=output.dtype)

        err = (torch.mean((output.detach() - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5 * 100
        print("L2 Relative Error Norm: ", err.item(), "%")
        return inputs, output, exact_output

    def plotting(self):
        self.approximate_solution.eval()
        with torch.no_grad():
            inputs, output, exact_output = self.relative_L2_error()

            # Move data to CPU for plotting
            inputs = inputs.cpu()
            output = output.cpu()
            exact_output = exact_output.cpu()

            fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
            im1 = axs[0].scatter(inputs[:, 1], inputs[:, 0], c=exact_output, cmap='jet')
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('t')
            plt.colorbar(im1, ax=axs[0])
            axs[0].grid(True, which='both', ls=':')

            im2 = axs[1].scatter(inputs[:, 1], inputs[:, 0], c=output, cmap='jet')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('t')
            plt.colorbar(im2, ax=axs[1])
            axs[1].grid(True, which='both', ls=':')

            axs[0].set_title('Exact Solution')
            axs[1].set_title('Approximate Solution')
            plt.show()

    def plot_training_points(self):
        # Plot the input training points # noqa
        input_sb_left_, output_sb_left_ = self.add_spatial_boundary_points_left()
        input_sb_right_, output_sb_right_ = self.add_spatial_boundary_points_right()
        input_tb_, output_tb_ = self.add_temporal_boundary_points()
        input_int_, output_int_ = self.add_interior_points()

        plt.figure(figsize=(16, 8), dpi=150)
        # plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label='Boundary Points')
        plt.scatter(input_sb_left_[:, 1].detach().numpy(), input_sb_left_[:, 0].detach().numpy(), label='Left Boundary Points')
        plt.scatter(input_sb_right_[:, 1].detach().numpy(), input_sb_right_[:, 0].detach().numpy(), label='Right Boundary Points')
        plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label='Interior Points')
        plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label='Initial Points')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_loss(hist):
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_training_history(history):
        """Plot training history including losses and learning rate."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot losses
        ax1.plot(history['total_loss'], label='Total Loss')
        ax1.plot(history['pde_loss'], label='PDE Loss')
        ax1.plot(history['boundary_loss'], label='Boundary Loss')
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        ax2.plot(history['learning_rate'], label='Learning Rate')
        ax2.set_yscale('log')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def optimizer_LBFGS(self, config):
        """Create LBFGS optimizer with specified configuration."""
        return torch.optim.LBFGS(
            self.approximate_solution.parameters(),
            lr=float(config.lr),
            max_iter=config.max_iter,
            max_eval=config.max_eval,
            history_size=config.history_size,
            line_search_fn=config.line_search_fn
        )

    def optimizer_ADAM(self, lr=1e-5):
        return optim.Adam(self.approximate_solution.parameters(), lr=float(lr))


    def save_model(self, path):
        torch.save(self.approximate_solution.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path)
        self.approximate_solution.load_state_dict(state['model_state'])