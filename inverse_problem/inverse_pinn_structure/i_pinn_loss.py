from abc import ABC

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Common import EarlyStopping, TrainingConfig
from inverse_problem.inverse_pinn_structure.adding_i_pinn_points import AddingIPINNPoints


class IPINNLoss(AddingIPINNPoints, ABC):
    pass

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, train_points, verbose=True, new_loss=None, no_right_boundary=False):
        (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right,
         inp_train_tb, u_train_tb, inp_train_int) = train_points

        inp_train_meas, u_train_meas = self.get_measurement_data()
        u_pred_meas = self.approximate_solution(inp_train_meas)

        # Compute boundary predictions
        u_pred_sb_left = self.apply_boundary_conditions_left(inp_train_sb_left)
        u_pred_sb_right = self.apply_boundary_conditions_right(inp_train_sb_right)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        # Validate shapes
        assert (u_pred_sb_left.shape[1] == u_train_sb_left.shape[1])
        assert (u_pred_sb_right.shape[1] == u_train_sb_right.shape[1])
        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])
        assert (u_pred_meas.shape[1] == u_train_meas.shape[1])


        # Compute individual losses
        loss_int = self.ms(self.compute_pde_residual(inp_train_int))
        loss_sb_left = self.ms(u_train_sb_left - u_pred_sb_left)
        loss_sb_right = self.ms(u_train_sb_right - u_pred_sb_right)
        loss_tb = self.ms(u_train_tb - u_pred_tb)
        loss_meas = self.ms(u_train_meas - u_pred_meas)

        # Compute boundary loss
        loss_u = loss_sb_left + loss_tb + loss_meas
        if not no_right_boundary:
            loss_u += loss_sb_right
        if new_loss is not None:
            loss_u += new_loss

        # Total loss with log scaling
        loss = torch.log10(self.lambda_u * loss_u + loss_int)

        if verbose:
            print(f'Total Loss: {loss.item():.4f} | '
                  f'Boundary Loss: {torch.log10(loss_u).item():.4f} | '
                  f'PDE Loss: {torch.log10(loss_int).item():.4f} | '
                  f'Coefficient Loss: {torch.log10(loss_meas).item():.4f}')

        return loss, loss_u, loss_int, loss_meas

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
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
                    loss, _, _, _ = self.compute_loss(train_points, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    def enhanced_fit(self, num_epochs, optimizer, config=None, verbose=True):
        if config is None:
            config = TrainingConfig(num_epochs=num_epochs)

        history = {
            'total_loss': [],
            'pde_loss': [],
            'boundary_loss': [],
            'coefficient_loss': [],
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

                def closure():
                    if is_lbfgs:
                        optimizer.zero_grad()

                    train_points = (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right, inp_train_tb, u_train_tb, inp_train_int)

                    loss, loss_u, loss_int, loss_meas = self.compute_loss(train_points, verbose=False)
                    loss.backward()

                    epoch_losses.append({
                        'total': loss.item(),
                        'pde': torch.log10(loss_int).item(),
                        'boundary': torch.log10(loss_u).item(),
                        'coefficient': torch.log10(loss_meas).item()
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
                for k in ['total', 'pde', 'boundary', 'coefficient']
            }

            # Update history
            history['total_loss'].append(avg_losses['total'])
            history['pde_loss'].append(avg_losses['pde'])
            history['boundary_loss'].append(avg_losses['boundary'])
            history['coefficient_loss'].append(avg_losses['coefficient'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Update scheduler
            if scheduler is not None:
                scheduler.step(avg_losses['total'])

            # Print progress
            if verbose and epoch % max(1, num_epochs // 10) == 0:
                print(f"Total Loss: {avg_losses['total']:.6f} | "
                      f"Boundary Loss: {avg_losses['boundary']:.6f} | "
                      f"PDE Loss: {avg_losses['pde']:.6f} | "
                      f"Coefficient Loss: {avg_losses['coefficient']:.6f} | "
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