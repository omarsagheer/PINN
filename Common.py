import os
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons,
                 regularization_param, regularization_exp, retrain_seed, device):
        super(NeuralNet, self).__init__()

        self.device = device
        self.dtype = torch.float64
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        # Move layers to a specified device
        self.input_layer = nn.Linear(self.input_dimension, self.neurons).to(self.dtype).to(device)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.neurons, self.neurons).to(self.dtype).to(device)
            for _ in range(n_hidden_layers - 1)
        ])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension).to(self.dtype).to(device)

        self.init_xavier()

    def forward(self, x):
        x = x.to(self.dtype).to(self.device)
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)
                # Move initialized weights to device
                m.weight.data = m.weight.data.to(self.dtype).to(self.device)
                m.bias.data = m.bias.data.to(self.dtype).to(self.device)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, model, loss):
        if self.best_loss is None:
            self.best_loss = loss
            # Store state on CPU to avoid memory issues
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        return self.should_stop

    def restore_best_weights(self, model):
        if self.best_state is not None:
            # Move state dict to model's device before loading
            device = next(model.parameters()).to(torch.float64).device
            state_dict = {k: v.to(torch.float64).to(device) for k, v in self.best_state.items()}
            model.load_state_dict(state_dict)


from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_epochs: int
    lambda_u: float = 10.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    validation_fraction: float = 0.1

    # LBFGS specific
    max_iter: int = 1000
    max_eval: int = None
    lr: float = 0.5
    history_size: int = 50
    line_search_fn: str = 'strong_wolfe'



