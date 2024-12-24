from abc import ABC

import torch

from inverse_problem.inverse_pinn_structure.base_i_pinn import BaseIPINN


class IPINNOptimizers(BaseIPINN, ABC):
    pass

    def optimizer_LBFGS(self, config):
        """Create LBFGS optimizer with specified configuration."""
        return torch.optim.LBFGS(
            list(self.approximate_solution.parameters())+list(self.approximate_coefficient.parameters()),
            lr=float(config.lr),
            max_iter=config.max_iter,
            max_eval=config.max_eval,
            history_size=config.history_size,
            line_search_fn=config.line_search_fn
        )

    def optimizer_ADAM(self, lr=1e-5):
        return torch.optim.Adam(list(self.approximate_solution.parameters())+list(self.approximate_coefficient.parameters()),
                                lr=float(lr))


    def save_model(self, path):
        torch.save(self.approximate_solution.state_dict(), path)


    def load_model(self, path):
        state = torch.load(path)
        self.approximate_solution.load_state_dict(state['model_state'])