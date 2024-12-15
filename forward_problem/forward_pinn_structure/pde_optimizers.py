from abc import ABC

import torch

from forward_problem.forward_pinn_structure.main_pde import MainPDE


class PDEOptimizers(MainPDE, ABC):
    pass

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
        return torch.optim.Adam(self.approximate_solution.parameters(), lr=float(lr))


    def save_model(self, path):
        torch.save(self.approximate_solution.state_dict(), path)


    def load_model(self, path):
        state = torch.load(path)
        self.approximate_solution.load_state_dict(state['model_state'])