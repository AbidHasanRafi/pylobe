"""Inverse antenna design: find geometry from performance specifications."""
import numpy as np
from scipy.optimize import minimize


class InverseDesigner:
    """Inverse design via surrogate gradient descent.

    Given target antenna specifications, infers geometry parameters by
    minimising a loss function on the surrogate model's predictions.

    Supports two approaches:
    1. Surrogate-based gradient descent on input parameters.
    2. Random multistart search through surrogate.

    Parameters
    ----------
    surrogate : NeuralSurrogate
        Trained surrogate model.
    bounds : dict
        {'param_name': (min, max)} for each design variable.
    output_names : list of str
        Names of surrogate output quantities in order (e.g. ['S11', 'gain']).
    """

    def __init__(self, surrogate, bounds: dict,
                 output_names: list = None):
        self.surrogate    = surrogate
        self.bounds       = bounds
        self.param_names  = list(bounds.keys())
        self.lo = np.array([bounds[k][0] for k in self.param_names])
        self.hi = np.array([bounds[k][1] for k in self.param_names])
        self.output_names = output_names or [f'out_{i}' for i in range(surrogate.output_dim)]

    def design_from_spec(self, target_spec: dict,
                         n_restarts: int = 10,
                         weights: dict = None) -> dict:
        """Find geometry parameters matching target specifications.

        Parameters
        ----------
        target_spec : dict
            Target values for any subset of output_names, e.g.:
            {'gain_dbi': 8.0, 's11_db': -15.0}.
        n_restarts : int
            Number of L-BFGS-B random restarts.
        weights : dict or None
            Per-output weights for the loss. Default uniform.

        Returns
        -------
        dict
            Best geometry parameters found:
            {'param_name': value, ..., 'predicted_outputs': {...}, 'loss': float}
        """
        target_vec = np.array([
            target_spec.get(name, 0.0) for name in self.output_names
        ])
        w_vec = np.array([
            (weights or {}).get(name, 1.0) for name in self.output_names
        ])
        # Mask: only penalise specified targets
        mask = np.array([name in target_spec for name in self.output_names],
                        dtype=float)

        bounds_list = list(zip(self.lo, self.hi))

        def loss(x):
            x_arr = np.clip(x, self.lo, self.hi).reshape(1, -1)
            pred  = self.surrogate.predict(x_arr)[0]
            diff  = mask * w_vec * (pred - target_vec) ** 2
            return float(np.sum(diff))

        best_loss = np.inf
        best_x    = None

        rng = np.random.default_rng(42)
        for _ in range(n_restarts):
            x0  = rng.uniform(self.lo, self.hi)
            res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds_list,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                best_x    = np.clip(res.x, self.lo, self.hi)

        if best_x is None:
            best_x = rng.uniform(self.lo, self.hi)

        pred_best = self.surrogate.predict(best_x.reshape(1, -1))[0]
        result = {k: float(v) for k, v in zip(self.param_names, best_x)}
        result['predicted_outputs'] = {
            name: float(pred_best[i]) for i, name in enumerate(self.output_names)
        }
        result['loss'] = float(best_loss)
        return result
