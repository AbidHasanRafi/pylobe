"""Bayesian Optimization with Gaussian Process surrogate and EI acquisition."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from pylobe.optimization.base_optimizer import BaseOptimizer, OptimizationResult


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using GP surrogate + Expected Improvement.

    EI(x) = (μ(x) - f_best - ξ)·Φ(Z) + σ(x)·φ(Z)
    where Z = (μ(x) - f_best - ξ) / σ(x),
          Φ = standard normal CDF,
          φ = standard normal PDF,
          ξ = exploration parameter (default 0.01).

    Parameters
    ----------
    objective_fn : callable
    bounds : dict
    n_iterations : int
        Number of BO acquisitions.
    n_initial : int
        Number of random initial evaluations.
    xi : float
        Exploration-exploitation balance (ξ ≥ 0).
    seed : int or None
    """

    def __init__(self, objective_fn, bounds: dict,
                 n_iterations: int = 50, n_initial: int = 10,
                 xi: float = 0.01, seed: int = None):
        super().__init__(objective_fn, bounds,
                         n_iterations=n_iterations, population_size=n_initial,
                         seed=seed)
        self.n_initial = n_initial
        self.xi = xi
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

    # ------------------------------------------------------------------
    # Expected Improvement
    # ------------------------------------------------------------------
    def expected_improvement(self, X_candidate: np.ndarray,
                              gp: GaussianProcessRegressor,
                              f_best: float) -> np.ndarray:
        """Compute EI for candidate points.

        Parameters
        ----------
        X_candidate : ndarray, shape (N, D)
        gp : fitted GaussianProcessRegressor
        f_best : float

        Returns
        -------
        ndarray, shape (N,)
        """
        mu, sigma = gp.predict(X_candidate, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        Z   = (f_best - mu - self.xi) / sigma
        ei  = (f_best - mu - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei

    # ------------------------------------------------------------------
    # Next sample (maximise EI)
    # ------------------------------------------------------------------
    def next_sample(self, gp: GaussianProcessRegressor,
                    f_best: float) -> np.ndarray:
        """Find next sample by maximising EI via L-BFGS-B (multistart).

        Returns
        -------
        ndarray, shape (D,)
        """
        D  = len(self.param_names)
        best_ei  = -np.inf
        best_x   = None
        bounds_list = list(zip(self.lo, self.hi))

        for _ in range(15):
            x0 = self.rng.uniform(self.lo, self.hi)
            res = minimize(
                lambda x: -self.expected_improvement(x.reshape(1, -1), gp, f_best)[0],
                x0,
                bounds=bounds_list,
                method='L-BFGS-B',
            )
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x  = res.x

        return np.clip(best_x, self.lo, self.hi)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def optimize(self) -> OptimizationResult:
        """Run Bayesian Optimisation and return OptimizationResult."""
        D = len(self.param_names)

        # Initial random evaluations
        X_obs = self._random_population()[:self.n_initial]
        y_obs = np.array([self._evaluate(self._vec_to_dict(x)) for x in X_obs])

        best_idx = np.argmin(y_obs)
        best_fit = y_obs[best_idx]
        best_vec = X_obs[best_idx].copy()
        history  = []

        X_list = list(X_obs)
        y_list = list(y_obs)

        for itr in range(1, self.n_iterations + 1):
            # Fit GP
            X_arr = np.array(X_list)
            y_arr = np.array(y_list)
            self._gp.fit(X_arr, y_arr)

            # Propose next point
            x_next = self.next_sample(self._gp, best_fit)
            f_next = self._evaluate(self._vec_to_dict(x_next))

            X_list.append(x_next)
            y_list.append(f_next)

            if f_next < best_fit:
                best_fit = f_next
                best_vec = x_next.copy()

            history.append((itr, best_fit))

        return OptimizationResult(
            best_params=self._vec_to_dict(best_vec),
            best_fitness=best_fit,
            history=history,
            all_evaluations=self._eval_log,
        )
