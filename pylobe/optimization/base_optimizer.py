"""Abstract base class for all optimizers."""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes
    ----------
    best_params : dict
        Best parameter dictionary found.
    best_fitness : float
        Objective function value at best_params (minimised).
    history : list of (iteration, best_fitness)
        Best fitness per iteration.
    all_evaluations : list of (params_dict, fitness)
        All evaluated points.
    """
    best_params:      dict
    best_fitness:     float
    history:          List[Tuple] = field(default_factory=list)
    all_evaluations:  List[Tuple] = field(default_factory=list)

    def convergence_curve(self) -> np.ndarray:
        """Return best_fitness per iteration as a 1-D array."""
        if not self.history:
            return np.array([])
        _, fits = zip(*self.history)
        return np.array(fits)


class BaseOptimizer(ABC):
    """Abstract base for antenna design optimizers.

    Parameters
    ----------
    objective_fn : callable
        Function(params_dict) → float. Minimised.
    bounds : dict
        {'param_name': (min_val, max_val)}.
    n_iterations : int
        Maximum iterations / generations.
    population_size : int
        Number of candidates per iteration.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, objective_fn, bounds: dict,
                 n_iterations: int, population_size: int,
                 seed: int = None):
        self.objective_fn   = objective_fn
        self.bounds         = bounds
        self.n_iterations   = n_iterations
        self.population_size = population_size
        self.param_names    = list(bounds.keys())
        self.lo             = np.array([bounds[k][0] for k in self.param_names])
        self.hi             = np.array([bounds[k][1] for k in self.param_names])
        self.rng            = np.random.default_rng(seed)
        self._eval_log: List[Tuple] = []

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run the optimisation and return results."""
        ...

    def run(self) -> OptimizationResult:
        """Alias for optimize()."""
        return self.optimize()

    def _evaluate(self, params: dict) -> float:
        """Evaluate objective with bounds clipping and logging.

        Parameters
        ----------
        params : dict

        Returns
        -------
        float
        """
        # Clip to bounds
        clipped = {k: float(np.clip(params[k], self.bounds[k][0], self.bounds[k][1]))
                   for k in self.param_names}
        try:
            fitness = float(self.objective_fn(clipped))
        except Exception:
            fitness = np.inf
        self._eval_log.append((clipped, fitness))
        return fitness

    def _vec_to_dict(self, vec: np.ndarray) -> dict:
        """Convert parameter vector to dict."""
        return {k: float(v) for k, v in zip(self.param_names, vec)}

    def _dict_to_vec(self, d: dict) -> np.ndarray:
        """Convert parameter dict to vector."""
        return np.array([d[k] for k in self.param_names])

    def _random_population(self) -> np.ndarray:
        """Generate uniform random population within bounds.

        Returns ndarray, shape (population_size, n_params).
        """
        n = len(self.param_names)
        pop = self.rng.uniform(0, 1, (self.population_size, n))
        return self.lo + pop * (self.hi - self.lo)
