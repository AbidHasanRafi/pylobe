"""Particle Swarm Optimization with constriction coefficient."""
import numpy as np
from pylobe.optimization.base_optimizer import BaseOptimizer, OptimizationResult


class ParticleSwarmOptimizer(BaseOptimizer):
    """Standard PSO (Kennedy & Eberhart, 1995) with Clerc constriction.

    Velocity update:
        v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i) + c2·r2·(gbest - x_i)

    Position update:
        x_i(t+1) = x_i(t) + v_i(t+1)

    Constriction coefficient (Clerc & Kennedy, 2002):
        χ = 2 / |2 - φ - sqrt(φ²-4φ)|,  φ = c1+c2 > 4
        φ = 4.1 → χ ≈ 0.7298

    Parameters
    ----------
    objective_fn : callable
    bounds : dict
    n_iterations : int
    n_particles : int
    w : float
        Inertia weight (default = χ ≈ 0.7298).
    c1 : float
        Cognitive coefficient.
    c2 : float
        Social coefficient.
    seed : int or None
    """

    def __init__(self, objective_fn, bounds: dict,
                 n_iterations: int = 100, n_particles: int = 30,
                 w: float = 0.7298, c1: float = 2.05, c2: float = 2.05,
                 seed: int = None):
        super().__init__(objective_fn, bounds,
                         n_iterations=n_iterations, population_size=n_particles,
                         seed=seed)
        self.w  = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self) -> OptimizationResult:
        """Run PSO and return OptimizationResult."""
        n_dim = len(self.param_names)
        n_par = self.population_size

        # Initialise positions and velocities
        pos = self._random_population()                           # (N, D)
        vel = self.rng.uniform(-1, 1, (n_par, n_dim)) * (self.hi - self.lo) * 0.1

        pbest_pos = pos.copy()
        pbest_fit = np.array([self._evaluate(self._vec_to_dict(p)) for p in pos])

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        history   = []

        for itr in range(1, self.n_iterations + 1):
            r1 = self.rng.random((n_par, n_dim))
            r2 = self.rng.random((n_par, n_dim))

            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))

            pos = pos + vel
            pos = np.clip(pos, self.lo, self.hi)

            fit = np.array([self._evaluate(self._vec_to_dict(p)) for p in pos])

            # Update personal bests
            improved = fit < pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fit[improved]

            # Update global best
            best_idx = np.argmin(pbest_fit)
            if pbest_fit[best_idx] < gbest_fit:
                gbest_fit = pbest_fit[best_idx]
                gbest_pos = pbest_pos[best_idx].copy()

            history.append((itr, gbest_fit))

        return OptimizationResult(
            best_params=self._vec_to_dict(gbest_pos),
            best_fitness=gbest_fit,
            history=history,
            all_evaluations=self._eval_log,
        )
