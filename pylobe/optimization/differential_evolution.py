"""Differential Evolution optimizer."""
import numpy as np
from pylobe.optimization.base_optimizer import BaseOptimizer, OptimizationResult


class DifferentialEvolution(BaseOptimizer):
    """Classic Differential Evolution (DE/rand/1/bin).

    Mutation:  v = x_r1 + F·(x_r2 - x_r3)
    Crossover: u_i = v_i if rand() < CR else x_i
    Selection: keep u if f(u) ≤ f(x)

    Parameters
    ----------
    objective_fn : callable
    bounds : dict
    n_generations : int
    pop_size : int
    F : float
        Mutation scale factor [0.4, 0.9] typical.
    CR : float
        Crossover probability [0, 1].
    seed : int or None
    """

    def __init__(self, objective_fn, bounds: dict,
                 n_generations: int = 100, pop_size: int = 40,
                 F: float = 0.5, CR: float = 0.7, seed: int = None):
        super().__init__(objective_fn, bounds,
                         n_iterations=n_generations, population_size=pop_size,
                         seed=seed)
        self.F  = F
        self.CR = CR

    def optimize(self) -> OptimizationResult:
        """Run DE and return OptimizationResult."""
        pop = self._random_population()
        fitness = np.array([self._evaluate(self._vec_to_dict(ind)) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        best_vec = pop[best_idx].copy()
        history  = []

        n = self.population_size
        D = len(self.param_names)

        for gen in range(1, self.n_iterations + 1):
            for i in range(n):
                # Select 3 distinct random indices ≠ i
                candidates = [j for j in range(n) if j != i]
                r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)

                # Mutation
                v = pop[r1] + self.F * (pop[r2] - pop[r3])
                v = np.clip(v, self.lo, self.hi)

                # Crossover (binomial)
                j_rand = self.rng.integers(0, D)
                mask   = self.rng.random(D) < self.CR
                mask[j_rand] = True
                u = np.where(mask, v, pop[i])

                # Selection
                fu = self._evaluate(self._vec_to_dict(u))
                if fu <= fitness[i]:
                    pop[i] = u
                    fitness[i] = fu
                    if fu < best_fit:
                        best_fit = fu
                        best_vec = u.copy()

            history.append((gen, best_fit))

        return OptimizationResult(
            best_params=self._vec_to_dict(best_vec),
            best_fitness=best_fit,
            history=history,
            all_evaluations=self._eval_log,
        )
