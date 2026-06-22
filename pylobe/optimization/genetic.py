"""Real-coded Genetic Algorithm with SBX crossover and polynomial mutation."""
import numpy as np
from pylobe.optimization.base_optimizer import BaseOptimizer, OptimizationResult


class GeneticAlgorithm(BaseOptimizer):
    """Real-coded Genetic Algorithm (GA).

    Operators:
    - Tournament selection (k=3)
    - Simulated Binary Crossover (SBX): η_c = 20
    - Polynomial mutation: η_m = 20
    - Elitism: top 10% survive unconditionally

    Parameters
    ----------
    objective_fn : callable
    bounds : dict
    n_generations : int
    pop_size : int
    crossover_prob : float
    mutation_prob : float
    eta_c : float
        SBX distribution index.
    eta_m : float
        Polynomial mutation index.
    seed : int or None
    """

    def __init__(self, objective_fn, bounds: dict,
                 n_generations: int = 100, pop_size: int = 50,
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1,
                 eta_c: float = 20.0, eta_m: float = 20.0,
                 seed: int = None):
        super().__init__(objective_fn, bounds,
                         n_iterations=n_generations, population_size=pop_size,
                         seed=seed)
        self.crossover_prob = crossover_prob
        self.mutation_prob  = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def optimize(self) -> OptimizationResult:
        """Run GA and return OptimizationResult."""
        pop = self._random_population()
        fitness = np.array([self._evaluate(self._vec_to_dict(ind)) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        best_vec = pop[best_idx].copy()
        history  = []

        n_elite = max(1, int(0.1 * self.population_size))

        for gen in range(1, self.n_iterations + 1):
            new_pop = []

            # Elitism: copy top n_elite individuals
            elite_idx = np.argsort(fitness)[:n_elite]
            new_pop.extend(pop[i].copy() for i in elite_idx)

            while len(new_pop) < self.population_size:
                # Tournament selection
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)

                # SBX crossover
                if self.rng.random() < self.crossover_prob:
                    c1, c2 = self.sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Polynomial mutation
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)

                new_pop.append(np.clip(c1, self.lo, self.hi))
                if len(new_pop) < self.population_size:
                    new_pop.append(np.clip(c2, self.lo, self.hi))

            pop = np.array(new_pop[:self.population_size])
            fitness = np.array([self._evaluate(self._vec_to_dict(ind)) for ind in pop])

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_vec = pop[gen_best_idx].copy()

            history.append((gen, best_fit))

        return OptimizationResult(
            best_params=self._vec_to_dict(best_vec),
            best_fitness=best_fit,
            history=history,
            all_evaluations=self._eval_log,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _tournament(self, pop: np.ndarray, fitness: np.ndarray,
                    k: int = 3) -> np.ndarray:
        """k-tournament selection."""
        indices = self.rng.integers(0, len(pop), k)
        winner  = indices[np.argmin(fitness[indices])]
        return pop[winner].copy()

    # ------------------------------------------------------------------
    # SBX crossover
    # ------------------------------------------------------------------
    def sbx_crossover(self, parent1: np.ndarray,
                      parent2: np.ndarray) -> tuple:
        """Simulated Binary Crossover (SBX).

        For each gene i:
            u ~ U(0,1)
            if u ≤ 0.5: β = (2u)^{1/(η+1)}
            else:       β = (1/(2(1-u)))^{1/(η+1)}
            child1[i] = 0.5·((1+β)·p1[i] + (1-β)·p2[i])
            child2[i] = 0.5·((1-β)·p1[i] + (1+β)·p2[i])
        """
        eta = self.eta_c
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            if abs(parent1[i] - parent2[i]) < 1e-14:
                continue
            u = self.rng.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
            child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i])
        return child1, child2

    # ------------------------------------------------------------------
    # Polynomial mutation
    # ------------------------------------------------------------------
    def polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        For each gene with probability mutation_prob:
            u ~ U(0,1)
            if u < 0.5: δ = (2u)^{1/(η+1)} - 1
            else:       δ = 1 - (2(1-u))^{1/(η+1)}
            gene += δ·(upper - lower)
        """
        eta = self.eta_m
        ind = individual.copy()
        for i in range(len(ind)):
            if self.rng.random() < self.mutation_prob:
                u = self.rng.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                ind[i] += delta * (self.hi[i] - self.lo[i])
        return ind
