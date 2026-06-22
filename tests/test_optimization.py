"""Tests for optimization algorithms."""
import numpy as np
import pytest


def _sphere(params: dict) -> float:
    """Positive sphere — minimum at x=y=0, value=0."""
    x, y = params['x'], params['y']
    return x**2 + y**2


BOUNDS = {'x': (-5.0, 5.0), 'y': (-5.0, 5.0)}


class TestGeneticAlgorithm:
    def test_returns_result(self):
        from pylobe.optimization.genetic import GeneticAlgorithm
        ga = GeneticAlgorithm(_sphere, BOUNDS, pop_size=20, n_generations=10)
        result = ga.run()
        assert result is not None

    def test_best_params_in_bounds(self):
        from pylobe.optimization.genetic import GeneticAlgorithm
        ga = GeneticAlgorithm(_sphere, BOUNDS, pop_size=20, n_generations=10)
        result = ga.run()
        for k, (lo, hi) in BOUNDS.items():
            assert lo <= result.best_params[k] <= hi

    def test_convergence_curve_length(self):
        from pylobe.optimization.genetic import GeneticAlgorithm
        ga = GeneticAlgorithm(_sphere, BOUNDS, pop_size=20, n_generations=15)
        result = ga.run()
        curve = result.convergence_curve()
        assert len(curve) == 15

    def test_convergence_non_decreasing(self):
        from pylobe.optimization.genetic import GeneticAlgorithm
        ga = GeneticAlgorithm(_sphere, BOUNDS, pop_size=30, n_generations=20)
        result = ga.run()
        curve = result.convergence_curve()
        # Minimiser: best fitness should be non-increasing
        assert np.all(np.diff(curve) <= 1e-10)

    def test_finds_near_optimum(self):
        from pylobe.optimization.genetic import GeneticAlgorithm
        ga = GeneticAlgorithm(_sphere, BOUNDS, pop_size=50, n_generations=50)
        result = ga.run()
        assert result.best_fitness < 1.0


class TestPSO:
    def test_returns_result(self):
        from pylobe.optimization.pso import ParticleSwarmOptimizer
        pso = ParticleSwarmOptimizer(_sphere, BOUNDS, n_particles=20, n_iterations=15)
        result = pso.run()
        assert result is not None

    def test_params_in_bounds(self):
        from pylobe.optimization.pso import ParticleSwarmOptimizer
        pso = ParticleSwarmOptimizer(_sphere, BOUNDS, n_particles=20, n_iterations=15)
        result = pso.run()
        for k, (lo, hi) in BOUNDS.items():
            assert lo <= result.best_params[k] <= hi

    def test_convergence_length(self):
        from pylobe.optimization.pso import ParticleSwarmOptimizer
        pso = ParticleSwarmOptimizer(_sphere, BOUNDS, n_particles=20, n_iterations=10)
        result = pso.run()
        assert len(result.convergence_curve()) == 10

    def test_finds_near_optimum(self):
        from pylobe.optimization.pso import ParticleSwarmOptimizer
        pso = ParticleSwarmOptimizer(_sphere, BOUNDS, n_particles=40, n_iterations=50)
        result = pso.run()
        assert result.best_fitness < 1.0


class TestDifferentialEvolution:
    def test_returns_result(self):
        from pylobe.optimization.differential_evolution import DifferentialEvolution
        de = DifferentialEvolution(_sphere, BOUNDS, pop_size=20, n_generations=15)
        result = de.run()
        assert result is not None

    def test_params_in_bounds(self):
        from pylobe.optimization.differential_evolution import DifferentialEvolution
        de = DifferentialEvolution(_sphere, BOUNDS, pop_size=20, n_generations=15)
        result = de.run()
        for k, (lo, hi) in BOUNDS.items():
            assert lo <= result.best_params[k] <= hi

    def test_finds_near_optimum(self):
        from pylobe.optimization.differential_evolution import DifferentialEvolution
        de = DifferentialEvolution(_sphere, BOUNDS, pop_size=30, n_generations=50)
        result = de.run()
        assert result.best_fitness < 1.0

    def test_history_stored(self):
        from pylobe.optimization.differential_evolution import DifferentialEvolution
        de = DifferentialEvolution(_sphere, BOUNDS, pop_size=10, n_generations=5)
        result = de.run()
        assert len(result.history) > 0


class TestBayesianOptimizer:
    def test_returns_result(self):
        from pylobe.optimization.bayesian import BayesianOptimizer
        bo = BayesianOptimizer(_sphere, BOUNDS, n_initial=5, n_iterations=8)
        result = bo.run()
        assert result is not None

    def test_params_in_bounds(self):
        from pylobe.optimization.bayesian import BayesianOptimizer
        bo = BayesianOptimizer(_sphere, BOUNDS, n_initial=5, n_iterations=8)
        result = bo.run()
        for k, (lo, hi) in BOUNDS.items():
            assert lo <= result.best_params[k] <= hi

    def test_all_evaluations_stored(self):
        from pylobe.optimization.bayesian import BayesianOptimizer
        n_init, n_iter = 5, 8
        bo = BayesianOptimizer(_sphere, BOUNDS, n_initial=n_init, n_iterations=n_iter)
        result = bo.run()
        assert len(result.all_evaluations) >= n_init + n_iter


class TestObjectiveFunctions:
    def test_maximize_gain_callable(self):
        from pylobe.optimization.objectives import maximize_gain
        def sim(params):
            return {'directivity_dbi': params.get('lf', 0.5) * 4.0}
        obj = maximize_gain(sim, metric='directivity_dbi')
        val = obj({'lf': 0.5})
        assert isinstance(val, float)
        assert val < 0   # negative (maximisation via minimisation)

    def test_multi_objective_weighted(self):
        from pylobe.optimization.objectives import multi_objective
        def sim(p):
            return {'gain': p['x'], 'sll': -p['x']}
        obj = multi_objective(sim, weights={'gain': 1.0, 'sll': -1.0})
        v1 = obj({'x': 2.0})
        v2 = obj({'x': 3.0})
        # Higher x → gain=x increases, sll=-x decreases (more negative)
        # weighted sum = x + (-1)*(-x) = 2x; negated for minimiser = -2x
        # So v2 < v1 (more negative)
        assert v2 < v1
