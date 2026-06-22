"""Sensitivity analysis via central finite differences."""
import numpy as np
from typing import Callable


class SensitivityAnalyzer:
    """Compute partial derivatives of antenna performance metrics.

    Uses central finite differences (2nd-order accurate):
        ∂f/∂p ≈ (f(p+Δp) - f(p-Δp)) / (2·Δp)

    Parameters
    ----------
    geometry_factory : callable
        Function(params_dict) → AntennaGeometry.
    solver : callable
        Function(geometry, freq) → RadiationPattern.
    metric_fn : callable
        Function(RadiationPattern) → float.
    parameters : dict
        {'param_name': nominal_value}.
    delta : float
        Relative step size for finite difference (default 0.01 = 1%).
    """

    def __init__(self, geometry_factory: Callable,
                 solver: Callable,
                 metric_fn: Callable,
                 parameters: dict,
                 delta: float = 0.01):
        self.factory    = geometry_factory
        self.solver     = solver
        self.metric     = metric_fn
        self.parameters = parameters
        self.delta      = delta
        self._gradient_cache = {}

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------
    def gradient(self, freq: float) -> dict:
        """Compute gradient of metric w.r.t. all parameters.

        Central difference: ∂f/∂p ≈ (f(p+Δp) - f(p-Δp)) / (2·Δp)

        Parameters
        ----------
        freq : float
            Evaluation frequency [Hz].

        Returns
        -------
        dict {'param_name': gradient_value}
        """
        grad = {}
        for name, p0 in self.parameters.items():
            dp = abs(p0) * self.delta if abs(p0) > 1e-30 else self.delta

            params_hi = {**self.parameters, name: p0 + dp}
            params_lo = {**self.parameters, name: p0 - dp}

            f_hi = self._evaluate(params_hi, freq)
            f_lo = self._evaluate(params_lo, freq)

            grad[name] = (f_hi - f_lo) / (2.0 * dp)
        self._gradient_cache[freq] = grad
        return grad

    def _evaluate(self, params: dict, freq: float) -> float:
        """Evaluate metric at given parameter point."""
        try:
            geom    = self.factory(params)
            pattern = self.solver(geom, freq)
            return float(self.metric(pattern))
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Multi-frequency sensitivity matrix
    # ------------------------------------------------------------------
    def sensitivity_matrix(self, freq_array: np.ndarray) -> np.ndarray:
        """Compute sensitivity matrix over a frequency array.

        Parameters
        ----------
        freq_array : ndarray, shape (Nf,)
            Evaluation frequencies [Hz].

        Returns
        -------
        ndarray, shape (Nf, N_params)
        """
        param_names = list(self.parameters.keys())
        mat = np.zeros((len(freq_array), len(param_names)))
        for i, f in enumerate(freq_array):
            grad = self.gradient(f)
            for j, name in enumerate(param_names):
                mat[i, j] = grad.get(name, np.nan)
        return mat

    def most_sensitive_parameters(self, top_n: int = 3,
                                   freq: float = None) -> list:
        """Rank parameters by |∂metric/∂p| · |p| (relative sensitivity).

        Parameters
        ----------
        top_n : int
            Number of top parameters to return.
        freq : float or None
            Frequency to evaluate at. Uses first cached if None.

        Returns
        -------
        list of (param_name, relative_sensitivity)
        """
        if freq is not None:
            grad = self.gradient(freq)
        elif self._gradient_cache:
            grad = list(self._gradient_cache.values())[-1]
        else:
            raise ValueError("Call gradient(freq) first.")

        rel_sens = {
            name: abs(g) * abs(self.parameters[name])
            for name, g in grad.items()
        }
        sorted_params = sorted(rel_sens.items(), key=lambda x: x[1], reverse=True)
        return sorted_params[:top_n]

    def response_surface(self, param1: str, param2: str,
                         freq: float, n_points: int = 20) -> tuple:
        """2-D response surface: vary two parameters on a grid.

        Parameters
        ----------
        param1, param2 : str
            Parameter names (must be in self.parameters).
        freq : float
            Evaluation frequency [Hz].
        n_points : int
            Grid points per axis.

        Returns
        -------
        tuple (values_p1, values_p2, metric_2d)
            Arrays of shape (n_points,), (n_points,), (n_points, n_points).
        """
        p1_nominal = self.parameters[param1]
        p2_nominal = self.parameters[param2]

        p1_arr = np.linspace(p1_nominal * 0.7, p1_nominal * 1.3, n_points)
        p2_arr = np.linspace(p2_nominal * 0.7, p2_nominal * 1.3, n_points)

        metric_grid = np.zeros((n_points, n_points))
        for i, v1 in enumerate(p1_arr):
            for j, v2 in enumerate(p2_arr):
                params = {**self.parameters, param1: v1, param2: v2}
                metric_grid[i, j] = self._evaluate(params, freq)

        return p1_arr, p2_arr, metric_grid
