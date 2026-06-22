"""Objective functions for antenna optimisation."""
import numpy as np


def maximize_gain(sim_fn, metric: str = 'directivity_dbi'):
    """Factory: maximise a gain metric returned by sim_fn.

    Parameters
    ----------
    sim_fn : callable
        Function(params_dict) → dict of metrics.
    metric : str
        Key in the result dict to maximise.

    Returns
    -------
    callable
        f(params_dict) → float  (negative for minimisation).
    """
    def objective(params):
        result = sim_fn(params)
        return -float(result.get(metric, 0.0))
    return objective


def minimize_sll(sim_fn, metric: str = 'sll_db', target_sll_db: float = -20.0):
    """Factory: penalise SLL above target.

    Parameters
    ----------
    sim_fn : callable
        Function(params_dict) → dict of metrics.
    metric : str
        Key for the SLL value in the result dict.
    target_sll_db : float
        Target SLL [dB] (negative, e.g. -20).

    Returns
    -------
    callable
    """
    def objective(params):
        result = sim_fn(params)
        sll = float(result.get(metric, 0.0))
        return max(0.0, sll - target_sll_db)
    return objective


def minimize_s11(freq_array: np.ndarray, s11_db: np.ndarray,
                 band: tuple, threshold: float = -10.0) -> float:
    """Maximise bandwidth below S11 threshold in a frequency band.

    Returns the fraction of the band above threshold (minimise → wider BW).

    Parameters
    ----------
    freq_array : ndarray [Hz]
    s11_db : ndarray [dB]
    band : tuple (f_lo, f_hi) [Hz]
    threshold : float
        S11 threshold [dB]. Default -10.

    Returns
    -------
    float
        Fraction of band where S11 > threshold [0, 1].
    """
    mask_band = (freq_array >= band[0]) & (freq_array <= band[1])
    if not mask_band.any():
        return 1.0
    s11_in_band = s11_db[mask_band]
    fraction_above = float(np.mean(s11_in_band > threshold))
    return fraction_above


def multi_objective(sim_fn, weights: dict = None):
    """Weighted-sum multi-objective factory.

    Parameters
    ----------
    sim_fn : callable
        Function(params_dict) → dict of metric values.
    weights : dict or None
        {'metric_key': weight, ...}. Positive weights maximise, negative minimise.

    Returns
    -------
    callable
        f(params_dict) → float  (for minimisation).

    Example
    -------
    >>> def sim(p):
    ...     return {'gain': p['x'], 'sll': -p['x']}
    >>> obj = multi_objective(sim, weights={'gain': 1.0, 'sll': -1.0})
    """
    if weights is None:
        weights = {'gain': 1.0}

    def objective(params):
        result = sim_fn(params)
        score = 0.0
        for key, w in weights.items():
            score += w * float(result.get(key, 0.0))
        return -score   # negate so minimiser maximises the weighted sum

    return objective


def efficiency_gain_product(radiation_pattern, input_power: float) -> float:
    """Realised gain = radiation efficiency × directivity.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    input_power : float
        Input power [W].

    Returns
    -------
    float
        -realised_gain_dBi (for minimisation).
    """
    from pylobe.analysis.metrics import radiation_efficiency
    P_rad = radiation_pattern.P_rad
    eta   = radiation_efficiency(P_rad, input_power)
    G     = eta * radiation_pattern.peak_directivity_linear
    return -10.0 * np.log10(G) if G > 0 else np.inf
