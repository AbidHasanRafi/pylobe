"""Surface and wire current distribution analysis."""
import numpy as np


def current_magnitude(current: np.ndarray) -> np.ndarray:
    """Compute |I| from complex current distribution.

    Parameters
    ----------
    current : ndarray of complex
        Complex current distribution [A].

    Returns
    -------
    ndarray of float
        Current magnitude |I| [A].
    """
    return np.abs(np.asarray(current, dtype=complex))


def current_phase_deg(current: np.ndarray) -> np.ndarray:
    """Phase of current distribution [degrees].

    Parameters
    ----------
    current : ndarray of complex

    Returns
    -------
    ndarray of float [degrees]
    """
    return np.angle(np.asarray(current, dtype=complex), deg=True)


def standing_wave_ratio_current(current: np.ndarray) -> float:
    """Ratio of maximum to minimum current magnitude (current SWR).

    Parameters
    ----------
    current : ndarray of complex

    Returns
    -------
    float
    """
    mag = current_magnitude(current)
    mag_min = mag[mag > 0].min() if np.any(mag > 0) else 1e-15
    return float(mag.max() / mag_min)


def current_to_power(current: np.ndarray, resistance: float) -> float:
    """Compute dissipated power from current distribution.

    P = Σ |I_n|² · R / 2   (phasor convention)

    Parameters
    ----------
    current : ndarray of complex
        Segment currents [A].
    resistance : float
        Segment resistance per unit length × length [Ω].

    Returns
    -------
    float [W]
    """
    return float(0.5 * np.sum(np.abs(current) ** 2) * resistance)
