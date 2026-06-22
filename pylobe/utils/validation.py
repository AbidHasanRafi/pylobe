"""Input validation helpers used by all geometry constructors and solvers."""
import warnings
import numpy as np


def check_positive(value, name: str) -> None:
    """Raise ValueError if value is not strictly positive.

    Parameters
    ----------
    value : float or array_like
    name : str
        Parameter name used in the error message.
    """
    if np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must be strictly positive, got {value!r}")


def check_nonnegative(value, name: str) -> None:
    """Raise ValueError if value is negative."""
    if np.any(np.asarray(value) < 0):
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def check_range(value, lo, hi, name: str) -> None:
    """Raise ValueError if value is outside [lo, hi]."""
    if np.any(np.asarray(value) < lo) or np.any(np.asarray(value) > hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")


def check_frequency(freq: float, name: str = "freq") -> None:
    """Validate that frequency is physically sensible.

    Parameters
    ----------
    freq : float
        Frequency [Hz].
    name : str
        Name used in messages.

    Raises
    ------
    ValueError
        If freq is not positive.
    ValueError
        If freq is above optical range (likely wrong units — GHz passed as Hz).
    """
    if freq <= 0:
        raise ValueError(f"{name} must be positive [Hz], got {freq!r}")
    if freq > 1e15:
        raise ValueError(
            f"{name} = {freq:.3e} Hz is above the optical range. "
            "Check units — PyLobe expects SI hertz (e.g. 2.4e9 for 2.4 GHz)."
        )
    if freq < 1e3:
        warnings.warn(
            f"{name} = {freq:.3e} Hz is very low (< 1 kHz). "
            "Check units — did you mean to pass GHz?",
            UserWarning, stacklevel=3,
        )


def check_eps_r(eps_r: float, name: str = "eps_r") -> None:
    """Validate relative permittivity.

    Parameters
    ----------
    eps_r : float
    name : str

    Raises
    ------
    ValueError
        If eps_r < 1 (must be a passive dielectric).
    """
    if eps_r < 1.0:
        raise ValueError(
            f"{name} must be >= 1.0 (passive dielectric medium), got {eps_r!r}"
        )


def check_substrate_thickness(h: float, freq: float, eps_r: float,
                               name: str = "h") -> None:
    """Warn if substrate is too thick relative to wavelength.

    A thick substrate (h > λ/10 in the medium) pushes the antenna into a
    dielectric waveguide regime where transmission-line models break down.

    Parameters
    ----------
    h : float
        Substrate thickness [m].
    freq : float
        Frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    name : str
    """
    from pylobe.constants import C0
    lambda_d = C0 / (freq * eps_r ** 0.5)
    ratio = h / lambda_d
    if ratio > 0.1:
        warnings.warn(
            f"{name} = {h*1e3:.2f} mm is thick relative to the guided "
            f"wavelength ({lambda_d*1e3:.2f} mm): h/λ_d = {ratio:.3f} > 0.1. "
            "Transmission-line model accuracy degrades for thick substrates.",
            UserWarning, stacklevel=3,
        )


def check_positive_integer(value, name: str) -> None:
    """Raise ValueError if value is not a positive integer."""
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def check_length_factor(lf: float, name: str = "length_factor") -> None:
    """Warn if dipole length factor is far from resonance.

    Parameters
    ----------
    lf : float
        Length as fraction of free-space wavelength.
    name : str
    """
    if lf <= 0:
        raise ValueError(f"{name} must be positive, got {lf!r}")
    if lf > 2.0:
        warnings.warn(
            f"{name} = {lf:.3f} (i.e. {lf:.3f}λ) is unusually long. "
            "Half-wave dipole analytical model is valid for 0.3 < lf < 0.7.",
            UserWarning, stacklevel=3,
        )


def ensure_array(x, dtype=float):
    """Convert scalar or list to 1-D NumPy array."""
    return np.atleast_1d(np.asarray(x, dtype=dtype))


def check_array_element_count(n: int, name: str = "n_elements") -> None:
    """Raise ValueError if array has fewer than 2 elements."""
    if n < 2:
        raise ValueError(
            f"{name} = {n}: an array requires at least 2 elements. "
            "Use a single-element geometry (dipole, patch, etc.) instead."
        )
