"""Physical and electromagnetic constants for PyLobe."""
from scipy import constants
import numpy as np

C0   = constants.c          # Speed of light in vacuum [m/s]
MU0  = constants.mu_0       # Permeability of free space [H/m]
EPS0 = constants.epsilon_0  # Permittivity of free space [F/m]
ETA0 = np.sqrt(MU0 / EPS0)  # Intrinsic impedance of free space [Ω] ≈ 376.73
PI   = constants.pi


def eta(eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Intrinsic impedance of a medium [Ω].

    Parameters
    ----------
    eps_r : float
        Relative permittivity (must be > 0).
    mu_r : float
        Relative permeability (must be > 0).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If eps_r or mu_r are not strictly positive.

    Examples
    --------
    >>> eta()          # free space
    376.73...
    >>> eta(eps_r=4.4) # FR-4 substrate
    179.6...
    """
    if eps_r <= 0:
        raise ValueError(f"eps_r must be positive (passive medium), got {eps_r}")
    if mu_r <= 0:
        raise ValueError(f"mu_r must be positive (passive medium), got {mu_r}")
    return ETA0 * (mu_r / eps_r) ** 0.5


def wavenumber(freq: float, eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Free-space wavenumber k = 2πf√(με) [rad/m].

    Parameters
    ----------
    freq : float
        Frequency [Hz] (must be positive).
    eps_r : float
        Relative permittivity (must be > 0).
    mu_r : float
        Relative permeability (must be > 0).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If freq, eps_r, or mu_r are not strictly positive.

    Examples
    --------
    >>> wavenumber(2.4e9)          # free space at 2.4 GHz
    50.27...
    """
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    if eps_r <= 0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    if mu_r <= 0:
        raise ValueError(f"mu_r must be positive, got {mu_r}")
    return 2.0 * PI * freq * np.sqrt(MU0 * mu_r * EPS0 * eps_r)


def wavelength(freq: float, eps_r: float = 1.0) -> float:
    """Wavelength in a medium λ = c / (f√εr) [m].

    Parameters
    ----------
    freq : float
        Frequency [Hz] (must be positive).
    eps_r : float
        Relative permittivity (must be > 0).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If freq or eps_r are not strictly positive.

    Examples
    --------
    >>> wavelength(2.4e9)           # free space at 2.4 GHz
    0.1249...
    >>> wavelength(2.4e9, eps_r=4.4) # in FR-4
    0.0595...
    """
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    if eps_r <= 0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    return C0 / (freq * eps_r ** 0.5)
