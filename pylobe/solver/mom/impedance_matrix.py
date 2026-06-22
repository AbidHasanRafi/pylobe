"""Impedance matrix elements for wire MoM solver.

Uses Gaussian quadrature for numerical integration.
Reference: Harrington, *Field Computation by Moment Methods*.
"""
import numpy as np
from scipy.integrate import quad
from pylobe.constants import PI, EPS0


def self_impedance(length: float, radius: float, k: float,
                   n_points: int = 20) -> complex:
    """Self-impedance of a wire segment using the reduced kernel.

    Avoids singularity by using:
        K_reduced = exp(-j·k·R_r) / R_r,   R_r = sqrt(z² + a²)

    Integrates numerically using Gaussian quadrature over [-Δ/2, +Δ/2].

    Parameters
    ----------
    length : float
        Segment length Δ [m].
    radius : float
        Wire radius a [m].
    k : float
        Wavenumber [rad/m].
    n_points : int
        Number of Gauss quadrature points.

    Returns
    -------
    complex
        Self-impedance Z_nn [Ω].
    """
    half = length / 2.0
    omega = k * 3e8 / (2 * PI)   # rough but not critical for self-term

    def kernel_re(z):
        R = np.sqrt(z**2 + radius**2)
        return np.cos(k * R) / R

    def kernel_im(z):
        R = np.sqrt(z**2 + radius**2)
        return -np.sin(k * R) / R

    re, _ = quad(kernel_re, -half, half, limit=50)
    im, _ = quad(kernel_im, -half, half, limit=50)
    G = (re + 1j * im) / (4.0 * PI)

    # Pocklington kernel factor: multiply by (k² + d²/dz²) integrated
    # For self term, simplified to:
    Z_self = 1j * k * ETA0 / (4.0 * PI) * (re + 1j * im) * length
    return complex(Z_self)


def mutual_impedance(zm: float, zn: float, delta: float,
                     radius: float, k: float) -> complex:
    """Mutual impedance between wire segments m and n.

    Uses the full free-space kernel:
        K = exp(-j·k·R) / (4·π·R),   R = sqrt((zm-z')² + a²)

    Parameters
    ----------
    zm : float
        Centre of observation segment m [m].
    zn : float
        Centre of source segment n [m].
    delta : float
        Segment length [m].
    radius : float
        Wire radius [m].
    k : float
        Wavenumber [rad/m].

    Returns
    -------
    complex
        Mutual impedance Z_mn [Ω].
    """
    from pylobe.constants import ETA0
    half = delta / 2.0

    def kernel_re(z_prime):
        dz = zm - z_prime
        R = np.sqrt(dz**2 + radius**2)
        return np.cos(k * R) / R

    def kernel_im(z_prime):
        dz = zm - z_prime
        R = np.sqrt(dz**2 + radius**2)
        return -np.sin(k * R) / R

    lo, hi = zn - half, zn + half
    re, _ = quad(kernel_re, lo, hi, limit=50)
    im, _ = quad(kernel_im, lo, hi, limit=50)

    Z_mn = 1j * k * ETA0 / (4.0 * PI) * (re + 1j * im) * delta
    return complex(Z_mn)


# Re-import ETA0 for use in impedance_matrix module functions
from pylobe.constants import ETA0
