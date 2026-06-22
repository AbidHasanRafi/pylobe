"""Free-space Green's functions for Method of Moments."""
import numpy as np
from pylobe.constants import PI


def scalar_green(R: np.ndarray, k: float) -> np.ndarray:
    """Free-space scalar Green's function.

    G(R) = exp(-j·k·R) / (4·π·R)

    Singularity at R=0 is avoided by clipping to a small epsilon.

    Parameters
    ----------
    R : array_like
        Distance |r - r'| [m]. Must be non-negative.
    k : float
        Wavenumber [rad/m].

    Returns
    -------
    ndarray of complex128
        Green's function values.
    """
    R = np.asarray(R, dtype=float)
    R_safe = np.where(R < 1e-15, 1e-15, R)
    return np.exp(-1j * k * R_safe) / (4.0 * PI * R_safe)


def dyadic_green(r: np.ndarray, r_prime: np.ndarray,
                 k: float) -> np.ndarray:
    """Dyadic (tensor) Green's function G_ij(r, r').

    G_ij = (δ_ij + (1/k²)·∂²/∂i∂j) G_scalar(|r-r'|)

    Returns the 3×3 dyadic matrix. Diagonal approximation used near
    singularity (|r-r'| < ε).

    Parameters
    ----------
    r : ndarray, shape (..., 3)
        Observation point(s) [m].
    r_prime : ndarray, shape (..., 3)
        Source point(s) [m].
    k : float
        Wavenumber [rad/m].

    Returns
    -------
    ndarray of complex128, shape (..., 3, 3)
    """
    r = np.asarray(r, dtype=float)
    r_prime = np.asarray(r_prime, dtype=float)
    diff = r - r_prime
    R = np.linalg.norm(diff, axis=-1, keepdims=True)   # (..., 1)
    R_safe = np.where(R < 1e-15, 1e-15, R)

    G0 = np.exp(-1j * k * R_safe) / (4.0 * PI * R_safe)  # (..., 1)

    # Unit vector r_hat components
    r_hat = diff / R_safe   # (..., 3)

    # Pre-factors
    jkR = 1j * k * R_safe
    factor1 = (1.0 - 1.0 / jkR - 1.0 / jkR**2)
    factor2 = (-1.0 + 3.0 / jkR + 3.0 / jkR**2)

    # Outer product r_hat ⊗ r_hat: shape (..., 3, 3)
    rr = r_hat[..., :, np.newaxis] * r_hat[..., np.newaxis, :]

    # Identity tensor
    I = np.eye(3)
    while I.ndim < rr.ndim:
        I = I[np.newaxis]

    G_dyadic = G0[..., np.newaxis] * (factor1[..., np.newaxis] * I
                                       + factor2[..., np.newaxis] * rr)
    return G_dyadic
