"""Array factor computation for ULA, 2-D planar, and circular arrays."""
import numpy as np
from pylobe.constants import C0, PI


def array_factor_ula(theta: np.ndarray, freq: float, N: int,
                     d: float, beta: float = 0.0,
                     amplitudes: np.ndarray = None) -> np.ndarray:
    """Normalised array factor for a Uniform Linear Array.

    AF(θ) = Σ_{n=0}^{N-1} a_n · exp(j·n·ψ)
    where  ψ = k·d·cosθ + β

    Supports scalar, 1-D, and multi-dimensional theta via broadcasting.

    Parameters
    ----------
    theta : array_like
        Polar angles [rad].
    freq : float
        Frequency [Hz].
    N : int
        Number of elements.
    d : float
        Element spacing [m].
    beta : float
        Progressive phase shift [rad].
    amplitudes : ndarray of shape (N,) or None
        Element excitation amplitudes. Default: uniform.

    Returns
    -------
    ndarray
        Normalised |AF| in linear scale.
    """
    theta = np.asarray(theta, dtype=float)
    k = 2.0 * PI * freq / C0
    if amplitudes is None:
        amplitudes = np.ones(N)
    amplitudes = np.asarray(amplitudes, dtype=complex)

    psi = k * d * np.cos(theta)[..., np.newaxis] + beta   # (..., 1)
    n_arr = np.arange(N, dtype=float)
    AF = np.sum(amplitudes * np.exp(1j * n_arr * psi), axis=-1)
    max_v = np.max(np.abs(AF))
    return np.abs(AF) / (max_v if max_v > 0 else 1.0)


def array_factor_2d(theta: np.ndarray, phi: np.ndarray,
                    freq: float, M: int, N: int,
                    dx: float, dy: float,
                    beta_x: float = 0.0, beta_y: float = 0.0,
                    amplitudes: np.ndarray = None) -> np.ndarray:
    """Normalised array factor for a 2-D planar array.

    AF(θ,φ) = Σ_m Σ_n exp(j·(m·ψx + n·ψy))
    ψx = k·dx·sin(θ)·cos(φ) + β_x
    ψy = k·dy·sin(θ)·sin(φ) + β_y

    Parameters
    ----------
    theta : ndarray, shape (Nt,)
        Polar angles [rad].
    phi : ndarray, shape (Np,)
        Azimuthal angles [rad].
    freq : float
        Frequency [Hz].
    M, N : int
        Number of elements along x and y.
    dx, dy : float
        Element spacings [m].
    beta_x, beta_y : float
        Progressive phase shifts [rad].
    amplitudes : ndarray, shape (M, N) or None
        Element amplitude weights. Default: uniform.

    Returns
    -------
    ndarray, shape (Nt, Np)
        Normalised |AF| in linear scale.
    """
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi, dtype=float)
    k = 2.0 * PI * freq / C0

    TH, PH = np.meshgrid(theta, phi, indexing='ij')   # (Nt, Np)
    sinT = np.sin(TH)
    cosP = np.cos(PH)
    sinP = np.sin(PH)

    psi_x = k * dx * sinT * cosP + beta_x   # (Nt, Np)
    psi_y = k * dy * sinT * sinP + beta_y   # (Nt, Np)

    if amplitudes is None:
        amplitudes = np.ones((M, N))

    AF = np.zeros_like(TH, dtype=complex)
    for m in range(M):
        for n in range(N):
            AF += amplitudes[m, n] * np.exp(1j * (m * psi_x + n * psi_y))

    max_v = np.max(np.abs(AF))
    return np.abs(AF) / (max_v if max_v > 0 else 1.0)


def grating_lobe_condition(d: float, freq: float,
                            scan_angle_deg: float) -> bool:
    """Check if grating lobes exist at given spacing and scan angle.

    Grating lobes appear when:
        d/λ > 1 / (1 + |sin(θ_scan)|)

    Parameters
    ----------
    d : float
        Element spacing [m].
    freq : float
        Frequency [Hz].
    scan_angle_deg : float
        Scan angle from broadside [degrees].

    Returns
    -------
    bool
        True if grating lobes exist.
    """
    lam = C0 / freq
    theta_scan = np.deg2rad(scan_angle_deg)
    threshold = 1.0 / (1.0 + abs(np.sin(theta_scan)))
    return bool((d / lam) >= threshold)


def array_factor_circular(theta: np.ndarray, phi: np.ndarray,
                           freq: float, N: int, R: float,
                           alpha: np.ndarray = None,
                           amplitudes: np.ndarray = None) -> np.ndarray:
    """Normalised array factor for a uniform circular array.

    AF(θ,φ) = Σ_n a_n · exp(j·k·R·sinθ·cos(φ - 2πn/N) + j·α_n)

    Parameters
    ----------
    theta : ndarray, shape (Nt,)
    phi   : ndarray, shape (Np,)
    freq  : float
    N     : int
        Number of elements.
    R     : float
        Array radius [m].
    alpha : ndarray, shape (N,) or None
        Element phase excitations [rad].
    amplitudes : ndarray, shape (N,) or None
        Element amplitudes.

    Returns
    -------
    ndarray, shape (Nt, Np)
    """
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi,   dtype=float)
    k = 2.0 * PI * freq / C0
    alpha = np.zeros(N) if alpha is None else np.asarray(alpha)
    amplitudes = np.ones(N) if amplitudes is None else np.asarray(amplitudes)

    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    AF = np.zeros_like(TH, dtype=complex)
    for n in range(N):
        phi_n = 2.0 * PI * n / N
        AF += amplitudes[n] * np.exp(
            1j * (k * R * np.sin(TH) * np.cos(PH - phi_n) + alpha[n])
        )
    max_v = np.max(np.abs(AF))
    return np.abs(AF) / (max_v if max_v > 0 else 1.0)
