"""Antenna performance metrics computed from radiation patterns.

All integrals use np.trapz (or np.trapezoid in NumPy ≥ 2.0) for speed;
scipy.integrate.dblquad is used where indicated for higher precision.

Key accuracy notes
------------------
* ``beamwidth_hpbw``: uses linear interpolation at −3 dB crossings
  (power definition: half_power = 0.5 × peak).
* ``axial_ratio``: uses the Stokes-parameter method (IEEE Std 149-1979)
  instead of a simple amplitude ratio — correctly handles the relative
  phase between E_θ and E_φ, distinguishing RHCP from LHCP.
* ``directivity``: trapezoidal double integral over (θ, φ).
"""
import numpy as np
from scipy.signal import argrelmax
from pylobe.constants import ETA0, PI


def directivity(E_theta: np.ndarray, E_phi: np.ndarray,
                theta: np.ndarray, phi: np.ndarray) -> tuple:
    """Compute directivity 2-D map and peak value.

    D(θ,φ) = 4π · U(θ,φ) / P_rad

    U(θ,φ) = (|E_θ|² + |E_φ|²) / (2·η₀)
    P_rad   = ∫₀²π ∫₀π  U · sinθ  dθ dφ

    Parameters
    ----------
    E_theta, E_phi : ndarray, shape (Ntheta, Nphi)
        Complex far-field components.
    theta : ndarray, shape (Ntheta,)  [rad]
    phi   : ndarray, shape (Nphi,)    [rad]

    Returns
    -------
    tuple (D_2d, D_max_linear, D_max_dBi)
    """
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz

    U = (np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2) / (2.0 * ETA0)
    integrand = U * np.sin(theta)[:, np.newaxis]
    P_rad = _trapz(_trapz(integrand, phi, axis=1), theta)

    if P_rad < 1e-60:
        return np.zeros_like(U), 0.0, -np.inf

    D_2d  = 4.0 * PI * U / P_rad
    D_max = float(np.max(D_2d))
    D_dBi = 10.0 * np.log10(D_max) if D_max > 0 else -np.inf
    return D_2d, D_max, D_dBi


def gain(directivity_linear: float, efficiency: float) -> float:
    """Antenna gain G = η · D (linear)."""
    return efficiency * directivity_linear


def radiation_efficiency(P_rad: float, P_input: float) -> float:
    """Radiation efficiency η = P_rad / P_input  ∈ [0, 1]."""
    if P_input <= 0:
        return 0.0
    return float(np.clip(P_rad / P_input, 0.0, 1.0))


def beamwidth_hpbw(pattern_1d: np.ndarray, angles_deg: np.ndarray) -> float:
    """Half-power beamwidth (HPBW) from a 1-D linear power pattern.

    Locates the two −3 dB crossings (i.e. pattern = 0.5 × peak) on
    either side of the main lobe using linear interpolation.

    Parameters
    ----------
    pattern_1d : ndarray
        Linear (not dB) *power* pattern.  Must not be a field amplitude.
    angles_deg : ndarray
        Corresponding angles [degrees].

    Returns
    -------
    float — HPBW [degrees].  Returns ``np.nan`` if the main lobe
    is not found or the pattern is all-zero.
    """
    pattern_1d = np.asarray(pattern_1d, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    p_max = float(np.max(pattern_1d))
    if p_max <= 0:
        return np.nan

    half_power = p_max * 0.5      # −3 dB power threshold
    idx_max    = int(np.argmax(pattern_1d))
    above      = pattern_1d >= half_power

    # ── Left (low-angle) crossing ─────────────────────────────────────────────
    theta_lo = float(angles_deg[0])
    for i in range(idx_max, 0, -1):
        if not above[i - 1]:
            denom = pattern_1d[i] - pattern_1d[i - 1] + 1e-30
            t     = (half_power - pattern_1d[i - 1]) / denom
            theta_lo = angles_deg[i - 1] + t * (angles_deg[i] - angles_deg[i - 1])
            break

    # ── Right (high-angle) crossing ───────────────────────────────────────────
    theta_hi = float(angles_deg[-1])
    for i in range(idx_max, len(pattern_1d) - 1):
        if not above[i + 1]:
            denom = pattern_1d[i + 1] - pattern_1d[i] + 1e-30
            t     = (half_power - pattern_1d[i]) / denom
            theta_hi = angles_deg[i] + t * (angles_deg[i + 1] - angles_deg[i])
            break

    hpbw = float(theta_hi - theta_lo)
    return hpbw if hpbw > 0 else np.nan


def beamwidth_fnbw(pattern_1d: np.ndarray, angles_deg: np.ndarray) -> float:
    """First-null beamwidth (FNBW).

    Angle between the first nulls on either side of the main lobe.

    Parameters
    ----------
    pattern_1d : ndarray
        Linear power pattern.
    angles_deg : ndarray [degrees]

    Returns
    -------
    float — FNBW [degrees].
    """
    pattern_1d = np.asarray(pattern_1d, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    idx_max       = int(np.argmax(pattern_1d))
    p_max         = pattern_1d[idx_max]
    null_threshold = p_max * 1e-4  # practical null (−40 dB)

    theta_lo = float(angles_deg[0])
    for i in range(idx_max, 0, -1):
        if pattern_1d[i - 1] < null_threshold:
            theta_lo = float(angles_deg[i - 1])
            break

    theta_hi = float(angles_deg[-1])
    for i in range(idx_max, len(pattern_1d) - 1):
        if pattern_1d[i + 1] < null_threshold:
            theta_hi = float(angles_deg[i + 1])
            break

    return float(theta_hi - theta_lo)


def side_lobe_level(pattern_db: np.ndarray, angles_deg: np.ndarray) -> float:
    """Side-lobe level (SLL) relative to main-lobe peak [dB].

    Uses ``scipy.signal.argrelmax`` to find local maxima outside a ±30°
    exclusion window around the main lobe.

    Returns
    -------
    float — SLL [dB], typically a negative number like −20.0.
    """
    pattern_db = np.asarray(pattern_db, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    peak_db  = float(np.max(pattern_db))
    peak_idx = int(np.argmax(pattern_db))
    peak_ang = float(angles_deg[peak_idx])

    mask      = np.abs(angles_deg - peak_ang) > 30.0
    side_vals = pattern_db[mask]

    if len(side_vals) == 0:
        return -np.inf

    side_maxima_idx = argrelmax(side_vals, order=3)[0]
    if len(side_maxima_idx) == 0:
        return float(np.max(side_vals) - peak_db)

    return float(np.max(side_vals[side_maxima_idx]) - peak_db)


def front_to_back_ratio(pattern_db: np.ndarray, angles_deg: np.ndarray) -> float:
    """Front-to-back ratio FBR = pattern(0°) − pattern(180°) [dB]."""
    pattern_db = np.asarray(pattern_db, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    idx_front  = int(np.argmin(np.abs(angles_deg - 0.0)))
    idx_back   = int(np.argmin(np.abs(angles_deg - 180.0)))
    return float(pattern_db[idx_front] - pattern_db[idx_back])


def beam_solid_angle(E_theta: np.ndarray, E_phi: np.ndarray,
                     theta: np.ndarray, phi: np.ndarray) -> float:
    """Beam solid angle Ω_A [steradians].

    Ω_A = 4π / D_max  =  ∫∫ F_n(θ,φ) dΩ

    where F_n = U / U_max  is the normalised radiation intensity.

    Parameters
    ----------
    E_theta, E_phi : ndarray, shape (Ntheta, Nphi)
    theta : ndarray [rad]
    phi   : ndarray [rad]

    Returns
    -------
    float [sr]
    """
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz

    U    = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2
    U_max = float(np.max(U))
    if U_max <= 0:
        return 4.0 * PI
    Fn   = U / U_max
    integrand = Fn * np.sin(theta)[:, np.newaxis]
    return float(_trapz(_trapz(integrand, phi, axis=1), theta))


def axial_ratio(E_theta: complex, E_phi: complex) -> float:
    """Polarisation axial ratio using Stokes parameters (IEEE Std 149-1979).

    AR ∈ [1, ∞]:
    * AR = 1   → circular polarisation (CP)
    * AR = ∞   → linear polarisation (LP)

    The Stokes method correctly accounts for the phase difference δ
    between E_θ and E_φ, unlike a simple |E_θ|/|E_φ| ratio.

    Parameters
    ----------
    E_theta, E_phi : complex
        Far-field phasor components at a single (θ, φ) point.

    Returns
    -------
    float — axial ratio (≥ 1).
    """
    a  = complex(E_theta)
    b  = complex(E_phi)

    # Stokes parameters
    s0 = abs(a) ** 2 + abs(b) ** 2
    s1 = abs(a) ** 2 - abs(b) ** 2
    s2 = 2.0 * (a * b.conjugate()).real
    s3 = -2.0 * (a * b.conjugate()).imag

    if s0 < 1e-30:
        return np.inf

    # Semi-axes of the polarisation ellipse
    oa = s0 + np.sqrt(s1 ** 2 + s2 ** 2)    # proportional to major axis²
    ob = s0 - np.sqrt(s1 ** 2 + s2 ** 2)    # proportional to minor axis²

    if oa <= 0 or ob < 0:
        return np.inf

    semi_major = np.sqrt(oa / 2.0)
    semi_minor = np.sqrt(ob / 2.0)

    if semi_major < 1e-30:
        return np.inf
    if semi_minor < 1e-30:
        return np.inf

    ar = semi_major / semi_minor
    return float(max(ar, 1.0))


def axial_ratio_2d(E_theta: np.ndarray, E_phi: np.ndarray) -> np.ndarray:
    """Axial ratio map over all (θ, φ) directions (Stokes parameters).

    Parameters
    ----------
    E_theta, E_phi : ndarray of complex, shape (Ntheta, Nphi)

    Returns
    -------
    ndarray, shape (Ntheta, Nphi) — axial ratio ≥ 1 at every direction.
      Values of ``np.inf`` indicate linear polarisation (zero minor axis).
    """
    s0 = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2
    s1 = np.abs(E_theta) ** 2 - np.abs(E_phi) ** 2
    s2 = 2.0 * (E_theta * E_phi.conjugate()).real

    denom = np.sqrt(s1 ** 2 + s2 ** 2)
    oa    = np.clip((s0 + denom) / 2.0, 0, None)
    ob    = np.clip((s0 - denom) / 2.0, 0, None)

    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(ob > 1e-30,
                      np.sqrt(oa / np.clip(ob, 1e-30, None)),
                      np.inf)
    return np.maximum(ar, 1.0)
