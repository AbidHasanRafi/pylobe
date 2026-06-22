"""Near-field distribution computation and analysis."""
import numpy as np
from pylobe.constants import C0, PI, ETA0, MU0, EPS0


def compute_nearfield_dipole(current: np.ndarray, z_seg: np.ndarray,
                              freq: float, x_obs: np.ndarray,
                              y_obs: np.ndarray, z_obs: float = 0.0) -> dict:
    """Compute near-field of a wire dipole from MoM current distribution.

    Parameters
    ----------
    current : ndarray of complex, shape (N,)
        Segment currents [A].
    z_seg : ndarray, shape (N,)
        Segment centre z-positions [m].
    freq : float
        Frequency [Hz].
    x_obs, y_obs : ndarray, shape (Mx, My)
        Observation grid [m].
    z_obs : float
        Observation plane z-coordinate [m].

    Returns
    -------
    dict with keys 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', all complex ndarrays.
    """
    k = 2.0 * PI * freq / C0
    X, Y = np.asarray(x_obs, dtype=float), np.asarray(y_obs, dtype=float)

    Ez = np.zeros_like(X, dtype=complex)
    Hphi = np.zeros_like(X, dtype=complex)

    for n, (z_n, I_n) in enumerate(zip(z_seg, current)):
        dx = X
        dy = Y
        dz = z_obs - z_n
        R = np.sqrt(dx**2 + dy**2 + dz**2)
        R = np.where(R < 1e-10, 1e-10, R)

        # Green's function
        G = np.exp(-1j * k * R) / (4.0 * PI * R)

        # Hertzian dipole contribution
        dl = z_seg[1] - z_seg[0] if len(z_seg) > 1 else 1e-3
        jkR = 1j * k * R
        factor = G * (1.0 + 1.0/jkR - 1.0/jkR**2) * dl

        Ez += 1j * k * ETA0 * I_n * (
            (1.0 - dz**2 / R**2) * (1.0 + 1.0/jkR - 1.0/jkR**2)
        ) * G * dl

    Ex = np.zeros_like(Ez)
    Ey = np.zeros_like(Ez)
    Hx = np.zeros_like(Ez)
    Hy = Hphi
    Hz = np.zeros_like(Ez)

    return {'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
            'Hx': Hx, 'Hy': Hy, 'Hz': Hz}


def compute_nearfield_patch(patch, freq: float,
                             x_range: tuple, y_range: tuple,
                             z_obs: float, n_pts: int = 50) -> dict:
    """Approximate near-field of patch antenna (cavity model).

    Parameters
    ----------
    patch : RectangularPatch
    freq : float
    x_range, y_range : tuple (lo, hi) [m]
    z_obs : float
        Observation height above substrate [m].
    n_pts : int
        Grid resolution per axis.

    Returns
    -------
    dict with 'Ez', 'x_grid', 'y_grid'.
    """
    k0 = 2.0 * PI * freq / C0
    x  = np.linspace(*x_range, n_pts)
    y  = np.linspace(*y_range, n_pts)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Dominant TM010 mode: Ez ≈ E0 · cos(πx/L) inside patch footprint
    inside = (
        (X >= 0) & (X <= patch.W) &
        (Y >= 0) & (Y <= patch.L)
    )
    Ez = np.where(inside, np.cos(PI * Y / patch.L), 0.0) * (1.0 + 0j)

    return {'Ez': Ez, 'x_grid': X, 'y_grid': Y}
