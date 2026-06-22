"""FDTD excitation source functions and classes."""
import numpy as np
from pylobe.constants import PI


# ------------------------------------------------------------------
# Waveform functions
# ------------------------------------------------------------------

def gaussian_pulse(t: np.ndarray, t0: float, sigma: float) -> np.ndarray:
    """Gaussian pulse.

    s(t) = exp(-(t - t0)² / (2·σ²))

    Broadband spectrum; -3 dB bandwidth ≈ 1/(2π·σ).

    Parameters
    ----------
    t : array_like
        Time array [s].
    t0 : float
        Peak time [s].
    sigma : float
        Temporal width [s].

    Returns
    -------
    ndarray
    """
    return np.exp(-(np.asarray(t) - t0) ** 2 / (2.0 * sigma ** 2))


def modulated_gaussian(t: np.ndarray, f0: float, t0: float,
                       bandwidth: float) -> np.ndarray:
    """Gaussian-modulated sinusoid (Ricker wavelet variant).

    s(t) = exp(-(t-t0)²/(2σ²)) · cos(2πf0·(t-t0))
    σ    = 1/(π·bandwidth)

    Parameters
    ----------
    t : array_like
        Time array [s].
    f0 : float
        Centre frequency [Hz].
    t0 : float
        Peak time [s].
    bandwidth : float
        Gaussian bandwidth (-3 dB) [Hz].

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    sigma = 1.0 / (PI * bandwidth)
    env = np.exp(-(t - t0) ** 2 / (2.0 * sigma ** 2))
    return env * np.cos(2.0 * PI * f0 * (t - t0))


def sinusoidal_cw(t: np.ndarray, f0: float, ramp_cycles: int = 5) -> np.ndarray:
    """CW sinusoidal source with smooth Hann ramp-up.

    The first ramp_cycles cycles are windowed by a Hann envelope to
    avoid startup transients.

    Parameters
    ----------
    t : array_like
        Time array [s].
    f0 : float
        Frequency [Hz].
    ramp_cycles : int
        Number of cycles over which amplitude ramps to 1.

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    T_ramp = ramp_cycles / f0
    ramp = np.where(
        t < T_ramp,
        0.5 * (1.0 - np.cos(PI * t / T_ramp)),  # Hann window on [0, T_ramp]
        1.0
    )
    return ramp * np.sin(2.0 * PI * f0 * t)


# ------------------------------------------------------------------
# Source classes
# ------------------------------------------------------------------

class HardSource:
    """Hard point source: overwrites E-field component at a grid point.

    Use total-field formulation only. Note that hard sources prevent
    backward waves from passing through — use SoftSource for bidirectional.

    Parameters
    ----------
    i, j, k : int
        Grid cell indices.
    component : str
        Field component: 'Ex', 'Ey', or 'Ez'.
    waveform : callable
        waveform(t) → float, the source time function.
    """

    def __init__(self, i: int, j: int, k: int, component: str,
                 waveform):
        self.i = i
        self.j = j
        self.k = k
        if component not in ('Ex', 'Ey', 'Ez'):
            raise ValueError(f"component must be Ex/Ey/Ez, got {component}")
        self.component = component
        self.waveform = waveform

    def inject(self, grid, t: float):
        """Overwrite field at source point.

        Parameters
        ----------
        grid : FDTDGrid
        t : float
            Current simulation time [s].
        """
        value = self.waveform(t)
        getattr(grid, self.component)[self.i, self.j, self.k] = value


class SoftSource:
    """Soft (additive) point source: adds to E-field.

    Backward-travelling waves pass through without reflection.

    Parameters
    ----------
    i, j, k : int
        Grid cell indices.
    component : str
        Field component: 'Ex', 'Ey', or 'Ez'.
    waveform : callable
        waveform(t) → float.
    """

    def __init__(self, i: int, j: int, k: int, component: str,
                 waveform):
        self.i = i
        self.j = j
        self.k = k
        if component not in ('Ex', 'Ey', 'Ez'):
            raise ValueError(f"component must be Ex/Ey/Ez, got {component}")
        self.component = component
        self.waveform = waveform

    def inject(self, grid, t: float):
        """Add source to field at source point.

        Parameters
        ----------
        grid : FDTDGrid
        t : float
            Current simulation time [s].
        """
        value = self.waveform(t)
        getattr(grid, self.component)[self.i, self.j, self.k] += value


class TFSFBoundary:
    """Total-Field / Scattered-Field (TF/SF) plane wave injection.

    Connects total-field interior to scattered-field exterior via
    correction terms applied to E and H on the TF/SF boundary surface.

    Only z-directed plane wave (θ=0) supported in this implementation.

    Parameters
    ----------
    grid : FDTDGrid
    tfsf_lo : tuple of int (ix_lo, iy_lo, iz_lo)
        Lower corner of TF region in cell indices.
    tfsf_hi : tuple of int (ix_hi, iy_hi, iz_hi)
        Upper corner of TF region in cell indices.
    waveform : callable
        waveform(t) → float, incident field time function.
    """

    def __init__(self, grid, tfsf_lo: tuple, tfsf_hi: tuple,
                 waveform):
        self.grid = grid
        self.lo = tfsf_lo
        self.hi = tfsf_hi
        self.waveform = waveform
        # 1-D incident field buffer (along z)
        Nz_buf = grid.Nz + 20
        self.E_inc = np.zeros(Nz_buf)
        self.H_inc = np.zeros(Nz_buf)

    def update_incident(self, dt: float, t: float):
        """Advance 1-D incident field for TF/SF correction."""
        from pylobe.constants import C0, EPS0, MU0
        dz = self.grid.dz
        c = C0
        Nz = len(self.E_inc) - 1
        H_old = self.H_inc.copy()
        # 1-D FDTD update
        self.H_inc[:-1] -= (dt / (MU0 * dz)) * (self.E_inc[1:] - self.E_inc[:-1])
        self.E_inc[1:] -= (dt / (EPS0 * dz)) * (self.H_inc[1:] - H_old[:-1])
        # Inject at z=0
        self.E_inc[0] += self.waveform(t)

    def apply_corrections(self, dt: float, t: float):
        """Apply TF/SF correction to boundary H and E fields."""
        self.update_incident(dt, t)
        g = self.grid
        ix_lo, iy_lo, iz_lo = self.lo
        ix_hi, iy_hi, iz_hi = self.hi

        # H correction on z-low face: Hy at iz_lo-1
        g.Hy[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo - 1] -= (
            dt / (MU0 * g.dz) * self.E_inc[iz_lo]
        )
        # H correction on z-high face: Hy at iz_hi
        g.Hy[ix_lo:ix_hi, iy_lo:iy_hi, iz_hi] += (
            dt / (MU0 * g.dz) * self.E_inc[iz_hi]
        )
