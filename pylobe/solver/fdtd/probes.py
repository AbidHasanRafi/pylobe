"""FDTD field probes for time-domain and frequency-domain measurements."""
import numpy as np
from pylobe.utils.signal import fft_with_freq


class PointProbe:
    """Record a single E- or H-field component at a fixed grid point vs time.

    Parameters
    ----------
    i, j, k : int
        Grid cell indices.
    component : str
        Field component to record: 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'.
    """

    def __init__(self, i: int, j: int, k: int, component: str):
        self.i = i
        self.j = j
        self.k = k
        self.component = component
        self._data = []

    def record(self, grid, t_step: int):
        """Append current field value to the time series.

        Parameters
        ----------
        grid : FDTDGrid
        t_step : int
            Current time step index (for bookkeeping).
        """
        value = float(getattr(grid, self.component)[self.i, self.j, self.k])
        self._data.append(value)

    def time_series(self) -> np.ndarray:
        """Return recorded time series as ndarray."""
        return np.array(self._data)


class VoltageProbe:
    """Line integral of E-field along a path (port voltage).

    V = -∫ E · dl

    Integrates Ez along z from iz_lo to iz_hi at fixed (ix, iy).

    Parameters
    ----------
    ix, iy : int
        Fixed x, y cell indices.
    iz_lo, iz_hi : int
        Integration path in z.
    """

    def __init__(self, ix: int, iy: int, iz_lo: int, iz_hi: int):
        self.ix = ix
        self.iy = iy
        self.iz_lo = iz_lo
        self.iz_hi = iz_hi
        self._data = []

    def record(self, grid, t_step: int):
        """Record port voltage at current time step."""
        Ez_path = grid.Ez[self.ix, self.iy, self.iz_lo:self.iz_hi]
        V = -np.sum(Ez_path) * grid.dz
        self._data.append(float(V))

    def time_series(self) -> np.ndarray:
        return np.array(self._data)


# Alias for typo in build plan
VoltagePorbe = VoltageProbe


class SParameterProbe:
    """Extract S11 from FDTD time-domain signals via FFT.

    Algorithm:
    1. Run with excitation + record incident signal (probe before antenna)
    2. Run with PEC short-circuit + record reflected
    3. OR use de-embedding: separate incident and reflected by time-gating

    Simple single-probe approach: subtract known incident from total.

    Parameters
    ----------
    voltage_probe : VoltageProbe
        Records total (incident + reflected) voltage.
    incident_probe : VoltageProbe or None
        Records incident voltage (free-space run). If None, infer from
        time-gating.
    dt : float
        FDTD time step [s].
    Z0 : float
        Reference impedance [Ω].
    """

    def __init__(self, voltage_probe: VoltageProbe,
                 incident_probe: VoltageProbe = None,
                 dt: float = 1e-12, Z0: float = 50.0):
        self.probe = voltage_probe
        self.incident_probe = incident_probe
        self.dt = dt
        self.Z0 = Z0

    def compute_s11(self) -> tuple:
        """Compute S11 from recorded signals.

        If incident_probe is provided:
            S11(f) = V_ref(f) / V_inc(f)
            V_ref = V_total - V_inc

        Returns
        -------
        tuple (freq_array [Hz], S11_complex)
        """
        v_total = self.probe.time_series()
        if self.incident_probe is not None:
            v_inc = self.incident_probe.time_series()
            n = min(len(v_total), len(v_inc))
            v_ref = v_total[:n] - v_inc[:n]
            v_inc = v_inc[:n]
        else:
            # Simple approximation: first half = incident, second half = reflected
            n = len(v_total) // 2
            v_inc = v_total[:n]
            v_ref = v_total[n:]
            n = min(len(v_inc), len(v_ref))
            v_inc = v_inc[:n]
            v_ref = v_ref[:n]

        freq, V_inc_f = fft_with_freq(v_inc, self.dt)
        _,    V_ref_f = fft_with_freq(v_ref, self.dt)

        # Avoid division by near-zero
        mask = np.abs(V_inc_f) > 1e-30
        S11 = np.where(mask, V_ref_f / V_inc_f, complex(0))
        return freq, S11

    def vswr(self) -> tuple:
        """Compute VSWR from S11.

        VSWR = (1 + |S11|) / (1 - |S11|)

        Returns
        -------
        tuple (freq_array, vswr_array)
        """
        freq, S11 = self.compute_s11()
        mag = np.abs(S11)
        mag = np.clip(mag, 0.0, 0.999)
        return freq, (1.0 + mag) / (1.0 - mag)

    def return_loss_db(self) -> tuple:
        """Return loss = -20·log10(|S11|) [dB].

        Returns
        -------
        tuple (freq_array, RL_dB)
        """
        freq, S11 = self.compute_s11()
        rl = -20.0 * np.log10(np.clip(np.abs(S11), 1e-15, None))
        return freq, rl
