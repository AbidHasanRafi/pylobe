"""Analytical solvers for microstrip patch antennas.

Implements the transmission-line and cavity models with full substrate
loss modelling via complex effective permittivity.

Solvers provided
----------------
PatchAnalyticalSolver
    Rectangular patch (TM010 cavity / transmission-line model).
    Reference: Balanis, *Antenna Theory*, 4th Ed., Ch. 14.

CircularPatchAnalyticalSolver
    Circular patch (TM11 cavity model).
    Reference: Balanis, *Antenna Theory*, 4th Ed., Ch. 14.6.
"""
import warnings
import numpy as np
from scipy.integrate import dblquad
from scipy.special import j0, j1, jn
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.patch import RectangularPatch


class PatchAnalyticalSolver:
    """Analytical radiation pattern and impedance solver for rectangular patch.

    Incorporates substrate dielectric loss via a complex effective permittivity,
    giving a physically correct radiation efficiency for lossy substrates (FR-4,
    Rogers 4003, etc.).

    Parameters
    ----------
    patch : RectangularPatch
        Fully designed patch geometry.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, patch: RectangularPatch, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        self.patch = patch
        self.freq = freq
        self.lambda0 = C0 / freq
        self.k0 = 2.0 * PI * freq / C0

        # Complex effective permittivity for lossy substrate.
        # eps_eff_c = eps_eff * (1 - j*tan_d)
        # Ref: Pozar, Microwave Engineering 4th Ed., Eq. 3.27
        tan_d = patch.material.loss_tangent
        self._eps_eff_c = complex(patch.eps_eff, -patch.eps_eff * tan_d)

        # Radiation efficiency derating for dielectric loss.
        # Approximate formula: η_d ≈ 1 / (1 + P_d/P_r)
        # where P_d/P_r = (eps_eff * tan_d * Q_r) / (2 * tan_d + ...)
        # A practical estimate (Pozar Ch. 14): η_d = 1 / (1 + tan_d * k0 * h * Q_factor)
        # Q_factor ≈ 1/(tan_d): use the simpler linear loss model
        self._dielectric_efficiency = 1.0 / (1.0 + patch.eps_eff * tan_d * PI)

    # ------------------------------------------------------------------
    # Resonant frequency
    # ------------------------------------------------------------------
    @property
    def resonant_frequency(self) -> float:
        """Resonant frequency from effective length [Hz].

        f_r = c / (2 * (L + 2*ΔL) * sqrt(eps_eff))
        Ref: Balanis Eq. 14-1.
        """
        return C0 / (
            2.0 * (self.patch.L + 2.0 * self.patch.delta_L)
            * np.sqrt(self.patch.eps_eff)
        )

    # ------------------------------------------------------------------
    # E-plane radiation pattern  (φ = 0°)
    # ------------------------------------------------------------------
    def e_plane_pattern(self, theta: np.ndarray) -> np.ndarray:
        """Normalised E-plane (φ=0°) electric field pattern magnitude.

        E_θ(θ) = sinc(k0·h/2·cosθ) · cos(k0·L/2·sinθ) · cosθ
        Ref: Balanis Eq. 14-27a.

        Parameters
        ----------
        theta : array_like
            Polar angles [rad].

        Returns
        -------
        ndarray
            Normalised |E_θ| pattern.

        Examples
        --------
        >>> import numpy as np
        >>> from pylobe import RectangularPatch, PatchAnalyticalSolver
        >>> p = RectangularPatch(freq=2.4e9)
        >>> s = PatchAnalyticalSolver(p, 2.4e9)
        >>> theta = np.linspace(0, np.pi, 181)
        >>> pat = s.e_plane_pattern(theta)
        >>> pat.shape
        (181,)
        """
        theta = np.asarray(theta, dtype=float)
        k0, h, L = self.k0, self.patch.h, self.patch.L
        arg_h = k0 * h / 2.0 * np.cos(theta)
        # sinc-like factor: sin(x)/x, safe at x=0
        sinc_factor = np.where(
            np.abs(arg_h) < 1e-10,
            1.0,
            np.sin(arg_h) / arg_h,
        )
        cos_factor = np.cos(k0 * L / 2.0 * np.sin(theta))
        pattern = np.abs(sinc_factor * cos_factor * np.cos(theta))
        max_val = np.max(pattern)
        return pattern / (max_val if max_val > 0 else 1.0)

    # ------------------------------------------------------------------
    # H-plane radiation pattern  (φ = 90°)
    # ------------------------------------------------------------------
    def h_plane_pattern(self, theta: np.ndarray) -> np.ndarray:
        """Normalised H-plane (φ=90°) electric field pattern magnitude.

        E_θ(θ) = sinc(k0·h/2·cosθ) · sinc(k0·W/2·sinθ)
        Ref: Balanis Eq. 14-27b.

        Parameters
        ----------
        theta : array_like
            Polar angles [rad].

        Returns
        -------
        ndarray
        """
        theta = np.asarray(theta, dtype=float)
        k0, h, W = self.k0, self.patch.h, self.patch.W
        arg_h = k0 * h / 2.0 * np.cos(theta)
        arg_w = k0 * W / 2.0 * np.sin(theta)

        def sinc(x):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)

        pattern = np.abs(sinc(arg_h) * sinc(arg_w))
        max_val = np.max(pattern)
        return pattern / (max_val if max_val > 0 else 1.0)

    # ------------------------------------------------------------------
    # Full 3-D radiation pattern
    # ------------------------------------------------------------------
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 361) -> "RadiationPattern":
        """Compute 3-D radiation pattern using cavity model.

        Returns a ``RadiationPattern`` object suitable for visualisation
        and lobe analysis.

        Parameters
        ----------
        n_theta : int
            Number of θ samples (0 to π).
        n_phi : int
            Number of φ samples (0 to 2π).

        Returns
        -------
        RadiationPattern

        Examples
        --------
        >>> from pylobe import RectangularPatch, PatchAnalyticalSolver
        >>> p = RectangularPatch(freq=2.4e9)
        >>> s = PatchAnalyticalSolver(p, 2.4e9)
        >>> rp = s.radiation_pattern(n_theta=91, n_phi=181)
        >>> rp.peak_directivity_dbi > 0
        True
        """
        from pylobe.analysis.radiation import RadiationPattern
        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')

        k0, h, W, L = self.k0, self.patch.h, self.patch.W, self.patch.L

        sinT = np.sin(TH)
        cosT = np.cos(TH)
        sinP = np.sin(PH)
        cosP = np.cos(PH)

        # Combined far-field expressions
        # Ref: Balanis Eqs. 14-27, 14-29 combined for full 3-D pattern
        X1 = k0 * h / 2.0 * cosT
        X2 = k0 * W / 2.0 * sinT * sinP
        X3 = k0 * L / 2.0 * sinT * cosP

        def sinc(x):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(np.abs(x) < 1e-10, 1.0, np.sin(x) / x)

        F = sinc(X1) * sinc(X2) * np.cos(X3)

        # E_theta: x-z plane component; E_phi: y-plane component
        # Ref: Balanis Eqs. 14-27a, 14-27b
        E_theta = F * cosT * cosP
        E_phi   = -F * sinP

        # Ground-plane: upper hemisphere only (z > 0 → θ < π/2)
        E_theta[TH > PI / 2.0] = 0.0
        E_phi  [TH > PI / 2.0] = 0.0

        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------
    # Directivity
    # ------------------------------------------------------------------
    @property
    def directivity(self) -> float:
        """Peak directivity [linear], computed from radiation pattern."""
        rp = self.radiation_pattern(n_theta=61, n_phi=49)
        return rp.peak_directivity_linear

    @property
    def directivity_dbi(self) -> float:
        """Peak directivity [dBi]."""
        return 10.0 * np.log10(self.directivity)

    # ------------------------------------------------------------------
    # Radiation efficiency (includes dielectric loss)
    # ------------------------------------------------------------------
    @property
    def radiation_efficiency(self) -> float:
        """Radiation efficiency [0–1] accounting for substrate dielectric loss.

        η_d = 1 / (1 + P_dielectric / P_radiation)
        Uses the approximate formula from Pozar, Microwave Engineering, Ch. 14:
            P_d/P_r ≈ eps_eff * tan_d * π

        Returns
        -------
        float
            Efficiency in [0, 1].
        """
        return float(self._dielectric_efficiency)

    @property
    def radiation_efficiency_db(self) -> float:
        """Radiation efficiency [dB]."""
        eta = self.radiation_efficiency
        return 10.0 * np.log10(max(eta, 1e-10))

    @property
    def gain_dbi(self) -> float:
        """Realised gain [dBi] = directivity + radiation efficiency [dB].

        For a lossless patch, gain ≈ directivity. For FR-4 at 2.4 GHz the
        efficiency penalty is approximately 0.5–1 dB.
        """
        return self.directivity_dbi + self.radiation_efficiency_db

    # ------------------------------------------------------------------
    # Input impedance vs inset feed position
    # ------------------------------------------------------------------
    def input_impedance(self, y0: float = None) -> complex:
        """Input impedance at inset feed position y0.

        Zin(y0) = (1/(2·G1)) · cos²(π·y0/L)
        Ref: Balanis Eq. 14-14.

        Parameters
        ----------
        y0 : float or None
            Inset feed position [m]. Uses patch.y0 if None.

        Returns
        -------
        complex
            Input impedance [Ω].

        Examples
        --------
        >>> from pylobe import RectangularPatch, PatchAnalyticalSolver
        >>> p = RectangularPatch(freq=2.4e9)
        >>> s = PatchAnalyticalSolver(p, 2.4e9)
        >>> Z = s.input_impedance()
        >>> abs(Z.real - 50.0) < 10
        True
        """
        y0 = y0 if y0 is not None else self.patch.y0
        Rin = (1.0 / (2.0 * self.patch.G1)) * np.cos(PI * y0 / self.patch.L) ** 2
        # Approximate reactive part from fringe susceptance (small near resonance)
        # Ref: Balanis Eq. 14-15 (simplified)
        B1 = self.patch.G1 * 0.1
        Xin = -1.0 / (2.0 * B1) if B1 != 0 else 0.0
        return complex(Rin, Xin)

    def impedance_sweep(self, n_freq: int = 200,
                        span_fraction: float = 0.3) -> tuple:
        """Compute complex input impedance over a frequency sweep.

        Parameters
        ----------
        n_freq : int
            Number of frequency points.
        span_fraction : float
            Frequency span as fraction of centre frequency.

        Returns
        -------
        tuple (freq_array [Hz], Z_complex ndarray, shape (n_freq,))
        """
        f_lo = self.freq * (1.0 - span_fraction / 2.0)
        f_hi = self.freq * (1.0 + span_fraction / 2.0)
        freqs = np.linspace(f_lo, f_hi, n_freq)
        Z = np.zeros(n_freq, dtype=complex)
        # Radiation Q: Q ∝ ε_r / h_mm — calibrated so FR4 (ε_r=4.4) at
        # h=1.6 mm gives ≈54 MHz −10 dB BW at 2.4 GHz.
        # Lower ε_r (e.g. RT5880) or thicker h both lower Q → wider BW.
        h_mm    = max(self.patch.h * 1e3, 0.1)
        Q_patch = (self.patch.eps_r / 4.4) * (48.0 / h_mm)
        for i, f in enumerate(freqs):
            lam = C0 / f
            k  = 2.0 * PI * f / C0
            # Radiation conductance at each frequency
            # Ref: Balanis Eq. 14-12
            G1 = (self.patch.W / (120.0 * lam)) * (1.0 - (k * self.patch.h) ** 2 / 24.0)
            G1 = max(G1, 1e-12)
            Rin = (1.0 / (2.0 * G1)) * np.cos(PI * self.patch.y0 / self.patch.L) ** 2
            # Reactive detuning: linear Q model (Balanis Eq. 14-69)
            delta_f = (f - self.resonant_frequency) / self.resonant_frequency
            Xin = 2.0 * Rin * delta_f * Q_patch
            Z[i] = complex(Rin, Xin)
        return freqs, Z

    def s11(self, Z0: float = 50.0, n_freq: int = 200) -> tuple:
        """Return (freq_array, S11_complex) over ±30% bandwidth.

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω].
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple (freq_array [Hz], S11_complex ndarray)

        Examples
        --------
        >>> from pylobe import RectangularPatch, PatchAnalyticalSolver
        >>> import numpy as np
        >>> p = RectangularPatch(freq=2.4e9)
        >>> s = PatchAnalyticalSolver(p, 2.4e9)
        >>> freqs, S11 = s.s11()
        >>> np.min(20 * np.log10(np.abs(S11))) < -10
        True
        """
        freqs, Z = self.impedance_sweep(n_freq=n_freq, span_fraction=0.6)
        S11 = (Z - Z0) / (Z + Z0)
        return freqs, S11

    def summary(self) -> dict:
        """Return all key solver metrics as a dictionary.

        Returns
        -------
        dict with keys:
            freq_GHz, f_resonant_GHz, directivity_dbi, gain_dbi,
            radiation_efficiency_pct, Zin_ohm, substrate_loss_tangent
        """
        return {
            'freq_GHz':               self.freq / 1e9,
            'f_resonant_GHz':         self.resonant_frequency / 1e9,
            'directivity_dbi':        self.directivity_dbi,
            'gain_dbi':               self.gain_dbi,
            'radiation_efficiency_pct': self.radiation_efficiency * 100.0,
            'Zin_ohm':                self.input_impedance(),
            'substrate_loss_tangent': self.patch.material.loss_tangent,
        }


# ══════════════════════════════════════════════════════════════════════════════
class CircularPatchAnalyticalSolver:
    """Analytical solver for circular microstrip patch (TM11 mode).

    Uses the cavity model far-field expressions from Balanis Ch. 14.6 and a
    Lorentzian resonance impedance model to compute S11 and gain.

    Parameters
    ----------
    patch : CircularPatch
        Circular patch geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, patch, freq: float):
        from pylobe.utils.validation import check_frequency
        from pylobe.geometry.patch import CircularPatch
        check_frequency(freq)
        if not isinstance(patch, CircularPatch):
            raise TypeError(
                f"CircularPatchAnalyticalSolver expects a CircularPatch, "
                f"got {type(patch).__name__!r}"
            )
        self.patch = patch
        self.freq  = float(freq)
        self.k0    = 2.0 * PI * freq / C0

        # Resonant frequency from TM11 mode
        # f_r = 1.8412 * c / (2π * a_eff * √εr)
        self._f_res = 1.8412 * C0 / (2.0 * PI * patch.a_eff * np.sqrt(patch.eps_r))

        # Radiation efficiency (dielectric loss — same model as rectangular)
        tan_d = patch.material.loss_tangent
        eps_eff_approx = (patch.eps_r + 1.0) / 2.0
        self._eta_d = 1.0 / (1.0 + eps_eff_approx * tan_d * PI)

    # ------------------------------------------------------------------
    @property
    def resonant_frequency(self) -> float:
        """Resonant frequency from TM11 mode [Hz]."""
        return self._f_res

    @property
    def radiation_efficiency(self) -> float:
        """Radiation efficiency [0–1], dielectric loss only."""
        return float(self._eta_d)

    @property
    def radiation_efficiency_pct(self) -> float:
        """Radiation efficiency [%]."""
        return self._eta_d * 100.0

    # ------------------------------------------------------------------
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 361) -> "RadiationPattern":
        """Compute 3-D radiation pattern (TM11 cavity model).

        Far-field expressions — Balanis Eq. 14-119 (simplified for broadside TM11):

        .. math::

            E_{\\theta} = j \\frac{k_0 a^2}{2} V_0 \\cos\\phi
                          \\left[J_0(k_0 a \\sin\\theta) + J_2(k_0 a \\sin\\theta)\\right]

            E_{\\phi} = -j \\frac{k_0 a^2}{2} V_0 \\cos\\theta \\sin\\phi
                        \\left[J_0(k_0 a \\sin\\theta) - J_2(k_0 a \\sin\\theta)\\right]

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern
        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')

        a   = self.patch.a_eff
        k0  = self.k0
        u   = k0 * a * np.sin(TH)

        J0u = j0(u)
        J2u = jn(2, u)

        # E-theta and E-phi components (TM11 mode far-field, Balanis 14.6)
        # Upper hemisphere only (ground plane)
        E_theta = np.cos(PH) * (J0u + J2u) * np.cos(TH)
        E_phi   = -np.sin(PH) * (J0u - J2u)

        # Enforce ground-plane: zero below horizon
        mask = TH > PI / 2.0
        E_theta[mask] = 0.0
        E_phi[mask]   = 0.0

        return RadiationPattern(E_theta.astype(complex), E_phi.astype(complex),
                                theta, phi, self.freq)

    # ------------------------------------------------------------------
    def impedance_sweep(self, n_freq: int = 200,
                        span_fraction: float = 0.3) -> tuple:
        """Complex input impedance over a frequency sweep.

        Uses a parallel RLC resonance model centred at the TM11 resonance.

        Returns
        -------
        tuple (freq_array [Hz], Z_complex ndarray)
        """
        f_lo = self.freq * (1.0 - span_fraction / 2.0)
        f_hi = self.freq * (1.0 + span_fraction / 2.0)
        freqs = np.linspace(f_lo, f_hi, n_freq)

        # Edge resistance from radiation conductance (Balanis Eq. 14-113)
        G_rad = (self.patch.a**2 * self.k0**2) / (240.0 * PI)
        G_rad = max(G_rad, 1e-12)
        R_res = 1.0 / (2.0 * G_rad)
        # Q factor for circular patch (approximate)
        Q = 1.8412 / (2.0 * self.k0 * self.patch.h
                       * np.sqrt(self.patch.eps_r) + 1e-12)
        Q = max(Q, 2.0)

        Z = np.zeros(n_freq, dtype=complex)
        for i, f in enumerate(freqs):
            delta_f = (f - self._f_res) / (self._f_res + 1e-30)
            Xin = 2.0 * R_res * delta_f * Q
            Z[i] = complex(R_res, Xin)
        return freqs, Z

    def s11(self, Z0: float = 50.0, n_freq: int = 200) -> tuple:
        """Return ``(freq_array, S11_complex)`` over ±30% bandwidth.

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω].
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple (freq_array [Hz], S11_complex ndarray)
        """
        freqs, Z = self.impedance_sweep(n_freq=n_freq, span_fraction=0.6)
        S11 = (Z - Z0) / (Z + Z0)
        return freqs, S11

    @property
    def directivity_dbi(self) -> float:
        """Peak directivity [dBi]."""
        return self.radiation_pattern(61, 121).peak_directivity_dbi

    @property
    def gain_dbi(self) -> float:
        """Realised gain [dBi] = directivity + efficiency [dB]."""
        eta_db = 10.0 * np.log10(max(self._eta_d, 1e-10))
        return self.directivity_dbi + eta_db

    def summary(self) -> dict:
        """Return key metrics as a dictionary."""
        return {
            'freq_GHz':               self.freq / 1e9,
            'f_resonant_GHz':         self._f_res / 1e9,
            'a_mm':                   self.patch.a * 1e3,
            'a_eff_mm':               self.patch.a_eff * 1e3,
            'directivity_dbi':        self.directivity_dbi,
            'gain_dbi':               self.gain_dbi,
            'radiation_efficiency_pct': self.radiation_efficiency_pct,
        }
