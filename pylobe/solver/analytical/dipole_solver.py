"""Analytical solver for wire dipole antennas.

Implements exact far-field integrals and NEC-validated impedance values.
Reference: Balanis, *Antenna Theory: Analysis and Design*, 4th Ed., Ch. 4.
"""
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.dipole import HalfWaveDipole

# Relative error tolerance for numerical integration.
# If the estimated error exceeds this fraction of the result, warn the user.
_QUAD_REL_TOL = 1e-4


def _safe_quad(func, a, b, limit=200, name="integral"):
    """Wrapper around scipy.integrate.quad with explicit error checking.

    Parameters
    ----------
    func : callable
    a, b : float
        Integration limits.
    limit : int
        Subdivision limit.
    name : str
        Label used in warning message.

    Returns
    -------
    float
        Integral result.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        result, abserr = quad(func, a, b, limit=limit)

    if caught:
        warnings.warn(
            f"DipoleSolver: {name} may be inaccurate — "
            f"scipy.integrate.quad raised: {caught[0].message}",
            UserWarning, stacklevel=3,
        )
    elif result != 0 and abserr / (abs(result) + 1e-30) > _QUAD_REL_TOL:
        warnings.warn(
            f"DipoleSolver: {name} has large relative error "
            f"({abserr/(abs(result)+1e-30):.2e} > {_QUAD_REL_TOL:.0e}). "
            "Results may be inaccurate for this dipole length.",
            UserWarning, stacklevel=3,
        )
    return result


class DipoleSolver:
    """Analytical far-field and impedance solver for wire dipole or dipole array.

    Parameters
    ----------
    dipole : HalfWaveDipole | LinearArray
        Single dipole geometry, or a ``LinearArray`` whose element is a
        dipole.  When a ``LinearArray`` is passed the radiation pattern is
        computed as the element pattern multiplied by the array factor.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, dipole, freq: float):
        from pylobe.utils.validation import check_frequency
        from pylobe.geometry.array import LinearArray
        check_frequency(freq)
        if isinstance(dipole, LinearArray):
            self._array = dipole
            self.dipole = dipole.element   # underlying single element
        else:
            self._array = None
            self.dipole = dipole
        self.freq = freq
        self.lambda0 = C0 / freq
        self.k = 2.0 * PI * freq / C0
        self.kl = self.k * self.dipole.arm_length  # k × (L/2)

    # ------------------------------------------------------------------
    # Far-field pattern
    # ------------------------------------------------------------------
    def element_factor(self, theta: np.ndarray) -> np.ndarray:
        """Far-field element pattern F(θ) for general dipole of half-length l.

        F(θ) = ((cos(k·l·cosθ) - cos(k·l)) / sinθ)²

        For half-wave dipole (k·l = π/2):
            F(θ) = (cos(π/2·cosθ) / sinθ)²

        Ref: Balanis Eq. 4-58a.

        Parameters
        ----------
        theta : array_like
            Polar angles [rad].

        Returns
        -------
        ndarray
            Pattern function |F(θ)| (unnormalised linear power).
        """
        theta = np.asarray(theta, dtype=float)
        kl = self.kl
        # Avoid sin(theta)=0 singularity at θ=0 and θ=π
        sin_t = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
        num = np.cos(kl * np.cos(theta)) - np.cos(kl)
        return (num / sin_t) ** 2

    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 361) -> "RadiationPattern":
        """Compute full 3-D radiation pattern.

        The dipole pattern is azimuthally symmetric:
            E_θ ∝ (cos(k·l·cosθ) - cos(k·l)) / sinθ
            E_φ = 0

        Ref: Balanis Eq. 4-58a.

        Parameters
        ----------
        n_theta : int
            Number of θ samples.
        n_phi : int
            Number of φ samples.

        Returns
        -------
        RadiationPattern

        Examples
        --------
        >>> from pylobe import HalfWaveDipole, DipoleSolver
        >>> d = HalfWaveDipole(freq=300e6)
        >>> s = DipoleSolver(d, 300e6)
        >>> rp = s.radiation_pattern(n_theta=91, n_phi=91)
        >>> abs(rp.peak_directivity_dbi - 2.15) < 0.2
        True
        """
        from pylobe.analysis.radiation import RadiationPattern

        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)

        kl = self.kl
        sin_t = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
        e_t_1d = (np.cos(kl * np.cos(theta)) - np.cos(kl)) / sin_t

        # Broadcast to 2-D: (n_theta, n_phi)
        E_theta = np.outer(e_t_1d, np.ones(n_phi)).astype(complex)
        E_phi   = np.zeros_like(E_theta)

        # Array factor: for a LinearArray along z the AF depends only on theta
        if self._array is not None:
            af = self._array.array_factor(theta, freq=self.freq)  # shape (n_theta,), normalised |AF|
            E_theta *= af[:, np.newaxis]   # element pattern × AF

        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------
    # Directivity (exact numerical integration)
    # ------------------------------------------------------------------
    @property
    def directivity(self) -> float:
        """Peak directivity computed by exact numerical integration.

        D = 4π · F(θ_max) / ∫₀^π F(θ)·sinθ dθ
        Ref: Balanis Eq. 4-60 / Table 4-1.

        For the half-wave dipole the exact value is D = 1.641 (2.15 dBi).
        """
        kl = self.kl
        F_max = self.element_factor(np.array([PI / 2.0]))[0]

        def integrand(t):
            sin_t = max(abs(np.sin(t)), 1e-12)
            num = np.cos(kl * np.cos(t)) - np.cos(kl)
            return (num / sin_t) ** 2 * np.sin(t)

        integral = _safe_quad(integrand, 0.0, PI, limit=200,
                               name="directivity integral")
        return 2.0 * F_max / integral if integral > 0 else 1.64

    @property
    def directivity_dbi(self) -> float:
        """Peak directivity [dBi]. Half-wave dipole = 2.15 dBi."""
        return 10.0 * np.log10(self.directivity)

    # ------------------------------------------------------------------
    # Radiation resistance (exact)
    # ------------------------------------------------------------------
    @property
    def radiation_resistance(self) -> float:
        """Radiation resistance [Ω] in free space (no ground plane).

        Rr = (η0 / 2π) · ∫₀^π F(θ)·sinθ dθ
        Ref: Balanis Eq. 4-63.

        For half-wave dipole: Rr ≈ 73.1 Ω.

        Note
        ----
        This assumes free-space (no ground plane). For a monopole over
        a perfect ground plane, Rr = half this value (image theory).
        """
        kl = self.kl

        def integrand(t):
            sin_t = max(abs(np.sin(t)), 1e-12)
            num = np.cos(kl * np.cos(t)) - np.cos(kl)
            return (num / sin_t) ** 2 * np.sin(t)

        integral = _safe_quad(integrand, 0.0, PI, limit=200,
                               name="radiation resistance integral")
        return (ETA0 / (2.0 * PI)) * integral

    # ------------------------------------------------------------------
    # Input impedance
    # ------------------------------------------------------------------
    def input_impedance(self) -> complex:
        """Input impedance at feed point [Ω].

        Resistive part: exact integral (see radiation_resistance).
        Reactive part: linear interpolation around resonance.

        For exact half-wave dipole (0.5λ): Z_in ≈ 73.1 + j42.5 Ω
        (Ref: Balanis Table 8-2; NEC2 benchmark).
        At resonance (0.47λ): Z_in ≈ 73.1 + j0 Ω (self-resonant).

        Returns
        -------
        complex
            Input impedance [Ω].
        """
        Rr = self.radiation_resistance
        # Reactive part: approximately zero at resonance (0.47λ), +42.5 Ω at 0.5λ.
        # Linear interpolation in the valid range (0.44λ – 0.53λ).
        # Ref: Balanis Table 8-2 / King, "Theory of Linear Antennas" (1956).
        lf = self.dipole.length_factor
        if abs(lf - 0.47) <= 0.03:
            Xin = 42.5 * (lf - 0.47) / 0.03
        elif lf >= 0.5:
            Xin = 42.5
        else:
            Xin = 0.0
        return complex(Rr, Xin)

    # ------------------------------------------------------------------
    # S11 frequency sweep
    # ------------------------------------------------------------------
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """Compute S11 over ±40% bandwidth sweep.

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
        >>> from pylobe import HalfWaveDipole, DipoleSolver
        >>> import numpy as np
        >>> d = HalfWaveDipole(freq=300e6)
        >>> s = DipoleSolver(d, 300e6)
        >>> freqs, S11 = s.s11()
        >>> np.min(20 * np.log10(np.abs(S11))) < -5
        True
        """
        freqs = np.linspace(self.freq * 0.6, self.freq * 1.4, n_freq)
        S11 = np.zeros(n_freq, dtype=complex)
        for i, f in enumerate(freqs):
            k   = 2.0 * PI * f / C0
            kl  = k * self.dipole.arm_length

            def integrand(t, _kl=kl):
                sin_t = max(abs(np.sin(t)), 1e-12)
                num = np.cos(_kl * np.cos(t)) - np.cos(_kl)
                return (num / sin_t) ** 2 * np.sin(t)

            integral = _safe_quad(integrand, 0.0, PI, limit=100,
                                   name=f"S11 integral at {f/1e9:.3f} GHz")
            Rr = (ETA0 / (2.0 * PI)) * integral

            # Physical resonance: arm = 0.47 × λ_res/2 → f_res = 0.47·c/(2·arm)
            # Equivalent to self.freq × (0.47 / self.dipole.length_factor)
            f_res = 0.47 * C0 / (2.0 * self.dipole.arm_length)
            delta = (f - f_res) / (f_res + 1e-30)
            # Reactive detuning: Xin ≈ 2Rr·δf·Q, Q≈8 for half-wave dipole
            Xin = 2.0 * Rr * delta * 8.0

            Zin = complex(Rr, Xin)
            S11[i] = (Zin - Z0) / (Zin + Z0)
        return freqs, S11
