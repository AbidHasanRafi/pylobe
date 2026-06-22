"""Analytical solver for horn antennas (aperture theory)."""
import numpy as np
from pylobe.constants import C0, PI
from pylobe.geometry.horn import PyramidalHorn


class HornSolver:
    """Aperture-theory solver for pyramidal horn antennas.

    The pyramidal horn pattern is computed from the Kirchhoff-Huygens
    aperture integration.  Separable E-plane and H-plane distributions
    are assumed (Balanis, Antenna Theory, 4th Ed., Chapter 13):

    E-plane (φ=0): uniform in x, sinusoidal distribution → sinc-like pattern
    H-plane (φ=90°): cosine distribution → broader sinc pattern

    Gain (aperture efficiency η_ap ≈ 0.51 for optimum horn):
        G = η_ap · (4π/λ²) · a_ap · b_ap

    Parameters
    ----------
    horn : PyramidalHorn
        Horn geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, horn: PyramidalHorn, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        if not isinstance(horn, PyramidalHorn):
            raise TypeError("horn must be a PyramidalHorn instance")
        self.horn    = horn
        self.freq    = freq
        self.lambda0 = C0 / freq
        self.k       = 2.0 * PI * freq / C0

    # ------------------------------------------------------------------ #
    # Pattern
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 181) -> "RadiationPattern":
        """3-D radiation pattern from aperture theory.

        The far-field pattern is approximated as separable in E and H planes:
            E_θ ≈ (1 + cosθ)/2 · F_H(θ,φ) · F_E(θ,φ)

        where:
            F_E (E-plane, along b_ap): sinc(k·b_ap·sinθ·cosφ / 2)
            F_H (H-plane, along a_ap): [cos(k·a_ap·sinθ·sinφ / 2)] / [1 - (k·a_ap·sinθ·sinφ/π)²]

        Parameters
        ----------
        n_theta, n_phi : int
            Angular resolution.

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern

        theta = np.linspace(0, PI,       n_theta)
        phi   = np.linspace(0, 2 * PI,  n_phi)
        k     = self.k
        a_ap  = self.horn.a_ap
        b_ap  = self.horn.b_ap

        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        cos_t  = np.cos(TH)
        sin_t  = np.sin(TH)
        cos_p  = np.cos(PH)
        sin_p  = np.sin(PH)

        # E-plane factor (along b_ap, y-axis) — uniform distribution → sinc
        u_E = k * b_ap * sin_t * cos_p / 2.0
        F_E = np.sinc(u_E / PI)   # numpy.sinc is sin(πx)/(πx)

        # H-plane factor (along a_ap, x-axis) — cosine distribution
        u_H  = k * a_ap * sin_t * sin_p / 2.0
        denom = 1.0 - (u_H / (PI / 2.0)) ** 2
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        F_H  = np.cos(u_H) / denom

        # Space factor (1+cosθ)/2 accounts for forward radiation only
        space_factor = (1.0 + cos_t) / 2.0

        E_theta = (space_factor * np.abs(F_E) * np.abs(F_H)).astype(complex)
        E_phi   = np.zeros_like(E_theta)

        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # Gain
    # ------------------------------------------------------------------ #
    @property
    def gain_dbi(self) -> float:
        """Approximate peak gain [dBi] from aperture theory."""
        return self.horn.gain_approx_dbi

    # ------------------------------------------------------------------ #
    # S11 (approximate — waveguide match)
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """S11 vs frequency for horn fed from a waveguide.

        A well-designed horn is a good travelling-wave structure with
        |Γ| < 0.1 over the operating bandwidth (typically TE10 single-mode
        range of the feed waveguide).  Below cut-off, S11 → 1 (total reflection).

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω] (usually 50 Ω coax-to-waveguide adapter).
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple  (freq_array [Hz], S11_complex ndarray)
        """
        f_co = C0 / (2.0 * self.horn.a_wg)  # TE10 cut-off of feed waveguide
        freqs = np.linspace(f_co * 0.8, self.freq * 1.5, n_freq)
        S11   = np.zeros(n_freq, dtype=complex)

        for i, f in enumerate(freqs):
            if f <= f_co:
                S11[i] = 1.0 + 0j  # below cut-off: total reflection
                continue

            # Waveguide characteristic impedance
            fc_norm = f_co / f
            Z_wg    = Z0 / np.sqrt(1.0 - fc_norm ** 2)

            # Horn match improves with gain — empirical VSWR model
            # VSWR ≈ 1 + exp(-G/5) where G is linear gain
            G_lin  = 10 ** (self.horn.gain_approx_dbi / 10.0)
            vswr   = 1.0 + np.exp(-G_lin / 50.0)
            Gamma  = (vswr - 1.0) / (vswr + 1.0)

            # Add slight frequency dependence (better match away from cut-off)
            taper  = 1.0 - np.exp(-(f / f_co - 1.0) * 3.0)
            S11[i] = Gamma * taper * np.exp(1j * PI * f / self.freq)

        return freqs, S11

    # ------------------------------------------------------------------ #
    # Input impedance
    # ------------------------------------------------------------------ #
    def input_impedance(self) -> complex:
        """Approximate input impedance at aperture plane [Ω]."""
        f_co   = C0 / (2.0 * self.horn.a_wg)
        fc_norm = f_co / self.freq
        # Waveguide impedance at design freq
        Z_wg = 377.0 / np.sqrt(1.0 - fc_norm ** 2)
        return complex(Z_wg, 0.0)
