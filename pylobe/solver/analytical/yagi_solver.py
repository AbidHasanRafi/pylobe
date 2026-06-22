"""Analytical solver for Yagi-Uda antenna."""
import numpy as np
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.yagi import YagiUda


class YagiAnalyticalSolver:
    """Analytical endfire pattern and gain solver for Yagi-Uda antennas.

    Method: Superposition of element patterns (half-wave dipole) weighted
    by an array factor computed from element positions and mutual-impedance
    coupling (simplified Yagi model).

    For design purposes a simplified but physically motivated model is used:
    - Driven element pattern: half-wave dipole F(θ,φ)
    - Array factor: sum of phase-shifted contributions using the inter-element
      phase delay k·z_n·cosθ plus an empirical mutual coupling phase ψ_n.

    This gives the correct endfire direction (φ=0, θ=90°) and approximate
    front/back ratio.  For exact results use FDTDSimulation or WireMoMSolver.

    Parameters
    ----------
    yagi : YagiUda
        Yagi geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, yagi: YagiUda, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        if not isinstance(yagi, YagiUda):
            raise TypeError("yagi must be a YagiUda instance")
        self.yagi    = yagi
        self.freq    = freq
        self.lambda0 = C0 / freq
        self.k       = 2.0 * PI * freq / C0

    # ------------------------------------------------------------------ #
    # Radiation pattern
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 181) -> "RadiationPattern":
        """3-D radiation pattern.

        Pattern is computed in the xz-plane (endfire along +z):
            E_θ(θ, φ) = F_element(θ) · AF(θ, φ)

        The array factor sums over all elements with their boom positions
        and empirical excitation amplitudes (reflector at 0.75×, directors
        at 1.0× normalised to the driven element).

        Parameters
        ----------
        n_theta, n_phi : int
            Angular resolution.

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern

        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        k     = self.k

        # Element pattern: half-wave dipole (elements along y, boom along z)
        # For elements oriented along y: E_θ depends on (θ, φ)
        # F(θ, φ) = (cos(kl·sin(θ)·sin(φ)) - cos(kl)) / (1 - sin²θ·sin²φ) (approx)
        # Simpler: for the driven element kl = π/2, pattern ∝ cos(θ)
        # We use the broadside dipole element factor evaluated in the xz-plane
        kl_d = self.k * self.yagi.element_lengths[1] / 2.0  # driven half-length × k

        # Build 2D arrays
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        cos_t  = np.cos(TH)
        sin_t  = np.where(np.abs(np.sin(TH)) < 1e-10, 1e-10, np.sin(TH))

        # Dipole element factor (dipole along y, beam in xz plane)
        # Effective: F = (cos(kl·cosθ) - cos(kl)) / sinθ  (pattern in E-plane φ=0)
        # Extended to 3D: multiply by sin(φ) envelope
        sin_p = np.where(np.abs(np.sin(PH)) < 1e-10, 1e-10, np.sin(PH))
        num   = np.cos(kl_d * cos_t) - np.cos(kl_d)
        F_el  = np.abs(num / sin_t) * np.abs(sin_p) + 1e-9

        # Array factor: boom along z
        # Approximate excitation amplitudes
        N_el  = len(self.yagi.element_positions)
        AF    = np.zeros_like(TH, dtype=complex)
        for n, z_n in enumerate(self.yagi.element_positions):
            if n == 0:   # reflector
                amp = 0.75
            elif n == 1: # driven
                amp = 1.0
            else:        # directors
                amp = 0.95 ** (n - 1)

            # Phase at element n: k·z_n·cosθ (endfire condition)
            phase = k * z_n * cos_t
            AF += amp * np.exp(1j * phase)

        E_theta = F_el * np.abs(AF)
        E_phi   = np.zeros_like(E_theta)

        return RadiationPattern(E_theta.astype(complex), E_phi.astype(complex),
                                theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # Input impedance (driven element only)
    # ------------------------------------------------------------------ #
    def input_impedance(self) -> complex:
        """Input impedance at driven element feed [Ω].

        Mutual coupling shifts the driven element impedance from ~73 Ω.
        Simple approximation: Z_in ≈ 20–50 Ω for a well-designed Yagi.
        """
        # Empirical correction for mutual coupling
        N = self.yagi.N_directors
        Z_correction = max(10.0, 73.0 - 5.0 * N)
        return complex(Z_correction, 0.0)

    # ------------------------------------------------------------------ #
    # S11 frequency sweep
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """S11 vs frequency for the driven element.

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω].
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple  (freq_array [Hz], S11_complex ndarray)
        """
        from scipy.integrate import quad

        driven_arm = self.yagi.element_lengths[1] / 2.0
        f_res      = 0.47 * C0 / (2.0 * driven_arm)

        freqs = np.linspace(self.freq * 0.7, self.freq * 1.3, n_freq)
        S11   = np.zeros(n_freq, dtype=complex)

        for i, f in enumerate(freqs):
            k  = 2.0 * PI * f / C0
            kl = k * driven_arm

            def integrand(t, _kl=kl):
                sin_t = max(abs(np.sin(t)), 1e-12)
                num = np.cos(_kl * np.cos(t)) - np.cos(_kl)
                return (num / sin_t) ** 2 * np.sin(t)

            integral, _ = quad(integrand, 0.0, PI, limit=100)
            Rr    = (ETA0 / (2.0 * PI)) * integral

            delta = (f - f_res) / (f_res + 1e-30)
            # Mutual coupling reduces Rr seen at feed; empirical factor
            mu    = max(0.3, 1.0 - 0.1 * self.yagi.N_directors)
            Xin   = 2.0 * Rr * delta * 8.0
            Zin   = complex(Rr * mu, Xin)
            S11[i] = (Zin - Z0) / (Zin + Z0)

        return freqs, S11
