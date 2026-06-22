"""Analytical solver for folded dipole and bow-tie antennas."""
import numpy as np
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.dipole import FoldedDipole, BowTieDipole


class FoldedDipoleSolver:
    """Analytical solver for folded dipole antennas.

    The folded dipole is equivalent to a half-wave dipole but with
    input impedance transformed by a factor of 4 (for equal-radius wires):
        Z_fd = 4 × Z_dipole ≈ 292 + j0 Ω at resonance

    Radiation pattern: identical to a half-wave dipole
    Gain: 2.15 dBi (same as half-wave dipole)
    Bandwidth: approximately 10× wider than a simple dipole

    Parameters
    ----------
    antenna : FoldedDipole | BowTieDipole
        Folded dipole geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, antenna, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        if not isinstance(antenna, (FoldedDipole, BowTieDipole)):
            raise TypeError("antenna must be FoldedDipole or BowTieDipole")
        self.antenna  = antenna
        self.freq     = freq
        self.lambda0  = C0 / freq
        self.k        = 2.0 * PI * freq / C0

    # ------------------------------------------------------------------ #
    # Pattern (same as half-wave dipole)
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 181) -> "RadiationPattern":
        """3-D radiation pattern.

        Identical to half-wave dipole: figure-of-eight in E-plane,
        omnidirectional in H-plane.

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

        arm_length = (self.antenna.arm_length if hasattr(self.antenna, 'arm_length')
                      else self.antenna.arm_length)
        kl    = self.k * arm_length
        sin_t = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
        e_1d  = (np.cos(kl * np.cos(theta)) - np.cos(kl)) / sin_t

        if isinstance(self.antenna, BowTieDipole):
            # Bow-tie: broader beam due to frequency-independent flare
            alpha = np.deg2rad(self.antenna.flare_angle_deg)
            # Taper the null depth at θ=0/π with flare correction
            e_1d = e_1d * (1.0 + 0.3 * np.sin(alpha) * np.sin(theta))

        E_theta = np.outer(e_1d, np.ones(n_phi)).astype(complex)
        E_phi   = np.zeros_like(E_theta)
        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # Input impedance
    # ------------------------------------------------------------------ #
    def input_impedance(self) -> complex:
        """Input impedance [Ω].

        Folded dipole: Z_in = 4 × Z_half_wave ≈ 292 + j0 Ω at resonance.
        Bow-tie:       Z_in ≈ 120π × ln(cot(α/2)) where α is flare angle.
        """
        from scipy.integrate import quad

        arm = (self.antenna.arm_length if hasattr(self.antenna, 'arm_length')
               else self.antenna.arm_length)
        kl  = self.k * arm

        def integrand(t):
            sin_t = max(abs(np.sin(t)), 1e-12)
            num = np.cos(kl * np.cos(t)) - np.cos(kl)
            return (num / sin_t) ** 2 * np.sin(t)

        integral, _ = quad(integrand, 0.0, PI, limit=200)
        Rr_dipole   = (ETA0 / (2.0 * PI)) * integral

        if isinstance(self.antenna, FoldedDipole):
            # 4× impedance transformation
            Z_in = complex(4.0 * Rr_dipole, 0.0)
        else:
            # Bow-tie: characteristic impedance from flare angle
            alpha = self.antenna.flare_angle_rad
            Z_char = ETA0 * np.log(1.0 / np.tan(alpha / 2.0)) / PI
            Z_in   = complex(Z_char, 0.0)

        return Z_in

    # ------------------------------------------------------------------ #
    # S11 frequency sweep
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 300.0, n_freq: int = 300) -> tuple:
        """S11 vs frequency.

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω].  Use 300 Ω for folded dipole
            (Yagi TV feed), 50 Ω for bow-tie with a matching network.
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple  (freq_array [Hz], S11_complex ndarray)
        """
        from scipy.integrate import quad

        arm   = self.antenna.arm_length
        f_res = 0.47 * C0 / (2.0 * arm)

        freqs = np.linspace(self.freq * 0.5, self.freq * 1.5, n_freq)
        S11   = np.zeros(n_freq, dtype=complex)

        for i, f in enumerate(freqs):
            k  = 2.0 * PI * f / C0
            kl = k * arm

            def integrand(t, _kl=kl):
                sin_t = max(abs(np.sin(t)), 1e-12)
                num = np.cos(_kl * np.cos(t)) - np.cos(_kl)
                return (num / sin_t) ** 2 * np.sin(t)

            integral, _ = quad(integrand, 0.0, PI, limit=100)
            Rr_dip = (ETA0 / (2.0 * PI)) * integral

            delta = (f - f_res) / (f_res + 1e-30)

            if isinstance(self.antenna, FoldedDipole):
                # 4× impedance; bandwidth ~10× broader → Q_eff ≈ 0.8
                Xin = 2.0 * 4 * Rr_dip * delta * 0.8
                Zin = complex(4.0 * Rr_dip, Xin)
            else:
                # Bow-tie: frequency-independent; slight roll-off at extremes
                alpha  = self.antenna.flare_angle_rad
                Z_char = ETA0 * np.log(1.0 / (np.tan(alpha / 2.0) + 1e-12)) / PI
                Xin    = Z_char * 0.1 * delta
                Zin    = complex(Z_char, Xin)

            S11[i] = (Zin - Z0) / (Zin + Z0)

        return freqs, S11
