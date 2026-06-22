"""Analytical solver for quarter-wave monopole and helical monopole."""
import numpy as np
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.monopole import QuarterWaveMonopole, HelicalMonopole


class MonopoleSolver:
    """Analytical solver for monopole antennas over a ground plane.

    Uses image theory: a monopole of length L over a perfect ground plane
    is equivalent to a dipole of length 2L in free space, with the pattern
    truncated to the upper hemisphere (θ ∈ [0, π/2]).

    Gain = 2 × dipole directivity (because radiated power occupies only
    the upper hemisphere while the pattern shape is the same).

    For the quarter-wave monopole:
        Z_in ≈ 36.5 + j0 Ω   (half of the 73 Ω half-wave dipole)
        Peak gain = 5.16 dBi  (3 dB above half-wave dipole)

    Parameters
    ----------
    antenna : QuarterWaveMonopole | HelicalMonopole
        Monopole geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, antenna, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        self.antenna  = antenna
        self.freq     = freq
        self.lambda0  = C0 / freq
        self.k        = 2.0 * PI * freq / C0

        if isinstance(antenna, QuarterWaveMonopole):
            self._type = 'quarter_wave'
            self.kl    = self.k * antenna.L
        elif isinstance(antenna, HelicalMonopole):
            self._type = 'helical'
        else:
            raise TypeError("antenna must be QuarterWaveMonopole or HelicalMonopole")

    # ------------------------------------------------------------------ #
    # Pattern (upper hemisphere only)
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 91,
                          n_phi: int = 181) -> "RadiationPattern":
        """Compute the upper-hemisphere radiation pattern.

        Parameters
        ----------
        n_theta : int
            Number of θ samples over [0, π/2] (upper hemisphere).
        n_phi : int
            Number of φ samples.

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern

        # Monopole: only upper hemisphere θ ∈ [0, π/2]
        theta = np.linspace(0, PI / 2.0, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)

        if self._type == 'quarter_wave':
            kl    = self.kl
            sin_t = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
            e_t_1d = (np.cos(kl * np.cos(theta)) - np.cos(kl)) / sin_t
        else:
            # Helical in axial mode: approximate end-fire pattern
            # F(θ) = cos(θ)  (pencil beam toward θ=0)
            e_t_1d = np.cos(theta)

        E_theta = np.outer(e_t_1d, np.ones(n_phi)).astype(complex)
        E_phi   = np.zeros_like(E_theta)
        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # Radiation resistance
    # ------------------------------------------------------------------ #
    @property
    def radiation_resistance(self) -> float:
        """Radiation resistance [Ω].

        For a perfect ground plane: Rr_mono = Rr_dipole / 2.
        """
        from scipy.integrate import quad

        kl = self.kl if self._type == 'quarter_wave' else self.k * self.antenna.total_length

        def integrand(t):
            sin_t = max(abs(np.sin(t)), 1e-12)
            num = np.cos(kl * np.cos(t)) - np.cos(kl)
            return (num / sin_t) ** 2 * np.sin(t)

        integral, _ = quad(integrand, 0.0, PI, limit=200)
        Rr_dipole = (ETA0 / (2.0 * PI)) * integral
        return Rr_dipole / 2.0

    # ------------------------------------------------------------------ #
    # Input impedance
    # ------------------------------------------------------------------ #
    def input_impedance(self) -> complex:
        """Input impedance [Ω].

        Quarter-wave monopole: Z_in ≈ 36.5 + j0 Ω at resonance.
        """
        if self._type == 'quarter_wave':
            Rr = self.radiation_resistance
            # Reactance: approximately zero at resonance (length_factor = 1.0)
            lf = self.antenna.L / (self.lambda0 / 4.0)
            Xin = 21.25 * (lf - 1.0) / 0.03 if abs(lf - 1.0) <= 0.05 else 0.0
            return complex(Rr, Xin)
        else:
            # Helical: approximate
            return complex(140.0, 0.0)

    # ------------------------------------------------------------------ #
    # S11 frequency sweep
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """Compute S11 over ±40% sweep around design frequency.

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

        if self._type == 'helical':
            freqs = np.linspace(self.freq * 0.6, self.freq * 1.4, n_freq)
            S11   = np.zeros(n_freq, dtype=complex)
            # Simple helical model: resonance at design freq, Q ≈ 5
            f_res = self.freq
            Rr_0  = 140.0
            for i, f in enumerate(freqs):
                delta = (f - f_res) / f_res
                Zin   = complex(Rr_0, 2 * Rr_0 * delta * 5.0)
                S11[i] = (Zin - Z0) / (Zin + Z0)
            return freqs, S11

        # Quarter-wave monopole: full analytical sweep
        if isinstance(self.antenna, QuarterWaveMonopole):
            arm_L  = self.antenna.L
            lf0    = self.antenna.L / (self.lambda0 / 4.0)
            f_res  = 0.47 * C0 / (2.0 * arm_L)   # uses dipole image length 2L
        else:
            arm_L = self.antenna.total_length
            f_res = 0.47 * C0 / (2.0 * arm_L)

        freqs = np.linspace(self.freq * 0.6, self.freq * 1.4, n_freq)
        S11   = np.zeros(n_freq, dtype=complex)

        for i, f in enumerate(freqs):
            k    = 2.0 * PI * f / C0
            kl   = k * arm_L

            def integrand(t, _kl=kl):
                sin_t = max(abs(np.sin(t)), 1e-12)
                num = np.cos(_kl * np.cos(t)) - np.cos(_kl)
                return (num / sin_t) ** 2 * np.sin(t)

            integral, _ = quad(integrand, 0.0, PI, limit=100)
            Rr = (ETA0 / (2.0 * PI)) * integral / 2.0  # divide by 2 for monopole

            delta = (f - f_res) / (f_res + 1e-30)
            Xin   = 2.0 * Rr * delta * 8.0
            Zin   = complex(Rr, Xin)
            S11[i] = (Zin - Z0) / (Zin + Z0)

        return freqs, S11
