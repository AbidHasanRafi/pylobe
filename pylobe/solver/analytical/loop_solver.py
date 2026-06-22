"""Analytical solver for loop antennas."""
import numpy as np
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.loop import SmallLoopAntenna, LargeLoopAntenna


class LoopSolver:
    """Analytical solver for small and large loop antennas.

    Small loop (C ≪ λ):
        Pattern: same as short magnetic dipole, sin²θ — identical shape to
        short electric dipole.
        Rr = 20 π² (C/λ)⁴  (Balanis Eq. 5-29)
        Gain = 1.76 dBi (3/2 = 1.76 dBi, identical to short dipole)

    Large/resonant loop (C ≈ λ):
        Pattern computed by summing contributions from each short current
        element around the loop perimeter (piecewise integration).
        At C = λ, broadside gain ≈ 3.4 dBi.

    Parameters
    ----------
    antenna : SmallLoopAntenna | LargeLoopAntenna
        Loop geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, antenna, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        if not isinstance(antenna, (SmallLoopAntenna, LargeLoopAntenna)):
            raise TypeError("antenna must be SmallLoopAntenna or LargeLoopAntenna")
        self.antenna  = antenna
        self.freq     = freq
        self.lambda0  = C0 / freq
        self.k        = 2.0 * PI * freq / C0

    # ------------------------------------------------------------------ #
    # Pattern
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 181) -> "RadiationPattern":
        """3-D radiation pattern.

        Small loop: sin²θ (magnetic dipole pattern, symmetric in φ).
        Large loop: numerical integration over loop perimeter.

        Parameters
        ----------
        n_theta, n_phi : int
            Angular resolution.

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern

        theta = np.linspace(0, PI,      n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        k     = self.k

        if isinstance(self.antenna, SmallLoopAntenna):
            # Small loop: magnetic dipole pattern  ∝ sinθ
            sin_t   = np.sin(theta)
            e_t_1d  = sin_t
            E_theta = np.outer(e_t_1d, np.ones(n_phi)).astype(complex)
            E_phi   = np.zeros_like(E_theta)
        else:
            # Large loop: piecewise numerical integration
            r = self.antenna.radius
            # Sample loop at N_loop points
            N_loop = 360
            phi_loop = np.linspace(0, 2 * PI, N_loop, endpoint=False)
            dphi = 2 * PI / N_loop

            TH, PH = np.meshgrid(theta, phi, indexing='ij')
            sin_TH = np.sin(TH); cos_TH = np.cos(TH)
            sin_PH = np.sin(PH); cos_PH = np.cos(PH)

            E_theta = np.zeros((n_theta, n_phi), dtype=complex)
            E_phi   = np.zeros((n_theta, n_phi), dtype=complex)

            for ph_loop in phi_loop:
                # Position on loop
                x0 = r * np.cos(ph_loop)
                y0 = r * np.sin(ph_loop)
                # Current element direction (tangential to loop)
                dx = -np.sin(ph_loop) * r * dphi
                dy =  np.cos(ph_loop) * r * dphi

                # Phase from source to observation direction
                r_dot = x0 * sin_TH * cos_PH + y0 * sin_TH * sin_PH
                phase = k * r_dot

                # θ-component of current element
                dI_theta = (dx * cos_TH * cos_PH + dy * cos_TH * sin_PH)
                # φ-component of current element
                dI_phi   = (-dx * sin_PH + dy * cos_PH)

                E_theta += dI_theta * np.exp(1j * phase)
                E_phi   += dI_phi   * np.exp(1j * phase)

        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # Radiation resistance and impedance
    # ------------------------------------------------------------------ #
    @property
    def radiation_resistance(self) -> float:
        """Radiation resistance [Ω]."""
        if isinstance(self.antenna, SmallLoopAntenna):
            return self.antenna.Rr
        else:
            C_over_lam = self.antenna.circumference / self.lambda0
            return 20.0 * PI ** 2 * C_over_lam ** 4 * max(1.0, C_over_lam)

    def input_impedance(self) -> complex:
        """Input impedance [Ω].

        Small loop: predominantly inductive  (Xin = ωL ≫ Rr).
        Large loop: approximately resistive near resonance.
        """
        Rr   = self.radiation_resistance
        if isinstance(self.antenna, SmallLoopAntenna):
            Xin = self.antenna.Xin
        else:
            cf  = self.antenna.circumference_factor
            Xin = 100.0 * (cf - 1.0)    # crosses zero near C = λ
        return complex(Rr, Xin)

    # ------------------------------------------------------------------ #
    # S11 frequency sweep
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """S11 vs frequency.

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
        mu0     = 4e-7 * PI
        freqs   = np.linspace(self.freq * 0.5, self.freq * 1.5, n_freq)
        S11     = np.zeros(n_freq, dtype=complex)

        if isinstance(self.antenna, SmallLoopAntenna):
            r    = self.antenna.radius
            N    = self.antenna.N_turns
            Lind = self.antenna.L_inductance

            for i, f in enumerate(freqs):
                k   = 2.0 * PI * f / C0
                C_lam = 2 * PI * r / (C0 / f)
                Rr  = 20.0 * PI ** 2 * C_lam ** 4 * N ** 2
                Xin = 2.0 * PI * f * Lind
                Zin = complex(Rr, Xin)
                S11[i] = (Zin - Z0) / (Zin + Z0)
        else:
            # Large loop — resonant near C = λ
            r  = self.antenna.radius
            cf = self.antenna.circumference_factor
            f_res = self.freq / cf   # resonance at C = λ
            mu0   = 4e-7 * PI
            Lind  = mu0 * r * (np.log(8 * r / (r / 100)) - 2.0)

            for i, f in enumerate(freqs):
                k     = 2.0 * PI * f / C0
                C_lam = 2 * PI * r / (C0 / f)
                Rr    = 20.0 * PI ** 2 * C_lam ** 4 * max(1.0, C_lam)
                Xin   = 2.0 * PI * f * Lind - 1.0 / (2.0 * PI * f * Lind / (2.0 * PI * f) ** 2 + 1e-30)
                delta = (f - f_res) / (f_res + 1e-30)
                Xin   = 100.0 * delta
                Zin   = complex(max(Rr, 5.0), Xin)
                S11[i] = (Zin - Z0) / (Zin + Z0)

        return freqs, S11
