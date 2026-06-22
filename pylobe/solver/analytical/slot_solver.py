"""Analytical solver for slot antenna (Babinet's principle)."""
import numpy as np
from pylobe.constants import C0, PI, ETA0
from pylobe.geometry.slot import SlotAntenna


class SlotSolver:
    """Analytical solver for resonant slot antennas.

    Uses Babinet's (complementarity) principle to derive the slot pattern
    and impedance from the equivalent half-wave dipole:

        Z_slot × Z_dipole* = (η₀/2)²  ≈ 35 476 Ω²

    Radiation pattern: complementary to the dipole — the E and H fields
    are exchanged, so the slot has a toroidal pattern but it is the
    H-field that varies as (cos(kl cosθ) − cos(kl))/sinθ.

    At resonance (length_factor ≈ 0.47):
        Z_slot ≈ 363 + j0 Ω  (purely resistive)
        Gain   ≈ 2.15 dBi    (same as half-wave dipole)

    Parameters
    ----------
    slot : SlotAntenna
        Slot geometry object.
    freq : float
        Analysis frequency [Hz].
    """

    def __init__(self, slot: SlotAntenna, freq: float):
        from pylobe.utils.validation import check_frequency
        check_frequency(freq)
        if not isinstance(slot, SlotAntenna):
            raise TypeError("slot must be a SlotAntenna instance")
        self.slot    = slot
        self.freq    = freq
        self.lambda0 = C0 / freq
        self.k       = 2.0 * PI * freq / C0

    # ------------------------------------------------------------------ #
    # Impedance via Babinet's principle
    # ------------------------------------------------------------------ #
    def input_impedance(self) -> complex:
        """Input impedance [Ω] via Babinet's complementarity principle."""
        from scipy.integrate import quad

        kl = self.k * self.slot.slot_length / 2.0

        def integrand(t):
            sin_t = max(abs(np.sin(t)), 1e-12)
            num = np.cos(kl * np.cos(t)) - np.cos(kl)
            return (num / sin_t) ** 2 * np.sin(t)

        integral, _ = quad(integrand, 0.0, PI, limit=200)
        Rr_dip  = (ETA0 / (2.0 * PI)) * integral
        Xin_dip = 42.5 * (self.slot.slot_length / (0.47 * self.lambda0) - 1.0)
        Z_dip   = complex(Rr_dip, Xin_dip)

        # Babinet: Z_slot = (η₀/2)² / Z_dip*
        BABINET = (ETA0 / 2.0) ** 2
        Z_slot  = BABINET / Z_dip.conjugate()
        return Z_slot

    # ------------------------------------------------------------------ #
    # Pattern
    # ------------------------------------------------------------------ #
    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 181) -> "RadiationPattern":
        """3-D radiation pattern.

        Slot pattern is the complement of the dipole: E and H planes
        are swapped.  The θ-polarised field follows the same angular
        dependence as the dipole's φ-polarised field.

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
        kl    = self.k * self.slot.slot_length / 2.0

        sin_t  = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
        e_1d   = (np.cos(kl * np.cos(theta)) - np.cos(kl)) / sin_t

        # Slot: E_φ varies like dipole E_θ (pattern rotated 90° in polarisation)
        E_phi   = np.outer(e_1d, np.ones(n_phi)).astype(complex)
        E_theta = np.zeros_like(E_phi)

        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)

    # ------------------------------------------------------------------ #
    # S11 frequency sweep
    # ------------------------------------------------------------------ #
    def s11(self, Z0: float = 50.0, n_freq: int = 300) -> tuple:
        """S11 vs frequency.

        Parameters
        ----------
        Z0 : float
            Reference impedance [Ω].  Slot is often fed by a coplanar strip
            or balanced line; Z_slot_resonance ≈ 363 Ω.  A balun transforms
            this to 50 Ω.  Set Z0=363 for direct slot analysis.
        n_freq : int
            Number of frequency points.

        Returns
        -------
        tuple  (freq_array [Hz], S11_complex ndarray)
        """
        from scipy.integrate import quad

        half_slot = self.slot.slot_length / 2.0
        f_res     = 0.47 * C0 / (2.0 * half_slot)

        freqs = np.linspace(self.freq * 0.6, self.freq * 1.4, n_freq)
        S11   = np.zeros(n_freq, dtype=complex)
        BABINET = (ETA0 / 2.0) ** 2

        for i, f in enumerate(freqs):
            k  = 2.0 * PI * f / C0
            kl = k * half_slot

            def integrand(t, _kl=kl):
                sin_t = max(abs(np.sin(t)), 1e-12)
                num = np.cos(_kl * np.cos(t)) - np.cos(_kl)
                return (num / sin_t) ** 2 * np.sin(t)

            integral, _ = quad(integrand, 0.0, PI, limit=100)
            Rr_dip  = (ETA0 / (2.0 * PI)) * integral
            delta   = (f - f_res) / (f_res + 1e-30)
            Xin_dip = 2.0 * Rr_dip * delta * 8.0
            Z_dip   = complex(Rr_dip, Xin_dip)
            Z_slot  = BABINET / Z_dip.conjugate()
            S11[i]  = (Z_slot - Z0) / (Z_slot + Z0)

        return freqs, S11
