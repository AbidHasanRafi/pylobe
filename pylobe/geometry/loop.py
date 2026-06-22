"""Loop antenna geometry classes (small loop and resonant large loop)."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER
from pylobe.constants import C0, PI
from pylobe.utils.validation import check_frequency, check_positive, check_positive_integer


class SmallLoopAntenna(AntennaGeometry):
    """Electrically small magnetic-dipole loop antenna.

    Valid when the circumference C = 2π·r ≪ λ (typically C < 0.1λ).
    Radiation pattern identical to a short magnetic dipole (same as
    short electric dipole rotated 90°):
        U(θ) ∝ sin²θ → peak broadside, null on axis.
    Gain ≈ 1.76 dBi (same as short dipole).
    Radiation resistance:  Rr = 20·(C/λ)⁴  Ω.

    For enhanced efficiency the loop is wound with N turns:
        Rr_N = N² · Rr_single.

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    radius : float
        Loop radius [m].
    N_turns : int
        Number of turns (single-turn = 1).
    wire_radius : float or None
        Wire radius [m]. Defaults to radius/100.
    conductor_material : Material or None
        Wire conductor. Defaults to COPPER.
    """

    def __init__(self, freq: float, radius: float, N_turns: int = 1,
                 wire_radius: float = None,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_positive(radius, "radius")
        check_positive_integer(N_turns, "N_turns")
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")
        super().__init__(
            name="SmallLoopAntenna",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.freq         = freq
        self.radius       = radius
        self.N_turns      = N_turns
        self.circumference = 2.0 * PI * radius
        self.wire_radius  = wire_radius if wire_radius is not None else radius / 100.0
        self.conductor_material = (conductor_material if conductor_material is not None
                                   else COPPER)

        # Electrically small criterion warning
        self.is_small = self.circumference < 0.1 * lambda0

        # Radiation resistance (Balanis Eq. 5-29 for electrically small loop)
        C_over_lam = self.circumference / lambda0
        self.Rr_single = 20.0 * PI ** 2 * C_over_lam ** 4
        self.Rr        = N_turns ** 2 * self.Rr_single
        self.gain_approx_dbi = 1.76   # same as short dipole

        # Reactance: approximately inductive (XL = ωL)
        mu0 = 4e-7 * PI
        # Self-inductance of a thin circular loop (Wheeler's formula)
        L_ind = mu0 * radius * (np.log(8 * radius / self.wire_radius) - 2.0) * N_turns ** 2
        self.L_inductance = L_ind
        self.Xin = 2.0 * PI * freq * L_ind

        # Build circular wire segments
        N_pts = 72
        phi = np.linspace(0, 2 * PI, N_pts, endpoint=False)
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = np.zeros(N_pts)
        self.vertices = np.column_stack([x, y, z])
        self.edges    = [(i, (i + 1) % N_pts) for i in range(N_pts)]
        self.feed_point = np.array([radius, 0.0, 0.0])

    def __repr__(self) -> str:
        lam = C0 / self.freq
        return (
            f"SmallLoopAntenna\n"
            f"  Frequency       : {self.freq/1e6:.2f} MHz\n"
            f"  Radius          : {self.radius*1e3:.2f} mm  "
            f"(C/λ = {self.circumference/(lam):.3f})\n"
            f"  Turns N         : {self.N_turns}\n"
            f"  Rr (N² · Rr1)  : {self.Rr:.4f} Ω\n"
            f"  Inductance L    : {self.L_inductance*1e9:.2f} nH\n"
            f"  Gain (approx)   : {self.gain_approx_dbi:.2f} dBi\n"
        )


class LargeLoopAntenna(AntennaGeometry):
    """Resonant large loop antenna (circumference ≈ λ).

    At resonance (C = λ) the loop behaves like a folded dipole:
        Z_in ≈ 100–200 Ω, gain ≈ 3–4 dBi broadside.
    For C = 2λ (two-wavelength loop): gain ≈ 3.5 dBi.

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    circumference_factor : float
        Loop circumference as a fraction of λ.  Default 1.0 (1λ loop).
    wire_radius : float or None
        Wire radius [m]. Defaults to circumference/200.
    N_pts : int
        Number of geometry discretisation points around the loop.
    conductor_material : Material or None
        Wire conductor. Defaults to COPPER.
    """

    def __init__(self, freq: float, circumference_factor: float = 1.0,
                 wire_radius: float = None, N_pts: int = 72,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_positive(circumference_factor, "circumference_factor")
        check_positive_integer(N_pts, "N_pts")
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")
        super().__init__(
            name="LargeLoopAntenna",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.freq                = freq
        self.circumference_factor = circumference_factor
        self.circumference       = circumference_factor * lambda0
        self.radius              = self.circumference / (2.0 * PI)
        self.wire_radius         = (wire_radius if wire_radius is not None
                                    else self.circumference / 200.0)
        self.conductor_material  = (conductor_material if conductor_material is not None
                                    else COPPER)

        # Approximate gain (large loop, Balanis Fig. 5-19)
        # C/λ = 1: G ≈ 3.4 dBi;  C/λ = 2: G ≈ 4.5 dBi
        cf = circumference_factor
        if cf <= 1.0:
            self.gain_approx_dbi = 1.76 + 5.0 * cf * cf  # smooth interpolation
        else:
            self.gain_approx_dbi = 3.4 + 1.1 * np.log10(cf)

        # Build loop vertices
        phi = np.linspace(0, 2 * PI, N_pts, endpoint=False)
        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)
        z = np.zeros(N_pts)
        self.vertices = np.column_stack([x, y, z])
        self.edges    = [(i, (i + 1) % N_pts) for i in range(N_pts)]
        self.feed_point = np.array([self.radius, 0.0, 0.0])

    def __repr__(self) -> str:
        return (
            f"LargeLoopAntenna\n"
            f"  Frequency       : {self.freq/1e6:.2f} MHz\n"
            f"  Circumference   : {self.circumference*1e3:.1f} mm  "
            f"({self.circumference_factor:.2f}λ)\n"
            f"  Radius          : {self.radius*1e3:.1f} mm\n"
            f"  Gain (approx)   : {self.gain_approx_dbi:.2f} dBi\n"
        )
