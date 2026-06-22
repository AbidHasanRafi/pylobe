"""Monopole antenna geometry classes."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER, PEC, ALUMINUM
from pylobe.constants import C0, PI
from pylobe.utils.validation import (
    check_frequency, check_positive, check_positive_integer, check_range,
)


class QuarterWaveMonopole(AntennaGeometry):
    """Quarter-wave monopole over an infinite (modelled) ground plane.

    Physical length L = λ/4.
    Input impedance ≈ 36.5 Ω (half of half-wave dipole by image theory).
    Gain = 5.16 dBi (3 dB over half-wave dipole).

    The ground plane is at z = 0; the monopole extends along +z.

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    length_factor : float
        Fraction of λ/4, default 1.0 (exact quarter-wave).
    wire_radius : float or None
        Wire radius [m]. Defaults to L/200.
    ground_radius : float or None
        Finite ground plane radius [m]. Defaults to λ/2.
    N_segments : int
        MoM discretisation segments.
    conductor_material : Material or None
        Wire conductor material. Defaults to COPPER.
    ground_material : Material or None
        Ground plane material. Defaults to ALUMINUM.
    """

    def __init__(self, freq: float, length_factor: float = 1.0,
                 wire_radius: float = None, ground_radius: float = None,
                 N_segments: int = 11,
                 conductor_material: Material = None,
                 ground_material: Material = None):
        check_frequency(freq)
        check_positive(length_factor, "length_factor")
        check_positive_integer(N_segments, "N_segments")
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")
        if ground_radius is not None:
            check_positive(ground_radius, "ground_radius")
        super().__init__(
            name="QuarterWaveMonopole",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.L = length_factor * lambda0 / 4.0
        self.wire_radius = wire_radius if wire_radius is not None else self.L / 200.0
        self.ground_radius = ground_radius if ground_radius is not None else lambda0 / 2.0
        self.N_segments = N_segments
        self.conductor_material = conductor_material if conductor_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else ALUMINUM

        # Image method: effective dipole length = 2L
        self.effective_dipole_length = 2.0 * self.L
        self.impedance_approx = complex(36.5, 0.0)
        self.gain_dbi = 5.16

        # Wire vertices along z axis, z=0 to z=L
        z_pts = np.linspace(0.0, self.L, N_segments + 1)
        wire_verts = np.column_stack([
            np.zeros(N_segments + 1),
            np.zeros(N_segments + 1),
            z_pts,
        ])

        # Ground plane disc (N_gnd points around circumference)
        N_gnd = 64
        phi_gnd = np.linspace(0, 2 * PI, N_gnd, endpoint=False)
        gnd_verts = np.column_stack([
            self.ground_radius * np.cos(phi_gnd),
            self.ground_radius * np.sin(phi_gnd),
            np.zeros(N_gnd),
        ])
        self.vertices = np.vstack([wire_verts, gnd_verts])
        self.edges = [(i, i + 1) for i in range(N_segments)]
        # Ground plane ring edges
        n0 = N_segments + 1
        self.edges += [(n0 + i, n0 + (i + 1) % N_gnd) for i in range(N_gnd)]
        self.feed_point = np.array([0.0, 0.0, 0.0])


class HelicalMonopole(AntennaGeometry):
    """Helical antenna operating in normal or axial mode.

    Normal mode (D ≪ λ): broadside radiation, like short dipole.
    Axial mode  (C ≈ λ, S ≈ λ/4): end-fire, circular polarisation.
    Axial mode gain: G ≈ 11.8 N C² S / λ³  (approximate, Kraus formula).

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    N_turns : int
        Number of helical turns.
    diameter : float
        Helix diameter [m].
    pitch_angle : float
        Pitch angle [degrees]. Determines turn spacing.
    mode : str
        'normal' or 'axial'.
    N_pts_per_turn : int
        Discretisation points per turn.
    conductor_material : Material or None
        Helix wire material. Defaults to COPPER.
    ground_material : Material or None
        Ground plane material. Defaults to ALUMINUM.
    ground_radius : float or None
        Ground plane radius [m]. Defaults to λ/2.
    """

    def __init__(self, freq: float, N_turns: int, diameter: float,
                 pitch_angle: float, mode: str = 'axial',
                 N_pts_per_turn: int = 32,
                 conductor_material: Material = None,
                 ground_material: Material = None,
                 ground_radius: float = None):
        check_frequency(freq)
        check_positive_integer(N_turns, "N_turns")
        check_positive(diameter, "diameter")
        check_range(pitch_angle, 1.0, 89.0, "pitch_angle")
        check_positive_integer(N_pts_per_turn, "N_pts_per_turn")
        if mode not in ('normal', 'axial'):
            raise ValueError(f"mode must be 'normal' or 'axial', got {mode!r}")
        if ground_radius is not None:
            check_positive(ground_radius, "ground_radius")
        super().__init__(
            name=f"HelicalMonopole_{mode}",
            material=AIR,
            freq_design=freq,
        )
        self.freq = freq
        self.N_turns = N_turns
        self.diameter = diameter
        self.pitch_angle = pitch_angle
        self.mode = mode
        self.conductor_material = conductor_material if conductor_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else ALUMINUM
        lambda0 = C0 / freq
        self.ground_radius = ground_radius if ground_radius is not None else lambda0 / 2.0

        # Helix geometry
        C = PI * diameter          # Circumference [m]
        pitch_rad = np.deg2rad(pitch_angle)
        S = C * np.tan(pitch_rad)  # Turn spacing (axial distance) [m]
        self.circumference = C
        self.turn_spacing = S
        self.total_length = N_turns * S

        # Approximate gain (axial mode)
        if mode == 'axial':
            self.gain_approx_dbi = 10 * np.log10(
                11.8 * N_turns * C**2 * S / lambda0**3
            ) if N_turns > 0 else 0.0
        else:
            self.gain_approx_dbi = 1.76  # ≈ short dipole

        # Generate helix vertices parametrically
        t = np.linspace(0, 2 * PI * N_turns, N_turns * N_pts_per_turn + 1)
        x = (diameter / 2.0) * np.cos(t)
        y = (diameter / 2.0) * np.sin(t)
        z = (S / (2 * PI)) * t

        helix_verts = np.column_stack([x, y, z])

        # Ground plane disc
        N_gnd = 64
        phi_gnd = np.linspace(0, 2 * PI, N_gnd, endpoint=False)
        gnd_verts = np.column_stack([
            self.ground_radius * np.cos(phi_gnd),
            self.ground_radius * np.sin(phi_gnd),
            np.zeros(N_gnd),
        ])

        n_helix = len(helix_verts)
        self.vertices = np.vstack([helix_verts, gnd_verts])
        self.edges = [(i, i + 1) for i in range(n_helix - 1)]
        # Ground ring edges
        self.edges += [(n_helix + i, n_helix + (i + 1) % N_gnd) for i in range(N_gnd)]
        self.feed_point = np.array([diameter / 2.0, 0.0, 0.0])
