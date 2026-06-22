"""Dipole antenna geometry classes."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER
from pylobe.constants import C0, PI
from pylobe.utils.validation import (
    check_frequency, check_length_factor, check_positive,
    check_positive_integer,
)


class HalfWaveDipole(AntennaGeometry):
    """Half-wave dipole antenna.

    Total length L = length_factor × λ (default 0.47λ for resonance).
    Approximate input impedance at resonance: Z_in ≈ 73 + j42.5 Ω.

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    length_factor : float
        Length as fraction of wavelength. 0.47 gives near-resonance.
    wire_radius : float or None
        Wire radius [m]. Defaults to L/200.
    N_segments : int
        Number of MoM discretisation segments (odd recommended).
    conductor_material : Material or None
        Wire conductor material. Defaults to COPPER.
    """

    def __init__(self, freq: float, length_factor: float = 0.47,
                 wire_radius: float = None, N_segments: int = 21,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_length_factor(length_factor)
        check_positive_integer(N_segments, "N_segments")
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")
        super().__init__(
            name="HalfWaveDipole",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.L_total = length_factor * lambda0
        self.arm_length = self.L_total / 2.0
        self.wire_radius = wire_radius if wire_radius is not None else self.L_total / 200.0
        self.N_segments = N_segments
        self.length_factor = length_factor
        self.impedance_approx = complex(73.1, 42.5)  # Ω, half-wave
        self.conductor_material = conductor_material if conductor_material is not None else COPPER

        # Discretise wire along z-axis from -arm_length to +arm_length
        z_points = np.linspace(-self.arm_length, self.arm_length, N_segments + 1)
        self.vertices = np.column_stack([
            np.zeros(N_segments + 1),
            np.zeros(N_segments + 1),
            z_points,
        ])
        self.edges = [(i, i + 1) for i in range(N_segments)]
        self.feed_point = np.array([0.0, 0.0, 0.0])

    @property
    def segment_length(self) -> float:
        """Length of each MoM segment [m]."""
        return self.L_total / self.N_segments


class FoldedDipole(AntennaGeometry):
    """Folded dipole antenna.

    Input impedance ≈ 4 × Z_dipole ≈ 292 Ω at resonance.
    Bandwidth approximately 10× that of a simple dipole.

    Parameters
    ----------
    freq : float
        Operating frequency [Hz].
    length_factor : float
        Total length as fraction of wavelength (default 0.47).
    wire_radius : float or None
        Wire radius [m].
    spacing : float or None
        Separation between the two parallel wires [m]. Defaults to λ/100.
    conductor_material : Material or None
        Wire conductor material. Defaults to COPPER.
    """

    def __init__(self, freq: float, length_factor: float = 0.47,
                 wire_radius: float = None, spacing: float = None,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_length_factor(length_factor)
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")
        if spacing is not None:
            check_positive(spacing, "spacing")
        super().__init__(
            name="FoldedDipole",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        L = length_factor * lambda0
        self.L_total = L
        self.arm_length = L / 2.0
        self.wire_radius = wire_radius if wire_radius is not None else L / 200.0
        self.spacing = spacing if spacing is not None else lambda0 / 100.0
        self.conductor_material = conductor_material if conductor_material is not None else COPPER

        d = self.spacing
        # Upper wire: z from -L/2 to +L/2 at y=0
        # Lower wire: z from -L/2 to +L/2 at y=d
        # Connecting segments at both ends
        N = 21
        z_pts = np.linspace(-L / 2.0, L / 2.0, N)
        upper = np.column_stack([np.zeros(N), np.zeros(N), z_pts])
        lower = np.column_stack([np.zeros(N), np.full(N, d), z_pts])
        conn_left  = np.array([[0.0, 0.0, -L/2.0], [0.0, d, -L/2.0]])
        conn_right = np.array([[0.0, 0.0,  L/2.0], [0.0, d,  L/2.0]])
        self.vertices = np.vstack([upper, lower, conn_left, conn_right])

        # Edges: upper wire, lower wire, connectors
        self.edges = (
            [(i, i + 1) for i in range(N - 1)]                  # upper
            + [(N + i, N + i + 1) for i in range(N - 1)]        # lower
            + [(2*N, 2*N + 1), (2*N + 2, 2*N + 3)]              # connectors
        )
        self.impedance_approx = complex(292.0, 0.0)
        self.feed_point = np.array([0.0, 0.0, 0.0])


class BowTieDipole(AntennaGeometry):
    """Bow-tie dipole antenna.

    Planar triangular arms give ultra-wide bandwidth.
    The flare angle determines the input impedance and bandwidth.

    Parameters
    ----------
    freq : float
        Centre frequency [Hz].
    arm_length : float or None
        Length of each triangular arm [m]. Defaults to λ/4.
    flare_angle_deg : float
        Half-flare angle of each arm [degrees]. Typically 30–60°.
    N_pts : int
        Points along each arm edge for discretisation.
    conductor_material : Material or None
        Arm conductor material. Defaults to COPPER.
    """

    def __init__(self, freq: float, arm_length: float = None,
                 flare_angle_deg: float = 45.0, N_pts: int = 10,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_positive_integer(N_pts, "N_pts")
        if arm_length is not None:
            check_positive(arm_length, "arm_length")
        from pylobe.utils.validation import check_range
        check_range(flare_angle_deg, 1.0, 89.0, "flare_angle_deg")
        super().__init__(
            name="BowTieDipole",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.arm_length = arm_length if arm_length is not None else lambda0 / 4.0
        self.flare_angle_rad = np.deg2rad(flare_angle_deg)
        self.flare_angle_deg = flare_angle_deg
        self.conductor_material = conductor_material if conductor_material is not None else COPPER

        L = self.arm_length
        half_w = L * np.tan(self.flare_angle_rad)

        # Right arm: tip at feed (0,0,0), base at (L, ±half_w, 0)
        right_verts = np.array([
            [0.0,      0.0,      0.0],
            [L,        half_w,   0.0],
            [L,       -half_w,   0.0],
        ])
        # Left arm: tip at feed, base at (-L, ±half_w, 0)
        left_verts = np.array([
            [0.0,      0.0,      0.0],
            [-L,       half_w,   0.0],
            [-L,      -half_w,   0.0],
        ])
        # Vertices: right_top, right_bottom, left_top, left_bottom, feed
        self.vertices = np.vstack([right_verts[1:], left_verts[1:],
                                   np.array([[0.0, 0.0, 0.0]])])
        self.faces = [
            [4, 0, 1],   # right arm triangle
            [4, 2, 3],   # left arm triangle
        ]
        # Outline edges
        self.edges = [
            (4, 0), (0, 1), (1, 4),   # right arm outline
            (4, 2), (2, 3), (3, 4),   # left arm outline
        ]
        self.feed_point = np.array([0.0, 0.0, 0.0])

    @property
    def length_factor(self) -> float:
        """Effective length factor for bow-tie arm (fixed at 0.5).

        Bow-tie antennas lack a single resonant length-factor like linear
        dipoles.  We return 0.5 (half-wavelength resonance) so that
        DipoleSolver.s11() and input_impedance() can operate on a bow-tie.
        """
        return 0.5
