"""Yagi-Uda antenna geometry."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER
from pylobe.constants import C0, PI
from pylobe.utils.validation import (
    check_frequency, check_positive, check_positive_integer, check_range,
)


class YagiUda(AntennaGeometry):
    """Yagi-Uda endfire array: reflector + driven dipole + N directors.

    Element lengths and spacings follow Viezbicke (NBS TN-688) optimum
    design tables scaled to the design frequency.  For N_directors in
    [1, 15] all element lengths are interpolated from the table; outside
    this range the standard rule-of-thumb is used.

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    N_directors : int
        Number of director elements (1–15 recommended).
    boom_length : float or None
        Total boom length [m].  If None, computed from spacing rules.
    reflector_spacing : float or None
        Spacing from reflector to driven element [m].  Defaults to 0.2λ.
    director_spacing : float or None
        Uniform director-to-director spacing [m].  Defaults to 0.25λ.
    wire_radius : float or None
        Wire radius [m].  Defaults to λ/200.
    conductor_material : Material or None
        Wire/boom conductor.  Defaults to COPPER.

    Attributes
    ----------
    element_lengths : list of float
        [reflector_L, driven_L, dir1_L, dir2_L, …] in metres.
    element_positions : list of float
        Positions along boom (z-axis) in metres.
        Driven element is at z = 0.
    gain_approx_dbi : float
        Approximate gain [dBi] from empirical formula.
    """

    # Viezbicke optimum half-lengths (normalised to λ), N_dir = 1 to 10
    # Format: [driven, dir1, dir2, ..., dirN] — all as fractions of λ
    _VIZ_DRIVEN   = 0.470
    _VIZ_REFL     = 0.500
    _VIZ_DIR_LF = {
        1:  [0.430],
        2:  [0.440, 0.424],
        3:  [0.440, 0.430, 0.415],
        4:  [0.440, 0.435, 0.425, 0.415],
        5:  [0.440, 0.438, 0.434, 0.428, 0.418],
        6:  [0.440, 0.440, 0.437, 0.433, 0.427, 0.418],
        7:  [0.440, 0.440, 0.439, 0.436, 0.431, 0.426, 0.418],
        8:  [0.440, 0.440, 0.440, 0.438, 0.434, 0.430, 0.425, 0.418],
        9:  [0.440, 0.440, 0.440, 0.440, 0.437, 0.433, 0.428, 0.423, 0.416],
        10: [0.440, 0.440, 0.440, 0.440, 0.440, 0.436, 0.431, 0.426, 0.420, 0.416],
    }

    def __init__(self, freq: float, N_directors: int = 3,
                 boom_length: float = None,
                 reflector_spacing: float = None,
                 director_spacing: float = None,
                 wire_radius: float = None,
                 conductor_material: Material = None):
        check_frequency(freq)
        check_positive_integer(N_directors, "N_directors")
        check_range(N_directors, 1, 20, "N_directors")
        if boom_length is not None:
            check_positive(boom_length, "boom_length")
        if reflector_spacing is not None:
            check_positive(reflector_spacing, "reflector_spacing")
        if director_spacing is not None:
            check_positive(director_spacing, "director_spacing")
        if wire_radius is not None:
            check_positive(wire_radius, "wire_radius")

        super().__init__(
            name=f"YagiUda_N{N_directors}",
            material=AIR,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.freq        = freq
        self.N_directors = N_directors
        self.lambda0     = lambda0

        # Element spacings
        self.reflector_spacing = (reflector_spacing if reflector_spacing is not None
                                  else 0.20 * lambda0)
        self.director_spacing  = (director_spacing if director_spacing is not None
                                  else 0.25 * lambda0)

        # Element length factors from Viezbicke table (or extrapolate)
        n = min(N_directors, 10)
        dir_lfs = list(self._VIZ_DIR_LF[n])
        # Pad with last entry if N_directors > 10
        while len(dir_lfs) < N_directors:
            dir_lfs.append(dir_lfs[-1] - 0.002)

        # Physical lengths [m]
        L_refl   = self._VIZ_REFL   * lambda0
        L_driven = self._VIZ_DRIVEN * lambda0
        L_dirs   = [lf * lambda0 for lf in dir_lfs]

        self.element_lengths   = [L_refl, L_driven] + L_dirs
        self.wire_radius       = wire_radius if wire_radius is not None else lambda0 / 200.0
        self.conductor_material = (conductor_material if conductor_material is not None
                                   else COPPER)

        # Boom positions (driven element at z = 0)
        pos_refl = -self.reflector_spacing
        positions = [pos_refl, 0.0]
        for i in range(N_directors):
            positions.append((i + 1) * self.director_spacing)
        self.element_positions = positions

        self.boom_length = abs(positions[0]) + positions[-1]

        # Approximate gain:  G ≈ 10·log10(N_total + 2) + 3.5  (empirical)
        N_total = N_directors + 2
        self.gain_approx_dbi = float(10 * np.log10(N_total) + 3.5)

        # Build vertices: each element is a horizontal dipole at its boom position
        all_verts = []
        all_edges = []
        v_offset = 0
        N_seg = 11  # segments per element
        for i, (pos_z, L) in enumerate(zip(positions, self.element_lengths)):
            half_L = L / 2.0
            y_pts = np.linspace(-half_L, half_L, N_seg + 1)
            verts = np.column_stack([
                np.zeros(N_seg + 1),
                y_pts,
                np.full(N_seg + 1, pos_z),
            ])
            edges = [(v_offset + j, v_offset + j + 1) for j in range(N_seg)]
            all_verts.append(verts)
            all_edges.extend(edges)
            v_offset += N_seg + 1

        self.vertices = np.vstack(all_verts)
        self.edges    = all_edges
        # Feed at centre of driven element (element index 1)
        driven_mid = N_seg + 1 + N_seg // 2
        self.feed_point = self.vertices[driven_mid].copy()
        self.feed_point[1] = 0.0   # y=0 (centre of driven element)

    def __repr__(self) -> str:
        return (
            f"YagiUda\n"
            f"  Frequency       : {self.freq / 1e6:.2f} MHz\n"
            f"  N_directors     : {self.N_directors}\n"
            f"  Boom length     : {self.boom_length * 1e3:.1f} mm\n"
            f"  Reflector L     : {self.element_lengths[0] * 1e3:.1f} mm\n"
            f"  Driven L        : {self.element_lengths[1] * 1e3:.1f} mm\n"
            f"  Director 1 L    : {self.element_lengths[2] * 1e3:.1f} mm\n"
            f"  Approx gain     : {self.gain_approx_dbi:.1f} dBi\n"
        )
