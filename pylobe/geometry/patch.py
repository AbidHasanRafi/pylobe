"""Microstrip patch antenna geometries."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, FR4, COPPER, PEC
from pylobe.constants import C0, PI
from pylobe.utils.validation import (
    check_frequency, check_eps_r, check_positive, check_substrate_thickness,
)


class RectangularPatch(AntennaGeometry):
    """Rectangular microstrip patch antenna.

    Physics-correct dimensions derived from transmission-line model.
    Reference: Balanis, *Antenna Theory*, 4th Ed., Ch. 14.

    Parameters
    ----------
    freq : float
        Design centre frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness [m].
    loss_tangent : float
        Substrate loss tangent.
    feed_z : float
        Target feed impedance [Ω].
    inset_feed : bool
        If True, compute inset feed position for impedance match.
    substrate_material : Material or None
        Substrate material. If None, a material is built from eps_r/loss_tangent.
    patch_material : Material or None
        Conductor material for the patch layer. Defaults to COPPER.
    ground_material : Material or None
        Conductor material for the ground plane. Defaults to PEC.
    """

    def __init__(self, freq: float, eps_r: float = 4.4, h: float = 1.6e-3,
                 loss_tangent: float = 0.02, feed_z: float = 50.0,
                 inset_feed: bool = True,
                 substrate_material: Material = None,
                 patch_material: Material = None,
                 ground_material: Material = None):

        check_frequency(freq)
        check_positive(h, "h (substrate thickness)")
        if substrate_material is not None:
            check_eps_r(substrate_material.eps_r, "substrate_material.eps_r")
            material = substrate_material
            eps_r = substrate_material.eps_r
            loss_tangent = substrate_material.loss_tangent
        else:
            check_eps_r(eps_r)
            check_positive(1.0 - loss_tangent + 1e-9, "loss_tangent must be < 1")
            material = Material(
                name=f"Substrate_eps{eps_r}",
                eps_r=eps_r,
                loss_tangent=loss_tangent,
                color=FR4.color,
            )
        check_substrate_thickness(h, freq, eps_r)
        if feed_z <= 0:
            raise ValueError(f"feed_z must be positive, got {feed_z!r}")

        super().__init__(
            name="RectangularPatch",
            material=material,
            freq_design=freq,
            feed_impedance=feed_z,
        )
        self.h = h
        self.freq = freq
        self.eps_r = eps_r
        self.inset_feed = inset_feed

        # Component materials
        self.substrate_material = material
        self.patch_material = patch_material if patch_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else PEC

        # ---- Design equations (Balanis Ch. 14) ----
        lambda0 = C0 / freq
        k0 = 2.0 * PI * freq / C0

        # Step 1: Patch width
        self.W = (C0 / (2.0 * freq)) * np.sqrt(2.0 / (eps_r + 1.0))

        # Step 2: Effective permittivity
        self.eps_eff = (
            (eps_r + 1.0) / 2.0
            + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / self.W) ** (-0.5)
        )

        # Step 3: Length extension (fringe effect)
        self.delta_L = (
            0.412 * h
            * (self.eps_eff + 0.3) * (self.W / h + 0.264)
            / ((self.eps_eff - 0.258) * (self.W / h + 0.8))
        )

        # Step 4: Effective length
        L_eff = C0 / (2.0 * freq * np.sqrt(self.eps_eff))

        # Step 5: Physical patch length
        self.L = L_eff - 2.0 * self.delta_L

        # Step 6: Radiation conductance and edge impedance
        self.G1 = (self.W / (120.0 * lambda0)) * (1.0 - (k0 * h) ** 2 / 24.0)
        self.Rin_edge = 1.0 / (2.0 * self.G1)

        # Step 7: Inset feed position for impedance match
        if inset_feed:
            ratio = np.clip(feed_z / self.Rin_edge, 0.0, 1.0)
            self.y0 = (self.L / PI) * np.arccos(np.sqrt(ratio))
        else:
            self.y0 = 0.0

        # Build geometry vertices (patch rectangle + substrate box)
        self._build_vertices()

    def _build_vertices(self):
        """Build vertex array: patch corners + substrate bottom corners."""
        W, L, h = self.W, self.L, self.h

        # Patch surface (z = h): 4 corners in x-y plane
        patch_verts = np.array([
            [0.0,  0.0,  h],
            [W,    0.0,  h],
            [W,    L,    h],
            [0.0,  L,    h],
        ])

        # Ground plane (z = 0): 4 corners
        ground_verts = np.array([
            [0.0,  0.0,  0.0],
            [W,    0.0,  0.0],
            [W,    L,    0.0],
            [0.0,  L,    0.0],
        ])

        self.vertices = np.vstack([patch_verts, ground_verts])

        # Faces: patch top, ground bottom, and 4 side walls
        self.faces = [
            [0, 1, 2, 3],        # patch
            [4, 5, 6, 7],        # ground
            [0, 1, 5, 4],        # side
            [1, 2, 6, 5],        # side
            [2, 3, 7, 6],        # side
            [3, 0, 4, 7],        # side
        ]

        # Edges (patch outline)
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        # Feed point
        if self.inset_feed:
            self.feed_point = np.array([W / 2.0, self.y0, h])
        else:
            self.feed_point = np.array([W / 2.0, 0.0, h])

    @property
    def resonant_frequency(self) -> float:
        """Resonant frequency from effective length [Hz]."""
        return C0 / (2.0 * (self.L + 2.0 * self.delta_L) * np.sqrt(self.eps_eff))

    @property
    def bandwidth_approx(self) -> float:
        """Approximate impedance bandwidth (fraction of centre frequency).

        BW ≈ 3.77 * (εr-1)/εr² * (W/L) * (h/λ0)
        """
        lambda0 = C0 / self.freq
        return (3.77 * (self.eps_r - 1.0) / self.eps_r ** 2
                * (self.W / self.L) * (self.h / lambda0))

    def summary(self) -> dict:
        """Return design parameter summary."""
        return {
            "W_mm":               self.W * 1e3,
            "L_mm":               self.L * 1e3,
            "h_mm":               self.h * 1e3,
            "eps_eff":            self.eps_eff,
            "delta_L_mm":         self.delta_L * 1e3,
            "Rin_edge_ohm":       self.Rin_edge,
            "y0_mm":              self.y0 * 1e3,
            "BW_percent":         self.bandwidth_approx * 100,
            "f_r_GHz":            self.resonant_frequency / 1e9,
            "substrate_material": self.substrate_material.name,
            "patch_material":     self.patch_material.name,
            "ground_material":    self.ground_material.name,
        }


class CircularPatch(AntennaGeometry):
    """Circular microstrip patch antenna.

    Radius from dominant TM11 mode:
        a = 8.791e9 / (f0 * sqrt(eps_r))  [cm]
    with fringe correction for effective radius.

    Parameters
    ----------
    freq : float
        Design centre frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness [m].
    loss_tangent : float
        Substrate loss tangent.
    substrate_material : Material or None
        Override substrate material (eps_r/loss_tangent taken from here if set).
    patch_material : Material or None
        Conductor for the patch layer. Defaults to COPPER.
    ground_material : Material or None
        Conductor for the ground plane. Defaults to PEC.
    """

    def __init__(self, freq: float, eps_r: float = 4.4, h: float = 1.6e-3,
                 loss_tangent: float = 0.02,
                 substrate_material: Material = None,
                 patch_material: Material = None,
                 ground_material: Material = None):

        check_frequency(freq)
        check_positive(h, "h (substrate thickness)")
        if substrate_material is not None:
            check_eps_r(substrate_material.eps_r, "substrate_material.eps_r")
            material = substrate_material
            eps_r = substrate_material.eps_r
            loss_tangent = substrate_material.loss_tangent
        else:
            check_eps_r(eps_r)
            material = Material(
                name=f"Substrate_eps{eps_r}",
                eps_r=eps_r,
                loss_tangent=loss_tangent,
                color=FR4.color,
            )
        check_substrate_thickness(h, freq, eps_r)

        super().__init__(name="CircularPatch", material=material,
                         freq_design=freq)
        self.h = h
        self.eps_r = eps_r
        self.substrate_material = material
        self.patch_material = patch_material if patch_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else PEC

        # Physical radius [m] from design chart formula
        a_cm = 8.791e9 / (freq * np.sqrt(eps_r))
        a = a_cm * 1e-2

        # Fringe-corrected effective radius
        self.a_eff = a * np.sqrt(
            1.0 + (2.0 * h) / (PI * eps_r * a)
            * (np.log(PI * a / (2.0 * h)) + 1.7726)
        )
        self.a = a

        # Vertices: approximate circle with N_seg points
        N_seg = 64
        phi_arr = np.linspace(0, 2 * PI, N_seg, endpoint=False)
        patch_top = np.column_stack([
            a * np.cos(phi_arr),
            a * np.sin(phi_arr),
            np.full(N_seg, h),
        ])
        ground = patch_top.copy()
        ground[:, 2] = 0.0
        self.vertices = np.vstack([patch_top, ground])
        self.edges = [(i, (i + 1) % N_seg) for i in range(N_seg)]
        self.feed_point = np.array([a * 0.5, 0.0, h])

    @property
    def radius(self) -> float:
        """Physical radius [m] (alias for ``a``)."""
        return self.a


class AnnularRingPatch(AntennaGeometry):
    """Annular ring microstrip patch antenna.

    Parameters
    ----------
    freq : float
        Design centre frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness [m].
    inner_radius : float
        Inner ring radius [m].
    outer_radius : float
        Outer ring radius [m].
    substrate_material : Material or None
        Override substrate material.
    patch_material : Material or None
        Conductor for ring patch. Defaults to COPPER.
    ground_material : Material or None
        Conductor for ground plane. Defaults to PEC.
    """

    def __init__(self, freq: float, eps_r: float = 4.4, h: float = 1.6e-3,
                 inner_radius: float = None, outer_radius: float = None,
                 substrate_material: Material = None,
                 patch_material: Material = None,
                 ground_material: Material = None):

        check_frequency(freq)
        check_positive(h, "h (substrate thickness)")
        if substrate_material is not None:
            check_eps_r(substrate_material.eps_r, "substrate_material.eps_r")
            material = substrate_material
            eps_r = substrate_material.eps_r
        else:
            check_eps_r(eps_r)
            material = Material(
                name=f"Substrate_eps{eps_r}",
                eps_r=eps_r,
                color=FR4.color,
            )

        super().__init__(name="AnnularRingPatch", material=material,
                         freq_design=freq)
        self.h = h
        self.eps_r = eps_r
        self.substrate_material = material
        self.patch_material = patch_material if patch_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else PEC

        # Default radii if not provided
        lambda0 = C0 / freq
        self.outer_radius = outer_radius if outer_radius is not None else lambda0 / (4.0 * np.sqrt(eps_r))
        self.inner_radius = inner_radius if inner_radius is not None else self.outer_radius * 0.5
        if self.outer_radius <= 0:
            raise ValueError(f"outer_radius must be positive, got {self.outer_radius!r}")
        if self.inner_radius <= 0:
            raise ValueError(f"inner_radius must be positive, got {self.inner_radius!r}")
        if self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"inner_radius ({self.inner_radius*1e3:.2f} mm) must be less than "
                f"outer_radius ({self.outer_radius*1e3:.2f} mm)"
            )

        N_seg = 64
        phi_arr = np.linspace(0, 2 * PI, N_seg, endpoint=False)

        outer_top = np.column_stack([
            self.outer_radius * np.cos(phi_arr),
            self.outer_radius * np.sin(phi_arr),
            np.full(N_seg, h),
        ])
        inner_top = np.column_stack([
            self.inner_radius * np.cos(phi_arr),
            self.inner_radius * np.sin(phi_arr),
            np.full(N_seg, h),
        ])
        self.vertices = np.vstack([outer_top, inner_top])
        self.edges = (
            [(i, (i + 1) % N_seg) for i in range(N_seg)]
            + [(N_seg + i, N_seg + (i + 1) % N_seg) for i in range(N_seg)]
        )
        self.feed_point = np.array([
            (self.inner_radius + self.outer_radius) / 2.0, 0.0, h
        ])


class ESlotPatch(AntennaGeometry):
    """E-shaped slot patch for dual-band operation.

    The patch has two parallel slots cut into it, creating an E-shape
    that supports two closely spaced resonances.

    Parameters
    ----------
    freq1, freq2 : float
        Two target frequencies [Hz].
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness [m].
    substrate_material : Material or None
        Override substrate material.
    patch_material : Material or None
        Conductor for patch. Defaults to COPPER.
    ground_material : Material or None
        Conductor for ground plane. Defaults to PEC.
    """

    def __init__(self, freq1: float, freq2: float,
                 eps_r: float = 4.4, h: float = 1.6e-3,
                 substrate_material: Material = None,
                 patch_material: Material = None,
                 ground_material: Material = None):

        check_frequency(freq1, "freq1")
        check_frequency(freq2, "freq2")
        check_positive(h, "h (substrate thickness)")
        if freq1 >= freq2:
            raise ValueError(
                f"freq1 ({freq1/1e9:.3f} GHz) must be less than "
                f"freq2 ({freq2/1e9:.3f} GHz)"
            )
        f_centre = (freq1 + freq2) / 2.0

        if substrate_material is not None:
            check_eps_r(substrate_material.eps_r, "substrate_material.eps_r")
            material = substrate_material
            eps_r = substrate_material.eps_r
        else:
            check_eps_r(eps_r)
            material = Material(
                name=f"Substrate_eps{eps_r}",
                eps_r=eps_r,
                color=FR4.color,
            )

        super().__init__(name="ESlotPatch", material=material,
                         freq_design=f_centre)
        self.h = h
        self.freq1 = freq1
        self.freq2 = freq2
        self.substrate_material = material
        self.patch_material = patch_material if patch_material is not None else COPPER
        self.ground_material = ground_material if ground_material is not None else PEC

        # Base patch sized for f_centre
        self.W = (C0 / (2.0 * f_centre)) * np.sqrt(2.0 / (eps_r + 1.0))
        eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (
            1.0 + 12.0 * h / self.W) ** (-0.5)
        delta_L = 0.412 * h * (eps_eff + 0.3) * (self.W / h + 0.264) / (
            (eps_eff - 0.258) * (self.W / h + 0.8))
        self.L = C0 / (2.0 * f_centre * np.sqrt(eps_eff)) - 2.0 * delta_L

        # Slot dimensions: horizontal slots forming the E
        self.slot_w = self.W * 0.6
        self.slot_h = self.L * 0.05
        self.slot_gap = self.L * 0.2

        # Build patch outline vertices (rectangular patch + slot markers)
        W, L = self.W, self.L
        self.vertices = np.array([
            [0.0, 0.0, h], [W, 0.0, h], [W, L, h], [0.0, L, h],
            [0.0, 0.0, 0.0], [W, 0.0, 0.0], [W, L, 0.0], [0.0, L, 0.0],
        ])
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        self.feed_point = np.array([W / 2.0, 0.0, h])
