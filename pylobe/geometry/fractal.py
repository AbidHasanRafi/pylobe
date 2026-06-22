"""Fractal antenna geometry classes (Koch, Sierpinski, Minkowski)."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR
from pylobe.constants import C0, PI


def _koch_segment(p1: np.ndarray, p2: np.ndarray) -> list:
    """Split one segment into 4 Koch sub-segments.

    Returns list of 5 points [p1, A, B, C, p2] where:
        A = p1 + (p2-p1)/3
        B = midpoint of equilateral triangle on middle third
        C = p1 + 2*(p2-p1)/3
    """
    d = (p2 - p1)
    A = p1 + d / 3.0
    C = p1 + 2.0 * d / 3.0

    # Rotate d/3 by 60° to get apex B
    rot60 = np.array([[np.cos(PI/3), -np.sin(PI/3), 0],
                       [np.sin(PI/3),  np.cos(PI/3), 0],
                       [0,             0,             1]])
    B = A + rot60 @ (d / 3.0)
    return [p1, A, B, C, p2]


def _koch_curve(p1: np.ndarray, p2: np.ndarray, iterations: int) -> np.ndarray:
    """Recursively generate Koch curve between p1 and p2.

    Returns array of points along the curve (endpoints included).
    """
    if iterations == 0:
        return np.array([p1, p2])
    pts = [p1, _mid_a(p1, p2), _apex(p1, p2), _mid_c(p1, p2), p2]

    d = p2 - p1
    A = p1 + d / 3.0
    C = p1 + 2.0 * d / 3.0
    rot = np.array([[np.cos(PI/3), -np.sin(PI/3), 0],
                    [np.sin(PI/3),  np.cos(PI/3), 0],
                    [0,             0,             1]])
    B = A + rot @ (d / 3.0)

    sub_segs = [
        (p1, A),
        (A,  B),
        (B,  C),
        (C,  p2),
    ]
    result = []
    for s1, s2 in sub_segs:
        pts_sub = _koch_curve(s1, s2, iterations - 1)
        result.append(pts_sub[:-1])  # avoid duplicate endpoint
    result.append(np.array([p2]))
    return np.vstack(result)


def _mid_a(p1, p2):
    return p1 + (p2 - p1) / 3.0

def _mid_c(p1, p2):
    return p1 + 2.0 * (p2 - p1) / 3.0

def _apex(p1, p2):
    d = (p2 - p1)
    A = p1 + d / 3.0
    rot = np.array([[np.cos(PI/3), -np.sin(PI/3), 0],
                    [np.sin(PI/3),  np.cos(PI/3), 0],
                    [0,             0,             1]])
    return A + rot @ (d / 3.0)


class KochDipole(AntennaGeometry):
    """Koch fractal dipole antenna.

    Iteration 0: straight wire (standard dipole).
    Each Koch iteration replaces every segment with a 4-sub-segment Koch curve.
    Self-similarity dimension: D = log(4)/log(3) ≈ 1.26.
    Miniaturisation factor per iteration: ≈ 1.29 (shorter effective wavelength).

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    iterations : int
        Number of Koch iterations (0 = straight dipole, 3–4 typical).
    length_factor : float
        Total length as fraction of λ. Default 0.47.
    """

    def __init__(self, freq: float, iterations: int = 3,
                 length_factor: float = 0.47):
        super().__init__(
            name=f"KochDipole_iter{iterations}",
            material=AIR,
            freq_design=freq,
        )
        self.iterations = iterations
        self.length_factor = length_factor
        lambda0 = C0 / freq
        L_half = length_factor * lambda0 / 2.0

        # Upper arm: from feed (0,0,0) to tip (0,0,+L_half)
        upper = _koch_curve(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, L_half]),
            iterations,
        )
        # Lower arm: from feed to tip (0,0,-L_half)
        lower = _koch_curve(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -L_half]),
            iterations,
        )

        # Combine (avoid duplicate feed point)
        self.vertices = np.vstack([np.flipud(lower[1:]), upper])
        self.edges = [(i, i+1) for i in range(len(self.vertices)-1)]
        self.feed_point = np.array([0.0, 0.0, 0.0])


def _sierpinski_subdivide(triangles: list, level: int) -> list:
    """Recursively subdivide triangles for Sierpinski gasket.

    Each triangle is split into 4, and the centre sub-triangle is removed.
    """
    if level == 0:
        return triangles
    new_triangles = []
    for tri in triangles:
        v0, v1, v2 = tri
        m01 = (v0 + v1) / 2.0
        m12 = (v1 + v2) / 2.0
        m02 = (v0 + v2) / 2.0
        new_triangles.extend([
            (v0,  m01, m02),
            (m01, v1,  m12),
            (m02, m12, v2),
            # centre triangle (m01, m12, m02) is omitted → aperture
        ])
    return _sierpinski_subdivide(new_triangles, level - 1)


class SierpinskiGasket(AntennaGeometry):
    """Sierpinski gasket monopole antenna.

    Self-similar multiband behaviour; band frequencies scale by log-period ≈ 2.
    Implemented via recursive triangle subdivision.

    Parameters
    ----------
    freq_base : float
        Lowest operating frequency [Hz].
    iterations : int
        Number of Sierpinski iterations (2–5 typical).
    height : float or None
        Triangle height [m]. Defaults to λ_base / 4.
    """

    def __init__(self, freq_base: float, iterations: int = 3,
                 height: float = None):
        super().__init__(
            name=f"SierpinskiGasket_iter{iterations}",
            material=AIR,
            freq_design=freq_base,
        )
        lambda0 = C0 / freq_base
        h_tri = height if height is not None else lambda0 / 4.0
        side = h_tri * 2.0 / np.sqrt(3.0)

        # Initial equilateral triangle
        v0 = np.array([0.0,       0.0, 0.0])
        v1 = np.array([side,      0.0, 0.0])
        v2 = np.array([side/2.0,  h_tri, 0.0])

        triangles = _sierpinski_subdivide([(v0, v1, v2)], iterations)
        self.triangles = triangles
        self.iterations = iterations

        # Collect all unique vertices
        all_verts = []
        for tri in triangles:
            all_verts.extend(tri)
        self.vertices = np.array(all_verts)
        self.faces = [[3*i, 3*i+1, 3*i+2] for i in range(len(triangles))]
        self.feed_point = np.array([side / 2.0, 0.0, 0.0])


class MinkowskiPatch(AntennaGeometry):
    """Minkowski loop iteration applied to patch perimeter.

    Each straight edge segment is replaced by a Minkowski curve
    (outward square notch), increasing electrical length → miniaturisation.

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness [m].
    iterations : int
        Minkowski iterations (1–3 typical).
    indent_ratio : float
        Depth of notch as fraction of segment length (default 0.33).
    """

    def __init__(self, freq: float, eps_r: float, h: float,
                 iterations: int = 2, indent_ratio: float = 0.33):
        material = Material(name=f"Substrate_eps{eps_r}", eps_r=eps_r)
        super().__init__(
            name=f"MinkowskiPatch_iter{iterations}",
            material=material,
            freq_design=freq,
        )
        self.h = h
        self.eps_r = eps_r
        self.iterations = iterations
        lambda0 = C0 / freq

        # Base patch dimensions (from transmission-line model)
        from pylobe.geometry.patch import RectangularPatch
        base = RectangularPatch(freq=freq, eps_r=eps_r, h=h)
        W, L = base.W, base.L

        # Build Minkowski perimeter
        corners = np.array([
            [0.0, 0.0, h],
            [W,   0.0, h],
            [W,   L,   h],
            [0.0, L,   h],
        ])
        perimeter = self._minkowski_perimeter(corners, iterations, indent_ratio)
        gnd = perimeter.copy()
        gnd[:, 2] = 0.0
        self.vertices = np.vstack([perimeter, gnd])
        self.feed_point = np.array([W / 2.0, 0.0, h])

    @staticmethod
    def _minkowski_segment(p1: np.ndarray, p2: np.ndarray,
                           indent: float) -> np.ndarray:
        """Replace one segment with Minkowski square-notch (5 points)."""
        d = p2 - p1
        length = np.linalg.norm(d)
        n = d / length
        # Normal (outward): rotate n by 90° in the patch plane (z=const)
        normal = np.array([-n[1], n[0], 0.0])

        A = p1 + n * length / 3.0
        B = A + normal * indent * length
        C = B + n * length / 3.0
        D = p1 + n * 2.0 * length / 3.0
        return np.array([p1, A, B, C, D, p2])

    def _minkowski_perimeter(self, corners: np.ndarray,
                              iters: int, ratio: float) -> np.ndarray:
        pts = list(corners)
        pts.append(corners[0])  # close loop
        for _ in range(iters):
            new_pts = []
            for k in range(len(pts) - 1):
                seg = self._minkowski_segment(pts[k], pts[k+1], ratio)
                new_pts.extend(list(seg[:-1]))
            new_pts.append(pts[-1])
            pts = new_pts
        return np.array(pts[:-1])  # remove duplicate closing point
