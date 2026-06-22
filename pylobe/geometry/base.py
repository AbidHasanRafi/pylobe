"""Base classes for all antenna geometries."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class Material:
    """Electromagnetic material definition.

    Parameters
    ----------
    name : str
        Human-readable material name.
    eps_r : float
        Relative permittivity (unitless).
    mu_r : float
        Relative permeability (unitless).
    loss_tangent : float
        Dielectric loss tangent.
    conductivity : float
        Electrical conductivity [S/m].
    color : str
        Hex color string for structure visualization (e.g. '#B87333').
    """
    name: str
    eps_r: float = 1.0
    mu_r: float = 1.0
    loss_tangent: float = 0.0
    conductivity: float = 0.0
    color: str = '#C0C0C0'

    def __repr__(self) -> str:
        return (
            f"Material('{self.name}', "
            f"eps_r={self.eps_r}, "
            f"tan_d={self.loss_tangent}, "
            f"sigma={self.conductivity:.2e})"
        )

    def is_conductor(self) -> bool:
        """Return True if material is primarily conductive (sigma > 1e4 S/m)."""
        return self.conductivity > 1e4

    def is_dielectric(self) -> bool:
        """Return True if material is primarily a dielectric."""
        return not self.is_conductor()


# ── Pre-defined dielectric substrates ────────────────────────────────────────
AIR         = Material("Air",            eps_r=1.0,   mu_r=1.0, loss_tangent=0.0,      color='#E8F4FD')
FR4         = Material("FR4",            eps_r=4.4,   mu_r=1.0, loss_tangent=0.02,     color='#2E7D32')
RT5880      = Material("RT/duroid5880",  eps_r=2.2,   mu_r=1.0, loss_tangent=0.0009,   color='#F5DEB3')
ROGERS4003  = Material("Rogers4003C",    eps_r=3.55,  mu_r=1.0, loss_tangent=0.0027,   color='#DEB887')
ROGERS3010  = Material("Rogers3010",     eps_r=10.2,  mu_r=1.0, loss_tangent=0.0023,   color='#D2B48C')
ARLON250    = Material("Arlon250LX",     eps_r=2.5,   mu_r=1.0, loss_tangent=0.0019,   color='#FAEBD7')
ISOLA370HR  = Material("Isola370HR",     eps_r=4.04,  mu_r=1.0, loss_tangent=0.0170,   color='#90EE90')
TEFLON      = Material("Teflon/PTFE",    eps_r=2.1,   mu_r=1.0, loss_tangent=0.0001,   color='#F8F8FF')
ALUMINA     = Material("Alumina",        eps_r=9.8,   mu_r=1.0, loss_tangent=0.0001,   color='#FFFACD')
SILICON     = Material("Silicon",        eps_r=11.7,  mu_r=1.0, loss_tangent=0.005,    color='#708090')
GaAs        = Material("GaAs",           eps_r=12.9,  mu_r=1.0, loss_tangent=0.002,    color='#8B7355')
InP         = Material("InP",            eps_r=12.4,  mu_r=1.0, loss_tangent=0.001,    color='#6B8E23')
CERAMIC     = Material("Ceramic",        eps_r=6.0,   mu_r=1.0, loss_tangent=0.001,    color='#F5F5DC')
FOAM        = Material("Rohacell Foam",  eps_r=1.07,  mu_r=1.0, loss_tangent=0.0017,   color='#FFF8DC')

# ── Pre-defined conductors ────────────────────────────────────────────────────
PEC         = Material("PEC",            eps_r=1.0,   mu_r=1.0, conductivity=1e7,      color='#A8A8A8')
COPPER      = Material("Copper",         eps_r=1.0,   mu_r=1.0, conductivity=5.8e7,    color='#B87333')
GOLD        = Material("Gold",           eps_r=1.0,   mu_r=1.0, conductivity=4.1e7,    color='#FFD700')
SILVER      = Material("Silver",         eps_r=1.0,   mu_r=1.0, conductivity=6.3e7,    color='#C0C0C0')
ALUMINUM    = Material("Aluminum",       eps_r=1.0,   mu_r=1.0, conductivity=3.5e7,    color='#D3D3D3')
BRASS       = Material("Brass",          eps_r=1.0,   mu_r=1.0, conductivity=1.5e7,    color='#CFB53B')
NICHROME    = Material("Nichrome",       eps_r=1.0,   mu_r=1.0, conductivity=9.0e5,    color='#808080')

# ── Global material library / registry ───────────────────────────────────────
MATERIAL_LIBRARY: Dict[str, Material] = {
    m.name: m for m in [
        # Dielectrics / substrates
        AIR, FR4, RT5880, ROGERS4003, ROGERS3010, ARLON250,
        ISOLA370HR, TEFLON, ALUMINA, SILICON, GaAs, InP,
        CERAMIC, FOAM,
        # Conductors
        PEC, COPPER, GOLD, SILVER, ALUMINUM, BRASS, NICHROME,
    ]
}


def register_material(mat: Material) -> None:
    """Add a custom material to the global library.

    Parameters
    ----------
    mat : Material
        Material to register. Overwrites any existing entry with the same name.

    Examples
    --------
    >>> my_sub = Material("MySubstrate", eps_r=3.0, loss_tangent=0.005, color='#AADDFF')
    >>> register_material(my_sub)
    >>> m = get_material("MySubstrate")
    """
    MATERIAL_LIBRARY[mat.name] = mat


def get_material(name: str) -> Material:
    """Look up a material by name from the global library.

    Parameters
    ----------
    name : str
        Material name (case-sensitive).

    Returns
    -------
    Material

    Raises
    ------
    KeyError
        If the name is not found in the library.
    """
    if name not in MATERIAL_LIBRARY:
        available = list(MATERIAL_LIBRARY.keys())
        raise KeyError(
            f"Material '{name}' not in library.\n"
            f"Available ({len(available)}): {available}"
        )
    return MATERIAL_LIBRARY[name]


def list_materials() -> List[str]:
    """Return sorted list of all registered material names."""
    return sorted(MATERIAL_LIBRARY.keys())


def print_material_table() -> None:
    """Pretty-print a table of all registered materials with key properties."""
    print(f"\n{'Name':<22} {'eps_r':>7} {'mu_r':>6} {'tan_d':>10} {'sigma [S/m]':>14} {'Type'}")
    print("-" * 70)
    for name, m in sorted(MATERIAL_LIBRARY.items()):
        mat_type = "Conductor" if m.is_conductor() else "Dielectric"
        print(
            f"{m.name:<22} {m.eps_r:>7.2f} {m.mu_r:>6.2f} "
            f"{m.loss_tangent:>10.4f} {m.conductivity:>14.2e}  {mat_type}"
        )


@dataclass
class AntennaGeometry:
    """Abstract base for all antenna geometries.

    Stores vertices, edges, faces, material, and feed point.
    All dimensions in metres (SI units).

    Attributes
    ----------
    name : str
        Antenna identifier.
    material : Material
        Primary substrate / bulk material.
    freq_design : float
        Design centre frequency [Hz].
    vertices : ndarray, shape (N, 3)
        3-D vertex coordinates [m].
    edges : list of (int, int)
        Pairs of vertex indices defining edges.
    faces : list of list of int
        Vertex index lists defining planar faces.
    feed_point : ndarray, shape (3,) or None
        Feed location [x, y, z] [m].
    feed_impedance : float
        Target feed impedance [Ω].
    """
    name: str
    material: Material
    freq_design: float
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    edges: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    feed_point: Optional[np.ndarray] = None
    feed_impedance: float = 50.0

    def bounding_box(self):
        """Returns (min_xyz, max_xyz) of the vertex cloud.

        Returns
        -------
        tuple of ndarray, shape (3,)
        """
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def dimensions(self) -> np.ndarray:
        """Axis-aligned bounding-box dimensions [Lx, Ly, Lz] in metres."""
        lo, hi = self.bounding_box()
        return hi - lo

    def translate(self, delta: np.ndarray):
        """Translate all vertices and feed point by delta [m].

        Parameters
        ----------
        delta : array_like, shape (3,)
        """
        delta = np.asarray(delta)
        self.vertices = self.vertices + delta
        if self.feed_point is not None:
            self.feed_point = self.feed_point + delta

    def __repr__(self) -> str:
        dims = self.dimensions() * 1e3
        return (
            f"{self.__class__.__name__}("
            f"f={self.freq_design/1e9:.3f} GHz, "
            f"dims=[{dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f}] mm)"
        )
