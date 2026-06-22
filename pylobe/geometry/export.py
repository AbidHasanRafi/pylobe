"""Geometry export utilities (DXF, STL, GDSII, JSON)."""
import json
import numpy as np
from pylobe.geometry.base import AntennaGeometry


def to_json(geometry: AntennaGeometry, filename: str):
    """Serialise antenna geometry to JSON.

    Parameters
    ----------
    geometry : AntennaGeometry
    filename : str
        Output file path.
    """
    data = {
        "class":          geometry.__class__.__name__,
        "name":           geometry.name,
        "freq_design_hz": geometry.freq_design,
        "material": {
            "name":         geometry.material.name,
            "eps_r":        geometry.material.eps_r,
            "mu_r":         geometry.material.mu_r,
            "loss_tangent": geometry.material.loss_tangent,
            "conductivity": geometry.material.conductivity,
        },
        "vertices":       geometry.vertices.tolist(),
        "edges":          geometry.edges,
        "faces":          geometry.faces,
        "feed_point":     geometry.feed_point.tolist() if geometry.feed_point is not None else None,
        "feed_impedance": geometry.feed_impedance,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def from_json(filename: str) -> AntennaGeometry:
    """Load antenna geometry from JSON.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    AntennaGeometry
    """
    from pylobe.geometry.base import Material
    with open(filename, "r") as f:
        data = json.load(f)
    mat = Material(
        name=data["material"]["name"],
        eps_r=data["material"]["eps_r"],
        mu_r=data["material"]["mu_r"],
        loss_tangent=data["material"]["loss_tangent"],
        conductivity=data["material"]["conductivity"],
    )
    geom = AntennaGeometry(
        name=data["name"],
        material=mat,
        freq_design=data["freq_design_hz"],
        vertices=np.array(data["vertices"]),
        edges=data["edges"],
        faces=data["faces"],
        feed_point=np.array(data["feed_point"]) if data["feed_point"] else None,
        feed_impedance=data["feed_impedance"],
    )
    return geom


def to_stl(geometry: AntennaGeometry, filename: str):
    """Export triangulated surface as ASCII STL.

    Triangulates each face by fan decomposition.

    Parameters
    ----------
    geometry : AntennaGeometry
    filename : str
        Output .stl file path.
    """
    verts = geometry.vertices
    with open(filename, "w") as f:
        f.write(f"solid {geometry.name}\n")
        for face in geometry.faces:
            if len(face) < 3:
                continue
            # Fan triangulation from face[0]
            for k in range(1, len(face) - 1):
                v0 = verts[face[0]]
                v1 = verts[face[k]]
                v2 = verts[face[k + 1]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                n = n / norm if norm > 0 else np.array([0, 0, 1])
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in (v0, v1, v2):
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
        f.write(f"endsolid {geometry.name}\n")


def to_dxf(geometry: AntennaGeometry, filename: str):
    """Export to AutoCAD DXF using ezdxf.

    Parameters
    ----------
    geometry : AntennaGeometry
    filename : str
        Output .dxf file path.
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf is required for DXF export: pip install ezdxf")

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    verts = geometry.vertices

    for edge in geometry.edges:
        p1 = verts[edge[0]]
        p2 = verts[edge[1]]
        msp.add_line(
            start=(float(p1[0]), float(p1[1]), float(p1[2])),
            end  =(float(p2[0]), float(p2[1]), float(p2[2])),
        )

    for face in geometry.faces:
        pts = [verts[i] for i in face]
        if len(pts) == 4:
            msp.add_3dface(
                [tuple(pts[i].tolist()) for i in range(4)]
            )
        elif len(pts) == 3:
            msp.add_3dface(
                [tuple(pts[i].tolist()) for i in range(3)] + [tuple(pts[2].tolist())]
            )

    doc.saveas(filename)


def to_gds(geometry: AntennaGeometry, filename: str, layer: int = 1):
    """Export to GDSII format (top-view projection onto XY plane).

    Parameters
    ----------
    geometry : AntennaGeometry
    filename : str
        Output .gds file path.
    layer : int
        GDSII layer number.
    """
    try:
        import gdspy
    except ImportError:
        raise ImportError("gdspy is required for GDSII export: pip install gdspy")

    lib = gdspy.GdsLibrary()
    cell = lib.new_cell(geometry.name[:32])

    verts_xy = geometry.vertices[:, :2]  # project to XY
    for face in geometry.faces:
        pts = verts_xy[[i for i in face]]
        poly = gdspy.Polygon(pts, layer=layer)
        cell.add(poly)

    lib.write_gds(filename)
