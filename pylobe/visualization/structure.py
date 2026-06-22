"""Physical antenna structure visualisation using Plotly 3-D.

Each function renders what the antenna *looks like* — layered materials,
wire conductors, ground planes, feed points — as an interactive Plotly figure.

Main entry point
----------------
    plot_antenna_structure(antenna)   # auto-dispatches by type

Specialised functions
---------------------
    plot_patch_structure(patch)
    plot_circular_patch_structure(patch)
    plot_dipole_structure(dipole)
    plot_monopole_structure(mono)
    plot_helical_structure(helix)
    plot_bowtie_structure(bowtie)
    plot_array_structure(array)
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional

from pylobe.geometry.base import AntennaGeometry, Material, COPPER, PEC, ALUMINUM, AIR
from pylobe.constants import PI

# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ── 3-D geometry primitives ───────────────────────────────────────────────────

def _box_mesh(x0, y0, z0, x1, y1, z1,
              color: str, opacity: float = 0.45,
              name: str = '', show_scale: bool = False) -> go.Mesh3d:
    """Axis-aligned solid box as Plotly Mesh3d."""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    # 6 faces × 2 triangles each
    i = [0, 0,  4, 4,  0, 0,  3, 3,  0, 0,  1, 1]
    j = [1, 2,  5, 6,  1, 5,  2, 6,  3, 7,  2, 6]
    k = [2, 3,  6, 7,  5, 4,  6, 7,  7, 4,  6, 5]
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo='name',
        flatshading=False,
        lighting=dict(ambient=0.85, diffuse=0.6, specular=0.1),
        showscale=show_scale,
    )


def _rect_face(x0, y0, z, x1, y1,
               color: str, opacity: float = 1.0,
               name: str = '') -> go.Mesh3d:
    """Flat rectangle (two triangles) at constant z."""
    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[y0, y0, y1, y1],
        z=[z,  z,  z,  z],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo='name',
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.5),
    )


def _disc_mesh(cx, cy, z, radius: float, n_seg: int = 64,
               color: str = '#A8A8A8', opacity: float = 1.0,
               name: str = '') -> go.Mesh3d:
    """Circular disc (filled polygon) at constant z."""
    phi = np.linspace(0, 2 * PI, n_seg, endpoint=False)
    xs = [cx] + list(cx + radius * np.cos(phi))
    ys = [cy] + list(cy + radius * np.sin(phi))
    zs = [z]  + [z] * n_seg
    i_idx = [0] * n_seg
    j_idx = list(range(1, n_seg + 1))
    k_idx = list(range(2, n_seg + 1)) + [1]
    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_idx, j=j_idx, k=k_idx,
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo='name',
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.5),
    )


def _ring_mesh(cx, cy, z, r_inner: float, r_outer: float,
               n_seg: int = 64, color: str = '#B87333',
               opacity: float = 1.0, name: str = '') -> go.Mesh3d:
    """Annular ring (hollow disc) at constant z."""
    phi = np.linspace(0, 2 * PI, n_seg, endpoint=False)
    x_out = cx + r_outer * np.cos(phi)
    y_out = cy + r_outer * np.sin(phi)
    x_inn = cx + r_inner * np.cos(phi)
    y_inn = cy + r_inner * np.sin(phi)

    xs = list(x_out) + list(x_inn)
    ys = list(y_out) + list(y_inn)
    zs = [z] * (2 * n_seg)

    i_idx, j_idx, k_idx = [], [], []
    for s in range(n_seg):
        s1 = (s + 1) % n_seg
        # Outer-inner quad → 2 triangles
        o0, o1 = s, s1
        i0, i1 = n_seg + s, n_seg + s1
        i_idx += [o0, o0]
        j_idx += [o1, i0]
        k_idx += [i0, i1]

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_idx, j=j_idx, k=k_idx,
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo='name',
        flatshading=True,
    )


def _cylinder_surface(x0, y0, z0, x1, y1, z1, radius: float,
                      color: str, n_seg: int = 16, name: str = '') -> go.Mesh3d:
    """Approximate cylinder between two points as Mesh3d."""
    # Direction vector
    axis = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-15:
        return go.Scatter3d(x=[], y=[], z=[])

    az = axis / length
    # Perpendicular basis
    perp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(az, perp)) > 0.9:
        perp = np.array([0.0, 1.0, 0.0])
    ax = np.cross(az, perp)
    ax /= np.linalg.norm(ax)
    ay = np.cross(az, ax)

    phi = np.linspace(0, 2 * PI, n_seg, endpoint=False)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # Bottom ring (p0) and top ring (p1)
    p0 = np.array([x0, y0, z0])
    p1 = np.array([x1, y1, z1])
    xs, ys, zs = [], [], []
    for cp, sp in zip(cos_p, sin_p):
        v = p0 + radius * (cp * ax + sp * ay)
        xs.append(v[0]); ys.append(v[1]); zs.append(v[2])
    for cp, sp in zip(cos_p, sin_p):
        v = p1 + radius * (cp * ax + sp * ay)
        xs.append(v[0]); ys.append(v[1]); zs.append(v[2])

    i_idx, j_idx, k_idx = [], [], []
    for s in range(n_seg):
        s1 = (s + 1) % n_seg
        b0, b1 = s, s1
        t0, t1 = n_seg + s, n_seg + s1
        i_idx += [b0, b0]
        j_idx += [b1, t0]
        k_idx += [t0, t1]

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_idx, j=j_idx, k=k_idx,
        color=color,
        opacity=1.0,
        name=name,
        hoverinfo='name',
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.7, specular=0.3),
    )


def _feed_marker(x, y, z, name: str = 'Feed Point') -> go.Scatter3d:
    """Red diamond marker at feed point."""
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers+text',
        marker=dict(size=10, color='#E74C3C', symbol='diamond',
                    line=dict(color='#922B21', width=2)),
        text=[name],
        textfont=dict(size=11, color='#E74C3C', family='Arial Bold'),
        textposition='top center',
        name=name,
    )


def _dim_label(x_mid, y_mid, z_mid, text: str) -> go.Scatter3d:
    """Dimension annotation text label in 3-D space."""
    return go.Scatter3d(
        x=[x_mid], y=[y_mid], z=[z_mid],
        mode='text',
        text=[text],
        textfont=dict(size=11, color='#2C3E50', family='Arial'),
        hoverinfo='none',
        showlegend=False,
    )


def _default_layout(title: str, height: int = 650) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color='#2C3E50'),
                   x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x (mm)', showgrid=True, gridcolor='#BDC3C7',
                       backgroundcolor='#F8FAFB'),
            yaxis=dict(title='y (mm)', showgrid=True, gridcolor='#BDC3C7',
                       backgroundcolor='#F8FAFB'),
            zaxis=dict(title='z (mm)', showgrid=True, gridcolor='#BDC3C7',
                       backgroundcolor='#F8FAFB'),
            bgcolor='#F0F3F4',
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=-1.4, z=1.1)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#BDC3C7', borderwidth=1,
                    font=dict(size=11)),
        autosize=True,
        height=height,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=20, r=20, t=60, b=20),
    )


# ── Main dispatcher ───────────────────────────────────────────────────────────

def plot_antenna_structure(antenna, **kwargs) -> go.Figure:
    """Plot the physical 3-D structure of any antenna geometry.

    Automatically dispatches to the appropriate specialised function
    based on the antenna type.

    Parameters
    ----------
    antenna : AntennaGeometry or array type
        Any PyLobe antenna object.
    **kwargs
        Passed through to the specific plot function.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    from pylobe.geometry.patch import (
        RectangularPatch, CircularPatch, AnnularRingPatch, ESlotPatch
    )
    from pylobe.geometry.dipole import HalfWaveDipole, FoldedDipole, BowTieDipole
    from pylobe.geometry.monopole import QuarterWaveMonopole, HelicalMonopole
    from pylobe.geometry.array import LinearArray, PlanarArray, CircularArray

    if isinstance(antenna, RectangularPatch):
        return plot_patch_structure(antenna, **kwargs)
    if isinstance(antenna, CircularPatch):
        return plot_circular_patch_structure(antenna, **kwargs)
    if isinstance(antenna, AnnularRingPatch):
        return plot_annular_patch_structure(antenna, **kwargs)
    if isinstance(antenna, ESlotPatch):
        return plot_eslot_patch_structure(antenna, **kwargs)
    if isinstance(antenna, BowTieDipole):
        return plot_bowtie_structure(antenna, **kwargs)
    if isinstance(antenna, (HalfWaveDipole, FoldedDipole)):
        return plot_dipole_structure(antenna, **kwargs)
    if isinstance(antenna, QuarterWaveMonopole):
        return plot_monopole_structure(antenna, **kwargs)
    if isinstance(antenna, HelicalMonopole):
        return plot_helical_structure(antenna, **kwargs)
    if isinstance(antenna, (LinearArray, PlanarArray, CircularArray)):
        return plot_array_structure(antenna, **kwargs)
    # Generic fallback
    return _plot_generic_structure(antenna, **kwargs)


# ── Patch antenna structures ──────────────────────────────────────────────────

def plot_patch_structure(patch, title: Optional[str] = None,
                         show_dimensions: bool = True) -> go.Figure:
    """3-D structure of a rectangular microstrip patch antenna.

    Shows the layered construction: ground plane → substrate → copper patch.
    Feed point and inset line are highlighted in red.

    Parameters
    ----------
    patch : RectangularPatch
    title : str or None
    show_dimensions : bool
        Annotate W, L, h, y0 dimensions.

    Returns
    -------
    go.Figure
    """
    W  = patch.W  * 1e3   # mm
    L  = patch.L  * 1e3
    h  = patch.h  * 1e3
    y0 = patch.y0 * 1e3

    sub_color   = patch.substrate_material.color
    patch_color = patch.patch_material.color
    gnd_color   = patch.ground_material.color

    traces = []

    # ── Ground plane ──
    margin = W * 0.15
    traces.append(_rect_face(
        -margin, -margin, 0.0,
        W + margin, L + margin,
        color=gnd_color, opacity=1.0,
        name=f'Ground Plane ({patch.ground_material.name})',
    ))

    # ── Substrate box (semi-transparent) ──
    traces.append(_box_mesh(
        0, 0, 0, W, L, h,
        color=sub_color, opacity=0.40,
        name=f'Substrate ({patch.substrate_material.name})',
    ))
    # Substrate outline for clarity
    sx = [0, W, W, 0, 0, W, W, 0, 0, 0, W, W]
    sy = [0, 0, L, L, 0, 0, L, L, 0, L, L, 0]
    sz = [h, h, h, h, h, h, h, h, 0, 0, 0, 0]
    traces.append(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode='lines',
        line=dict(color='#7F8C8D', width=1.5, dash='dash'),
        showlegend=False, hoverinfo='none',
    ))

    # ── Copper patch (top face) ──
    traces.append(_rect_face(
        0, 0, h, W, L,
        color=patch_color, opacity=1.0,
        name=f'Patch ({patch.patch_material.name})',
    ))

    # ── Inset feed slot ──
    if patch.inset_feed and y0 > 0:
        slot_w = W * 0.05
        traces.append(go.Scatter3d(
            x=[W/2, W/2], y=[0, y0], z=[h, h],
            mode='lines',
            line=dict(color='#E74C3C', width=5),
            name='Inset Feed',
        ))
        # Small feed gap indicator
        traces.append(go.Scatter3d(
            x=[W/2 - slot_w, W/2 + slot_w, W/2 + slot_w, W/2 - slot_w, W/2 - slot_w],
            y=[0, 0, y0, y0, 0],
            z=[h]*5,
            mode='lines',
            line=dict(color='#922B21', width=1.5, dash='dot'),
            showlegend=False, hoverinfo='none',
        ))

    # ── Feed probe (coaxial) ──
    fp = patch.feed_point * 1e3
    traces.append(go.Scatter3d(
        x=[fp[0], fp[0]], y=[fp[1], fp[1]], z=[0, h],
        mode='lines',
        line=dict(color='#E74C3C', width=4),
        name='Feed Probe',
    ))
    traces.append(_feed_marker(fp[0], fp[1], h, 'Feed Point'))

    # ── Dimension annotations ──
    if show_dimensions:
        ext = W * 0.12
        # W arrow line
        traces.append(go.Scatter3d(
            x=[0, W], y=[L + ext, L + ext], z=[h]*2,
            mode='lines+text',
            line=dict(color='#34495E', width=2),
            text=['', f'W = {W:.2f} mm'],
            textfont=dict(size=10, color='#34495E'),
            textposition='middle right',
            showlegend=False, hoverinfo='none',
        ))
        # L arrow line
        traces.append(go.Scatter3d(
            x=[W + ext, W + ext], y=[0, L], z=[h]*2,
            mode='lines+text',
            line=dict(color='#34495E', width=2),
            text=['', f'L = {L:.2f} mm'],
            textfont=dict(size=10, color='#34495E'),
            textposition='middle right',
            showlegend=False, hoverinfo='none',
        ))
        # h label
        traces.append(go.Scatter3d(
            x=[-ext], y=[L/2], z=[h/2],
            mode='text',
            text=[f'h = {h:.2f} mm'],
            textfont=dict(size=10, color='#8E44AD'),
            showlegend=False, hoverinfo='none',
        ))
        if patch.inset_feed and y0 > 0:
            traces.append(go.Scatter3d(
                x=[W/2 + ext*1.5], y=[y0/2], z=[h],
                mode='text',
                text=[f'y₀ = {y0:.2f} mm'],
                textfont=dict(size=10, color='#E74C3C'),
                showlegend=False, hoverinfo='none',
            ))

    title = title or (
        f'Rectangular Patch — {patch.freq/1e9:.3f} GHz  '
        f'| {patch.substrate_material.name}  '
        f'| patch: {patch.patch_material.name}'
    )
    fig = go.Figure(data=traces)
    fig.update_layout(**_default_layout(title))
    return fig


def plot_circular_patch_structure(patch, title: Optional[str] = None) -> go.Figure:
    """3-D structure of a circular microstrip patch antenna.

    Parameters
    ----------
    patch : CircularPatch
    title : str or None

    Returns
    -------
    go.Figure
    """
    a  = patch.a  * 1e3   # mm (physical radius)
    h  = patch.h  * 1e3

    sub_color   = patch.substrate_material.color
    patch_color = patch.patch_material.color
    gnd_color   = patch.ground_material.color

    traces = []
    margin = a * 0.25

    # Ground disc
    traces.append(_disc_mesh(
        0, 0, 0, a + margin, n_seg=128,
        color=gnd_color, opacity=1.0,
        name=f'Ground Plane ({patch.ground_material.name})',
    ))

    # Substrate cylinder (approximate as box for simplicity)
    n_cyl = 64
    phi = np.linspace(0, 2 * PI, n_cyl, endpoint=False)
    for s in range(n_cyl):
        s1 = (s + 1) % n_cyl
        x0, y0_ = a * np.cos(phi[s]), a * np.sin(phi[s])
        x1, y1_ = a * np.cos(phi[s1]), a * np.sin(phi[s1])
        traces.append(go.Mesh3d(
            x=[x0, x1, x1, x0],
            y=[y0_, y1_, y1_, y0_],
            z=[0, 0, h, h],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=sub_color, opacity=0.35,
            showlegend=(s == 0),
            name=f'Substrate ({patch.substrate_material.name})' if s == 0 else '',
            hoverinfo='name' if s == 0 else 'none',
        ))

    # Patch disc on top
    traces.append(_disc_mesh(
        0, 0, h, a, n_seg=128,
        color=patch_color, opacity=1.0,
        name=f'Patch ({patch.patch_material.name})',
    ))

    # Feed probe
    fp = patch.feed_point * 1e3
    traces.append(go.Scatter3d(
        x=[fp[0], fp[0]], y=[fp[1], fp[1]], z=[0, h],
        mode='lines',
        line=dict(color='#E74C3C', width=4),
        name='Feed Probe',
    ))
    traces.append(_feed_marker(fp[0], fp[1], h))

    # Dimension annotation
    traces.append(go.Scatter3d(
        x=[0, a], y=[0, 0], z=[h]*2,
        mode='lines+text',
        line=dict(color='#34495E', width=2, dash='dash'),
        text=['', f'a = {a:.2f} mm'],
        textfont=dict(size=11, color='#34495E'),
        textposition='middle right',
        showlegend=False, hoverinfo='none',
    ))

    title = title or (
        f'Circular Patch — {patch.freq_design/1e9:.3f} GHz  '
        f'| {patch.substrate_material.name}'
    )
    fig = go.Figure(data=traces)
    fig.update_layout(**_default_layout(title))
    return fig


def plot_annular_patch_structure(patch, title: Optional[str] = None) -> go.Figure:
    """3-D structure of an annular ring microstrip patch antenna.

    Parameters
    ----------
    patch : AnnularRingPatch
    title : str or None

    Returns
    -------
    go.Figure
    """
    r_in  = patch.inner_radius * 1e3
    r_out = patch.outer_radius * 1e3
    h     = patch.h * 1e3

    sub_color   = patch.substrate_material.color
    patch_color = patch.patch_material.color
    gnd_color   = patch.ground_material.color

    traces = []

    # Ground disc (larger than outer ring)
    margin = r_out * 0.2
    traces.append(_disc_mesh(
        0, 0, 0, r_out + margin,
        color=gnd_color, name=f'Ground Plane ({patch.ground_material.name})',
    ))

    # Substrate disc
    traces.append(_disc_mesh(
        0, 0, h, r_out + margin / 2,
        color=sub_color, opacity=0.35,
        name=f'Substrate ({patch.substrate_material.name})',
    ))

    # Annular ring patch
    traces.append(_ring_mesh(
        0, 0, h, r_in, r_out,
        color=patch_color, opacity=1.0,
        name=f'Ring Patch ({patch.patch_material.name})',
    ))

    # Feed marker
    fp = patch.feed_point * 1e3
    traces.append(_feed_marker(fp[0], fp[1], h))

    # Dimension labels
    traces.append(go.Scatter3d(
        x=[0, r_in], y=[0, 0], z=[h]*2,
        mode='lines+text',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        text=['', f'r_in={r_in:.1f} mm'],
        textfont=dict(size=10, color='#E74C3C'),
        textposition='middle right',
        showlegend=False, hoverinfo='none',
    ))
    traces.append(go.Scatter3d(
        x=[0, r_out], y=[r_out*0.05, r_out*0.05], z=[h]*2,
        mode='lines+text',
        line=dict(color='#2980B9', width=2, dash='dash'),
        text=['', f'r_out={r_out:.1f} mm'],
        textfont=dict(size=10, color='#2980B9'),
        textposition='middle right',
        showlegend=False, hoverinfo='none',
    ))

    title = title or f'Annular Ring Patch — {patch.freq_design/1e9:.3f} GHz'
    fig = go.Figure(data=traces)
    fig.update_layout(**_default_layout(title))
    return fig


def plot_eslot_patch_structure(patch, title: Optional[str] = None) -> go.Figure:
    """3-D structure of an E-slot dual-band patch antenna.

    Parameters
    ----------
    patch : ESlotPatch
    title : str or None

    Returns
    -------
    go.Figure
    """
    W  = patch.W * 1e3
    L  = patch.L * 1e3
    h  = patch.h * 1e3

    sub_color   = patch.substrate_material.color
    patch_color = patch.patch_material.color
    gnd_color   = patch.ground_material.color
    slot_gap    = patch.slot_gap * 1e3
    slot_h_mm   = patch.slot_h  * 1e3
    slot_w_mm   = patch.slot_w  * 1e3

    traces = []

    # Ground plane
    margin = W * 0.15
    traces.append(_rect_face(
        -margin, -margin, 0,
        W + margin, L + margin,
        color=gnd_color,
        name=f'Ground Plane ({patch.ground_material.name})',
    ))

    # Substrate
    traces.append(_box_mesh(
        0, 0, 0, W, L, h,
        color=sub_color, opacity=0.40,
        name=f'Substrate ({patch.substrate_material.name})',
    ))

    # Main patch (without slots)
    traces.append(_rect_face(
        0, 0, h, W, L,
        color=patch_color,
        name=f'Patch ({patch.patch_material.name})',
    ))

    # E-slots (shown as substrate-colored cutouts slightly above patch)
    slot_x0 = (W - slot_w_mm) / 2
    slot_x1 = slot_x0 + slot_w_mm
    for k in range(2):
        y_slot = L * 0.3 + k * slot_gap
        traces.append(_rect_face(
            slot_x0, y_slot, h + 0.01,
            slot_x1, y_slot + slot_h_mm,
            color='#ECF0F1', opacity=1.0,
            name='E-Slot' if k == 0 else '',
        ))

    # Feed
    fp = patch.feed_point * 1e3
    traces.append(_feed_marker(fp[0], fp[1], h))

    title = title or (
        f'E-Slot Patch — {patch.freq1/1e9:.3f}/{patch.freq2/1e9:.3f} GHz'
    )
    fig = go.Figure(data=traces)
    fig.update_layout(**_default_layout(title))
    return fig


# ── Wire antenna structures ───────────────────────────────────────────────────

def plot_dipole_structure(dipole, title: Optional[str] = None,
                          show_dimensions: bool = True) -> go.Figure:
    """3-D structure of a half-wave or folded dipole antenna.

    Renders the wire as a solid cylinder with realistic proportions.

    Parameters
    ----------
    dipole : HalfWaveDipole or FoldedDipole
    title : str or None
    show_dimensions : bool

    Returns
    -------
    go.Figure
    """
    from pylobe.geometry.dipole import FoldedDipole

    verts = dipole.vertices * 1e3    # mm
    wire_r = dipole.wire_radius * 1e3
    cond_color = dipole.conductor_material.color
    traces = []

    if isinstance(dipole, FoldedDipole):
        # Draw each edge as a cylinder
        for i0, i1 in dipole.edges:
            v0, v1 = verts[i0], verts[i1]
            traces.append(_cylinder_surface(
                v0[0], v0[1], v0[2],
                v1[0], v1[1], v1[2],
                radius=wire_r, color=cond_color,
                name=f'Wire ({dipole.conductor_material.name})',
            ))
        # Feed gap visualization
        fp = dipole.feed_point * 1e3
        traces.append(_feed_marker(fp[0], fp[1], fp[2]))
    else:
        # Half-wave dipole: upper and lower arms
        arm_mm = dipole.arm_length * 1e3
        gap_mm = max(wire_r * 2, arm_mm / 30.0)

        # Upper arm
        traces.append(_cylinder_surface(
            0, 0, gap_mm, 0, 0, arm_mm,
            radius=wire_r, color=cond_color,
            name=f'Upper Arm ({dipole.conductor_material.name})',
        ))
        # Lower arm
        traces.append(_cylinder_surface(
            0, 0, -arm_mm, 0, 0, -gap_mm,
            radius=wire_r, color=cond_color,
            name=f'Lower Arm ({dipole.conductor_material.name})',
        ))
        # Feed gap
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-gap_mm, gap_mm],
            mode='lines',
            line=dict(color='#E74C3C', width=6, dash='dot'),
            name='Feed Gap',
        ))
        traces.append(_feed_marker(0, 0, 0))

        if show_dimensions:
            arm_mm = dipole.arm_length * 1e3
            ext = arm_mm * 0.18
            traces.append(go.Scatter3d(
                x=[ext], y=[0], z=[arm_mm / 2],
                mode='text',
                text=[f'L/2 = {arm_mm:.1f} mm'],
                textfont=dict(size=11, color='#34495E'),
                showlegend=False, hoverinfo='none',
            ))
            traces.append(go.Scatter3d(
                x=[ext*2], y=[0], z=[0],
                mode='text',
                text=[f'r = {wire_r:.2f} mm'],
                textfont=dict(size=10, color='#8E44AD'),
                showlegend=False, hoverinfo='none',
            ))

    title = title or (
        f'{"Folded " if isinstance(dipole, FoldedDipole) else ""}Half-Wave Dipole — '
        f'{dipole.freq_design/1e9:.3f} GHz  '
        f'| {dipole.conductor_material.name}'
    )
    fig = go.Figure(data=traces)
    layout = _default_layout(title)
    layout['scene']['camera'] = dict(eye=dict(x=2.0, y=1.5, z=0.5))
    fig.update_layout(**layout)
    return fig


def plot_bowtie_structure(bowtie, title: Optional[str] = None) -> go.Figure:
    """3-D structure of a bow-tie dipole antenna.

    Renders both triangular arms as solid flat surfaces.

    Parameters
    ----------
    bowtie : BowTieDipole
    title : str or None

    Returns
    -------
    go.Figure
    """
    arm_mm = bowtie.arm_length * 1e3
    half_w = arm_mm * np.tan(bowtie.flare_angle_rad)
    cond_color = bowtie.conductor_material.color
    traces = []

    # Right arm triangle: feed=(0,0,0), corners at (arm_mm, ±half_w, 0)
    traces.append(go.Mesh3d(
        x=[0, arm_mm, arm_mm],
        y=[0, half_w, -half_w],
        z=[0, 0, 0],
        i=[0], j=[1], k=[2],
        color=cond_color,
        opacity=1.0,
        name=f'Right Arm ({bowtie.conductor_material.name})',
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.5),
    ))
    # Left arm triangle: feed=(0,0,0), corners at (-arm_mm, ±half_w, 0)
    traces.append(go.Mesh3d(
        x=[0, -arm_mm, -arm_mm],
        y=[0, half_w, -half_w],
        z=[0, 0, 0],
        i=[0], j=[1], k=[2],
        color=cond_color,
        opacity=1.0,
        name=f'Left Arm ({bowtie.conductor_material.name})',
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.5),
    ))
    # Outline edges
    for x_tip in [arm_mm, -arm_mm]:
        traces.append(go.Scatter3d(
            x=[0, x_tip, x_tip, 0],
            y=[0, half_w, -half_w, 0],
            z=[0, 0, 0, 0],
            mode='lines',
            line=dict(color='#2C3E50', width=2),
            showlegend=False, hoverinfo='none',
        ))
    # Feed gap
    gap = arm_mm * 0.04
    traces.append(go.Scatter3d(
        x=[-gap, gap], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='#E74C3C', width=6, dash='dot'),
        name='Feed Gap',
    ))
    traces.append(_feed_marker(0, 0, 0))

    # Dimension labels
    traces.append(go.Scatter3d(
        x=[arm_mm / 2], y=[half_w * 1.3], z=[0],
        mode='text',
        text=[f'arm = {arm_mm:.1f} mm  |  α = {bowtie.flare_angle_deg:.0f}°'],
        textfont=dict(size=11, color='#34495E'),
        showlegend=False, hoverinfo='none',
    ))

    title = title or (
        f'Bow-Tie Dipole — {bowtie.freq_design/1e9:.3f} GHz  '
        f'| flare {bowtie.flare_angle_deg:.0f}°  '
        f'| {bowtie.conductor_material.name}'
    )
    fig = go.Figure(data=traces)
    layout = _default_layout(title)
    layout['scene']['camera'] = dict(eye=dict(x=0.5, y=2.5, z=1.5))
    fig.update_layout(**layout)
    return fig


# ── Monopole / helical structures ─────────────────────────────────────────────

def plot_monopole_structure(mono, title: Optional[str] = None,
                            show_dimensions: bool = True) -> go.Figure:
    """3-D structure of a quarter-wave monopole antenna.

    Shows the vertical wire above a circular ground plane disc.

    Parameters
    ----------
    mono : QuarterWaveMonopole
    title : str or None
    show_dimensions : bool

    Returns
    -------
    go.Figure
    """
    L_mm     = mono.L * 1e3
    r_gnd_mm = mono.ground_radius * 1e3
    wire_r   = mono.wire_radius * 1e3
    cond_color = mono.conductor_material.color
    gnd_color  = mono.ground_material.color
    traces = []

    # Ground plane disc
    traces.append(_disc_mesh(
        0, 0, 0, r_gnd_mm,
        color=gnd_color, opacity=1.0,
        name=f'Ground Plane ({mono.ground_material.name})',
    ))
    # Ground plane edge ring
    phi = np.linspace(0, 2 * PI, 128)
    traces.append(go.Scatter3d(
        x=r_gnd_mm * np.cos(phi),
        y=r_gnd_mm * np.sin(phi),
        z=np.zeros(128),
        mode='lines',
        line=dict(color='#7F8C8D', width=1.5),
        showlegend=False, hoverinfo='none',
    ))

    # Monopole wire (cylinder)
    traces.append(_cylinder_surface(
        0, 0, 0, 0, 0, L_mm,
        radius=wire_r, color=cond_color,
        name=f'Monopole Wire ({mono.conductor_material.name})',
    ))

    # Feed point
    traces.append(_feed_marker(0, 0, 0, 'Feed (base)'))

    if show_dimensions:
        ext = r_gnd_mm * 0.3
        traces.append(go.Scatter3d(
            x=[ext], y=[0], z=[L_mm / 2],
            mode='text',
            text=[f'L = {L_mm:.1f} mm (λ/4)'],
            textfont=dict(size=11, color='#34495E'),
            showlegend=False, hoverinfo='none',
        ))
        traces.append(go.Scatter3d(
            x=[r_gnd_mm / 2], y=[r_gnd_mm * 0.1], z=[0],
            mode='text',
            text=[f'GND r = {r_gnd_mm:.1f} mm'],
            textfont=dict(size=10, color='#7F8C8D'),
            showlegend=False, hoverinfo='none',
        ))

    title = title or (
        f'Quarter-Wave Monopole — {mono.freq_design/1e9:.3f} GHz  '
        f'| {mono.conductor_material.name}'
    )
    fig = go.Figure(data=traces)
    layout = _default_layout(title)
    layout['scene']['camera'] = dict(eye=dict(x=1.8, y=1.5, z=0.8))
    fig.update_layout(**layout)
    return fig


def plot_helical_structure(helix, title: Optional[str] = None) -> go.Figure:
    """3-D structure of a helical antenna.

    Shows the spiral wire above a circular ground plane.

    Parameters
    ----------
    helix : HelicalMonopole
    title : str or None

    Returns
    -------
    go.Figure
    """
    n_helix = len(helix.vertices) - 64   # helix verts (exclude ground disc)
    helix_verts  = helix.vertices[:n_helix] * 1e3
    r_gnd_mm     = helix.ground_radius * 1e3
    wire_r       = (helix.diameter / 200.0) * 1e3
    cond_color   = helix.conductor_material.color
    gnd_color    = helix.ground_material.color

    traces = []

    # Ground plane disc
    traces.append(_disc_mesh(
        0, 0, 0, r_gnd_mm,
        color=gnd_color, opacity=1.0,
        name=f'Ground Plane ({helix.ground_material.name})',
    ))

    # Helix wire as cylinder segments
    for i in range(len(helix_verts) - 1):
        v0, v1 = helix_verts[i], helix_verts[i + 1]
        traces.append(_cylinder_surface(
            v0[0], v0[1], v0[2],
            v1[0], v1[1], v1[2],
            radius=wire_r, color=cond_color, n_seg=8,
            name=f'Helix ({helix.conductor_material.name})' if i == 0 else '',
        ))

    # Feed
    fp = helix.feed_point * 1e3
    traces.append(_feed_marker(fp[0], fp[1], 0, 'Feed Point'))

    # Dimension labels
    total_h = helix.total_length * 1e3
    diam_mm = helix.diameter * 1e3
    traces.append(go.Scatter3d(
        x=[diam_mm], y=[0], z=[total_h / 2],
        mode='text',
        text=[f'{helix.N_turns} turns  |  D={diam_mm:.1f} mm  |  h={total_h:.1f} mm'],
        textfont=dict(size=10, color='#34495E'),
        showlegend=False, hoverinfo='none',
    ))

    title = title or (
        f'Helical Antenna ({helix.mode.capitalize()} Mode) — '
        f'{helix.freq_design/1e9:.3f} GHz  '
        f'| {helix.N_turns} turns'
    )
    fig = go.Figure(data=traces)
    layout = _default_layout(title, height=700)
    layout['scene']['camera'] = dict(eye=dict(x=2.0, y=1.5, z=0.6))
    fig.update_layout(**layout)
    return fig



# ── Array structures ──────────────────────────────────────────────────────────

def plot_array_structure(array, title: Optional[str] = None,
                         show_element_shape: bool = True) -> go.Figure:
    """3-D structure visualisation for antenna arrays.

    Renders element positions with amplitude/phase colouring and optionally
    a miniature copy of the element geometry at each position.

    Parameters
    ----------
    array : LinearArray, PlanarArray, or CircularArray
    title : str or None
    show_element_shape : bool
        Show element geometry at each position (uses element vertices).

    Returns
    -------
    go.Figure
    """
    from pylobe.geometry.array import LinearArray, PlanarArray, CircularArray

    traces = []

    if isinstance(array, LinearArray):
        positions = array.positions() * 1e3   # mm
        amps = array.amplitudes / array.amplitudes.max()
        d_mm = array.d * 1e3
        elem_name = array.element.name if hasattr(array.element, 'name') else 'Element'

        # Amplitude-coloured element markers
        traces.append(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=amps,
                colorscale='Viridis',
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title='Amplitude', thickness=14,
                              tickfont=dict(size=10)),
                line=dict(color='#2C3E50', width=1),
            ),
            text=[f'#{i} | a={a:.2f}' for i, a in enumerate(amps)],
            textfont=dict(size=8),
            name=f'{elem_name} positions (N={array.N})',
        ))

        # Element-to-element lines
        traces.append(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(color='#BDC3C7', width=2, dash='dash'),
            showlegend=False, hoverinfo='none',
        ))

        # Show element shape at each position
        if show_element_shape and hasattr(array.element, 'vertices'):
            ev = array.element.vertices * 1e3
            for pos in positions:
                ev_t = ev + pos
                for i0, i1 in array.element.edges:
                    traces.append(go.Scatter3d(
                        x=[ev_t[i0, 0], ev_t[i1, 0]],
                        y=[ev_t[i0, 1], ev_t[i1, 1]],
                        z=[ev_t[i0, 2], ev_t[i1, 2]],
                        mode='lines',
                        line=dict(color='#95A5A6', width=1.5),
                        showlegend=False, hoverinfo='none',
                    ))

        # Spacing annotation
        if array.N > 1:
            mid = array.N // 2
            traces.append(go.Scatter3d(
                x=[positions[mid, 0], positions[mid + 1, 0]],
                y=[positions[mid, 1] - d_mm * 0.2]*2,
                z=[positions[mid, 2], positions[mid + 1, 2]],
                mode='lines+text',
                line=dict(color='#E74C3C', width=2),
                text=['', f'd = {d_mm:.1f} mm'],
                textfont=dict(size=10, color='#E74C3C'),
                textposition='middle right',
                showlegend=False, hoverinfo='none',
            ))

        beta_deg = np.rad2deg(array.beta) if array.beta != 0 else 0.0
        title = title or (
            f'Linear Array  N={array.N}  d={d_mm:.1f} mm  β={beta_deg:.1f}°  '
            f'| {elem_name}'
        )

    elif isinstance(array, CircularArray):
        positions = array.positions() * 1e3
        amps = array.amplitudes / array.amplitudes.max()
        R_mm = array.R * 1e3
        elem_name = array.element.name if hasattr(array.element, 'name') else 'Element'

        traces.append(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=amps,
                colorscale='Plasma',
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title='Amplitude', thickness=14),
            ),
            text=[f'#{i}' for i in range(array.N)],
            textfont=dict(size=8),
            name=f'Circular Array N={array.N}, R={R_mm:.1f} mm',
        ))
        # Ring line
        phi = np.linspace(0, 2 * PI, 256)
        traces.append(go.Scatter3d(
            x=R_mm * np.cos(phi),
            y=R_mm * np.sin(phi),
            z=np.zeros(256),
            mode='lines',
            line=dict(color='#BDC3C7', width=1.5, dash='dot'),
            showlegend=False, hoverinfo='none',
        ))
        title = title or f'Circular Array  N={array.N}  R={R_mm:.1f} mm'

    elif isinstance(array, PlanarArray):
        elem_name = array.element.name if hasattr(array.element, 'name') else 'Element'
        dx_mm = array.dx * 1e3
        dy_mm = array.dy * 1e3
        xs, ys, zs, texts = [], [], [], []
        for m in range(array.M):
            for n in range(array.N):
                xs.append(m * dx_mm)
                ys.append(n * dy_mm)
                zs.append(0.0)
                a_mn = float(array.amplitudes[m, n])
                texts.append(f'({m},{n})\na={a_mn:.2f}')

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(
                size=9,
                color=zs,
                colorscale='Viridis',
                line=dict(color='#2C3E50', width=1),
            ),
            text=texts,
            name=f'Planar Array {array.M}×{array.N}  |  {elem_name}',
        ))
        title = title or (
            f'Planar Array {array.M}×{array.N}  '
            f'dx={dx_mm:.1f} mm  dy={dy_mm:.1f} mm  '
            f'| {elem_name}'
        )

    fig = go.Figure(data=traces)
    layout = _default_layout(title or 'Antenna Array Structure')
    layout['scene']['camera'] = dict(eye=dict(x=1.5, y=-1.8, z=1.2))
    fig.update_layout(**layout)
    return fig


# ── Generic fallback ──────────────────────────────────────────────────────────

def _plot_generic_structure(geometry: AntennaGeometry,
                            title: Optional[str] = None) -> go.Figure:
    """Generic wireframe + vertex visualisation for any AntennaGeometry."""
    verts = geometry.vertices * 1e3
    traces = []

    # Edges
    for i0, i1 in geometry.edges:
        traces.append(go.Scatter3d(
            x=[verts[i0, 0], verts[i1, 0]],
            y=[verts[i0, 1], verts[i1, 1]],
            z=[verts[i0, 2], verts[i1, 2]],
            mode='lines',
            line=dict(color=geometry.material.color, width=4),
            showlegend=False,
        ))

    # Vertex cloud
    traces.append(go.Scatter3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        mode='markers',
        marker=dict(size=4, color='#3498DB'),
        name='Vertices',
    ))

    # Feed point
    if geometry.feed_point is not None:
        fp = geometry.feed_point * 1e3
        traces.append(_feed_marker(fp[0], fp[1], fp[2]))

    title = title or f'{geometry.__class__.__name__} — {geometry.freq_design/1e9:.3f} GHz'
    fig = go.Figure(data=traces)
    fig.update_layout(**_default_layout(title))
    return fig
