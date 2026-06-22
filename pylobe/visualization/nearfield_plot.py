"""Research-grade near-field distribution and current distribution visualisation.

Upgrades over legacy version
-----------------------------
* ``'Plasma'`` colormap replaces ``'hot'`` (perceptually uniform, print-safe).
* ``plot_nearfield_2d``: optional side-by-side magnitude + phase layout.
* ``plot_current_distribution``: improved Turbo colorscale; proper colorbar
  using a dedicated Scatter3d colorscale instead of a proxy trace.
* ``plot_geometry_3d``: cleaner layout with background grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from pylobe.visualization.polar import _setup_style, PALETTE

_NF_CMAP         = 'plasma'   # replaces 'hot'
_CURRENT_CSCALE  = 'Turbo'    # for 3-D current distribution


def plot_nearfield_2d(field_component: np.ndarray, x: np.ndarray,
                      y: np.ndarray, component_name: str,
                      freq: float, log_scale: bool = True,
                      show_phase: bool = False) -> plt.Figure:
    """Research-grade 2-D colour map of near-field distribution.

    Upgrades
    --------
    * ``'plasma'`` colormap (replaces ``'hot'``).
    * Optional ``show_phase`` panel showing the unwrapped phase alongside
      the magnitude — useful for understanding field structure.
    * Contour lines overlaid at −3 dB / −10 dB / −20 dB below peak.

    Parameters
    ----------
    field_component : ndarray, shape (Nx, Ny)
        Complex or real field component.
    x, y : ndarray
        Grid axes [m].
    component_name : str
        Field component label (e.g. ``'Ez'``, ``'|E|'``).
    freq : float
        Frequency [Hz].
    log_scale : bool
        Display ``20·log₁₀(|F| / max)`` when *True*.
    show_phase : bool
        Add a phase panel alongside the magnitude panel.

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    n_cols = 2 if show_phase else 1
    fig, axes = plt.subplots(1, n_cols,
                              figsize=(8.5 * n_cols, 6.5),
                              squeeze=False)
    ax_mag = axes[0, 0]

    field_mag = np.abs(field_component)
    if log_scale:
        f_max     = float(np.max(field_mag)) if float(np.max(field_mag)) > 0 else 1.0
        data_mag  = 20.0 * np.log10(np.clip(field_mag / f_max, 1e-5, None))
        cbar_lbl  = f'|{component_name}| (dB, normalised)'
        vmin, vmax = -40, 0
        contour_levels = [-3, -10, -20]
    else:
        data_mag  = field_mag
        cbar_lbl  = f'|{component_name}|'
        vmin, vmax = None, None
        contour_levels = []

    x_mm = x * 1e3
    y_mm = y * 1e3

    # ── Magnitude panel ───────────────────────────────────────────────────────
    im = ax_mag.pcolormesh(y_mm, x_mm, data_mag,
                           cmap=_NF_CMAP, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)

    # Contour overlays
    if contour_levels and log_scale:
        Xg, Yg = np.meshgrid(y_mm, x_mm)
        cs = ax_mag.contour(Xg, Yg, data_mag,
                            levels=contour_levels,
                            colors=['white', 'lightgrey', 'grey'],
                            linewidths=[1.5, 1.0, 0.8],
                            linestyles=['solid', 'dashed', 'dotted'])
        ax_mag.clabel(cs, inline=True, fontsize=7.5,
                      fmt={-3: '−3 dB', -10: '−10 dB', -20: '−20 dB'})

    div = make_axes_locatable(ax_mag)
    cax = div.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_lbl, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax_mag.set_xlabel('y (mm)', fontsize=12)
    ax_mag.set_ylabel('x (mm)', fontsize=12)
    ax_mag.set_title(
        f'Near-Field: {component_name} @ {freq / 1e9:.3f} GHz',
        fontsize=13, fontweight='bold', color='#2C3E50',
    )
    ax_mag.set_aspect('equal')
    ax_mag.tick_params(labelsize=10)

    # ── Phase panel (optional) ────────────────────────────────────────────────
    if show_phase and np.iscomplexobj(field_component):
        ax_ph = axes[0, 1]
        phase_deg = np.angle(field_component, deg=True)
        im_ph = ax_ph.pcolormesh(y_mm, x_mm, phase_deg,
                                  cmap='RdBu', vmin=-180, vmax=180,
                                  shading='auto', rasterized=True)
        div_ph = make_axes_locatable(ax_ph)
        cax_ph = div_ph.append_axes('right', size='4%', pad=0.08)
        cbar_ph = fig.colorbar(im_ph, cax=cax_ph)
        cbar_ph.set_label(f'Phase {component_name} (degrees)', fontsize=11)
        cbar_ph.set_ticks(range(-180, 181, 45))
        cbar_ph.ax.tick_params(labelsize=9)
        ax_ph.set_xlabel('y (mm)', fontsize=12)
        ax_ph.set_ylabel('x (mm)', fontsize=12)
        ax_ph.set_title(
            f'Phase: {component_name} @ {freq / 1e9:.3f} GHz',
            fontsize=13, fontweight='bold', color='#2C3E50',
        )
        ax_ph.set_aspect('equal')
        ax_ph.tick_params(labelsize=10)

    fig.tight_layout()
    return fig


def plot_current_distribution(geometry, current: np.ndarray,
                               freq: float,
                               title: str = 'Current Distribution') -> go.Figure:
    """Research-grade 3-D Plotly visualisation of wire current distribution.

    Upgrades
    --------
    * ``'Turbo'`` colorscale replaces manual RGB interpolation.
    * Segment thickness proportional to |I|.
    * Proper colorbar using a dedicated invisible trace.
    * Phase information shown in hover text.

    Parameters
    ----------
    geometry : AntennaGeometry
    current : ndarray of complex, shape (N_segments,) [A]
    freq : float
    title : str

    Returns
    -------
    go.Figure
    """
    verts = geometry.vertices * 1e3   # → mm
    mag   = np.abs(current)
    mag_n = mag / float(mag.max()) if float(mag.max()) > 0 else mag.copy()

    traces = []

    # Turbo colorscale sampled at 256 levels
    import plotly.express as px
    _turbo = px.colors.get_colorscale('Turbo')

    def _turbo_rgb(val: float) -> str:
        """Linearly interpolate Turbo colorscale at val ∈ [0, 1]."""
        idx = val * (len(_turbo) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(_turbo) - 1)
        t   = idx - lo
        c0  = _turbo[lo][1]
        c1  = _turbo[hi][1]
        # Parse 'rgb(r,g,b)' or '#rrggbb'
        def _parse(c):
            c = c.strip()
            if c.startswith('rgb'):
                parts = c[4:-1].split(',')
                return [float(p) for p in parts]
            else:
                r_v = int(c[1:3], 16)
                g_v = int(c[3:5], 16)
                b_v = int(c[5:7], 16)
                return [float(r_v), float(g_v), float(b_v)]
        rgb0 = _parse(c0)
        rgb1 = _parse(c1)
        r = int(rgb0[0] + t * (rgb1[0] - rgb0[0]))
        g = int(rgb0[1] + t * (rgb1[1] - rgb0[1]))
        b = int(rgb0[2] + t * (rgb1[2] - rgb0[2]))
        return f'rgb({r},{g},{b})'

    for idx, (i0, i1) in enumerate(geometry.edges):
        seg_idx = min(idx, len(current) - 1)
        val     = float(mag_n[seg_idx])
        colour  = _turbo_rgb(val)
        width   = 2.0 + 6.0 * val

        traces.append(go.Scatter3d(
            x=[float(verts[i0, 0]), float(verts[i1, 0])],
            y=[float(verts[i0, 1]), float(verts[i1, 1])],
            z=[float(verts[i0, 2]), float(verts[i1, 2])],
            mode='lines',
            line=dict(color=colour, width=width),
            showlegend=False,
            hovertext=(f'Segment {seg_idx}<br>'
                       f'|I| = {mag[seg_idx]:.4f} A<br>'
                       f'Phase = {np.angle(current[seg_idx], deg=True):.1f}°<br>'
                       f'Norm. = {val:.3f}'),
            hoverinfo='text',
        ))

    # Dedicated colorscale reference trace (invisible markers)
    if len(current) > 0:
        traces.append(go.Scatter3d(
            x=[float(verts[0, 0])] * 2,
            y=[float(verts[0, 1])] * 2,
            z=[float(verts[0, 2])] * 2,
            mode='markers',
            marker=dict(
                size=0.01,
                color=[0.0, float(np.max(mag))],
                colorscale=_CURRENT_CSCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(text='|I| (A)', side='right', font=dict(size=12)),
                    thickness=18,
                    tickfont=dict(size=10),
                    outlinewidth=1, outlinecolor='#888888',
                ),
            ),
            showlegend=False,
            hoverinfo='none',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'{title} @ {freq / 1e9:.3f} GHz',
                   font=dict(size=15, color='#2C3E50'), x=0.5),
        scene=dict(
            xaxis=dict(title='x (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            yaxis=dict(title='y (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            zaxis=dict(title='z (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            bgcolor='rgba(248,249,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.9)),
            aspectmode='data',
        ),
        autosize=True,
        height=640,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig


def plot_geometry_3d(geometry, title: str = 'Antenna Geometry') -> go.Figure:
    """3-D wireframe visualisation of antenna geometry.

    Upgrades
    --------
    * Cleaner background with subtle axis-plane colour.
    * Feed point rendered as a bright red cross-diamond with hover info.

    Parameters
    ----------
    geometry : AntennaGeometry
    title : str

    Returns
    -------
    go.Figure
    """
    verts  = geometry.vertices * 1e3  # → mm
    traces = []

    # Edges
    xs, ys, zs = [], [], []
    for i0, i1 in geometry.edges:
        xs.extend([float(verts[i0, 0]), float(verts[i1, 0]), None])
        ys.extend([float(verts[i0, 1]), float(verts[i1, 1]), None])
        zs.extend([float(verts[i0, 2]), float(verts[i1, 2]), None])

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='#2C3E50', width=3),
        name='Geometry',
        hoverinfo='skip',
    ))

    # Vertices
    traces.append(go.Scatter3d(
        x=verts[:, 0].tolist(),
        y=verts[:, 1].tolist(),
        z=verts[:, 2].tolist(),
        mode='markers',
        marker=dict(size=3.5, color='#2980B9',
                    line=dict(color='white', width=0.5)),
        name='Vertices',
        hovertext=[f'Vertex {i}: ({verts[i,0]:.2f}, {verts[i,1]:.2f}, '
                   f'{verts[i,2]:.2f}) mm'
                   for i in range(len(verts))],
        hoverinfo='text',
    ))

    # Feed point
    if geometry.feed_point is not None:
        fp = geometry.feed_point * 1e3
        traces.append(go.Scatter3d(
            x=[float(fp[0])], y=[float(fp[1])], z=[float(fp[2])],
            mode='markers+text',
            marker=dict(size=12, color='#E74C3C', symbol='cross',
                        line=dict(color='white', width=1.5)),
            text=['Feed'],
            textposition='top center',
            textfont=dict(size=10, color='#E74C3C'),
            name='Feed point',
            hovertext=f'Feed: ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f}) mm',
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#2C3E50'), x=0.5),
        scene=dict(
            xaxis=dict(title='x (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.5)'),
            yaxis=dict(title='y (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.5)'),
            zaxis=dict(title='z (mm)', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.5)'),
            bgcolor='rgba(250,251,252,1)',
            camera=dict(eye=dict(x=1.5, y=1.3, z=1.0)),
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.88)',
                    bordercolor='rgba(160,160,160,0.4)',
                    borderwidth=1),
        autosize=True,
        height=600,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig
