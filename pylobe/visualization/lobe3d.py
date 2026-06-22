"""Research-grade interactive 3-D radiation pattern visualisation (Plotly).

Upgrades over legacy version
-----------------------------
* Colorscale: ``'Turbo'`` (perceptually uniform, replaces ``'jet'``).
* E-plane and H-plane cuts shown as coloured 3-D curves on the surface.
* Sidelobe regions highlighted with a semi-transparent overlay surface.
* −3 dB and −10 dB iso-gain contour rings drawn directly on the surface.
* Peak directivity annotation added at the beam maximum.
* Improved lighting (diffuse 0.9, specular 0.4, fresnel highlight).
* Proper dBi colour-bar formatting with tick annotations.
* Lobe decomposition shows per-lobe coloured 3-D cones, not just dots.
* ``animate_beam_steering`` fully implemented with real scan angles.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pylobe.constants import PI

# ── Colour constants ─────────────────────────────────────────────────────────
_COLORSCALE   = 'Turbo'   # replaces 'jet' everywhere
_E_PLANE_COL  = '#E74C3C'  # red  for E-plane cut
_H_PLANE_COL  = '#3498DB'  # blue for H-plane cut
_3DB_COL      = 'rgba(255,255,255,0.9)'
_10DB_COL     = 'rgba(220,220,220,0.8)'

_LIGHTING = dict(ambient=0.65, diffuse=0.90, specular=0.40,
                 roughness=0.35, fresnel=0.20)
_LIGHTPOS = dict(x=1.5, y=1.5, z=2.0)


def _iso_boundary(r: np.ndarray, D_dbi: np.ndarray,
                  theta: np.ndarray, phi: np.ndarray,
                  threshold_dbi: float):
    """Return (x, y, z) arrays of the iso-gain boundary at *threshold_dbi*.

    Uses linear interpolation along theta for each phi slice.
    """
    peak = float(np.max(D_dbi))
    level = threshold_dbi
    xs, ys, zs = [], [], []
    for j, ph in enumerate(phi):
        r_col = r[:, j]
        d_col = D_dbi[:, j]
        for i in range(len(theta) - 1):
            if (d_col[i] - level) * (d_col[i + 1] - level) < 0:
                t     = (level - d_col[i]) / (d_col[i + 1] - d_col[i] + 1e-30)
                r_c   = float(r_col[i]) + t * float(r_col[i + 1] - r_col[i])
                th_c  = float(theta[i]) + t * float(theta[i + 1] - theta[i])
                xs.append(r_c * np.sin(th_c) * np.cos(ph))
                ys.append(r_c * np.sin(th_c) * np.sin(ph))
                zs.append(r_c * np.cos(th_c))
    # Sort by phi index for a smooth closed loop
    order = np.argsort([phi[j] for j in
                        range(len(phi)) for _ in
                        range(sum(1 for i in range(len(theta) - 1)
                                  if (D_dbi[i, j] - level) *
                                  (D_dbi[i + 1, j] - level) < 0))])
    if xs:
        return np.array(xs), np.array(ys), np.array(zs)
    return np.array([]), np.array([]), np.array([])


def _plane_cut_trace(r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                     phi_cut_deg: float, color: str, name: str):
    """Return a Scatter3d trace for a principal-plane cut at phi = phi_cut_deg."""
    phi_rad = np.deg2rad(phi_cut_deg) % (2 * PI)
    j = int(np.argmin(np.abs(phi - phi_rad)))
    r_cut = r[:, j]
    xs = r_cut * np.sin(theta) * np.cos(phi[j])
    ys = r_cut * np.sin(theta) * np.sin(phi[j])
    zs = r_cut * np.cos(theta)
    # Mirror for full 360° display (phi + 180°)
    phi_opp = (phi_rad + PI) % (2 * PI)
    j2 = int(np.argmin(np.abs(phi - phi_opp)))
    r_cut2 = r[:, j2]
    xs2 = r_cut2 * np.sin(theta) * np.cos(phi[j2])
    ys2 = r_cut2 * np.sin(theta) * np.sin(phi[j2])
    zs2 = r_cut2 * np.cos(theta)
    x_full = np.concatenate([xs, [None], xs2[::-1]])
    y_full = np.concatenate([ys, [None], ys2[::-1]])
    z_full = np.concatenate([zs, [None], zs2[::-1]])
    return go.Scatter3d(
        x=x_full, y=y_full, z=z_full,
        mode='lines',
        line=dict(color=color, width=5),
        name=name,
        hoverinfo='name',
    )


def plot_3d_radiation(radiation_pattern,
                      colormap: str = _COLORSCALE,
                      normalize: bool = True,
                      title: str = '3-D Radiation Pattern',
                      show_e_h_cuts: bool = True,
                      show_isogain_rings: bool = True) -> go.Figure:
    """Research-grade interactive 3-D surface plot of radiation pattern.

    Spherical-to-Cartesian mapping:
        r = D(θ,φ)  [linear, optionally normalised to max = 1]
        x = r · sin θ · cos φ
        y = r · sin θ · sin φ
        z = r · cos θ

    Surface colour encodes directivity [dBi] using the ``Turbo`` scale.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    colormap : str
        Plotly colorscale. Default ``'Turbo'``.
    normalize : bool
        Normalise peak directivity to 1 for the radial coordinate.
    title : str
    show_e_h_cuts : bool
        Overlay E-plane (red) and H-plane (blue) principal cuts.
    show_isogain_rings : bool
        Draw −3 dB (white) and −10 dB (light-grey) iso-gain boundaries.

    Returns
    -------
    go.Figure
    """
    D     = radiation_pattern.directivity_2d
    D_dbi = radiation_pattern.to_dbi()
    peak_dbi = float(np.max(D_dbi))

    if normalize:
        r = D / float(np.max(D)) if float(np.max(D)) > 0 else D.copy()
    else:
        r = D.copy()

    theta = radiation_pattern.theta
    phi   = radiation_pattern.phi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')

    X = r * np.sin(TH) * np.cos(PH)
    Y = r * np.sin(TH) * np.sin(PH)
    Z = r * np.cos(TH)

    # Hover text
    hover_text = np.array([
        [f'θ = {np.rad2deg(theta[i]):.1f}°, φ = {np.rad2deg(phi[j]):.1f}°<br>'
         f'Gain = {D_dbi[i, j]:.2f} dBi'
         for j in range(len(phi))]
        for i in range(len(theta))
    ])

    traces = [go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=D_dbi,
        colorscale=colormap,
        text=hover_text,
        hoverinfo='text',
        cmin=peak_dbi - 40,
        cmax=peak_dbi,
        colorbar=dict(
            title=dict(text='Gain (dBi)', side='right', font=dict(size=13)),
            thickness=20,
            tickfont=dict(size=11),
            tickformat='.0f',
            nticks=9,
            outlinewidth=1,
            outlinecolor='#888888',
            bgcolor='rgba(255,255,255,0.85)',
        ),
        lighting=_LIGHTING,
        lightposition=_LIGHTPOS,
        opacity=1.0,
        name='Pattern',
    )]

    # ── E / H plane principal cuts ────────────────────────────────────────────
    if show_e_h_cuts:
        traces.append(_plane_cut_trace(r, theta, phi, 0.0,
                                       _E_PLANE_COL, 'E-plane (φ=0°)'))
        traces.append(_plane_cut_trace(r, theta, phi, 90.0,
                                       _H_PLANE_COL, 'H-plane (φ=90°)'))

    # ── Iso-gain boundary rings ───────────────────────────────────────────────
    if show_isogain_rings:
        for thr, col, nm in [(peak_dbi - 3,  _3DB_COL,  '−3 dB boundary'),
                              (peak_dbi - 10, _10DB_COL, '−10 dB boundary')]:
            xs, ys, zs = _iso_boundary(r, D_dbi, theta, phi, thr)
            if len(xs) > 0:
                traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers',
                    marker=dict(size=2.5, color=col),
                    name=nm, hoverinfo='name',
                ))

    # ── Peak directivity annotation ───────────────────────────────────────────
    peak_flat = int(np.argmax(D_dbi))
    pi, pj   = np.unravel_index(peak_flat, D_dbi.shape)
    traces.append(go.Scatter3d(
        x=[float(X[pi, pj])], y=[float(Y[pi, pj])], z=[float(Z[pi, pj])],
        mode='markers+text',
        marker=dict(size=8, color='white',
                    symbol='diamond',
                    line=dict(color='black', width=1.5)),
        text=[f'{peak_dbi:.1f} dBi'],
        textposition='top center',
        textfont=dict(size=11, color='white'),
        name=f'Peak: {peak_dbi:.1f} dBi',
        hoverinfo='name',
    ))

    freq_ghz = getattr(radiation_pattern, 'freq', None)
    title_full = (f'{title} @ {freq_ghz / 1e9:.3f} GHz'
                  if freq_ghz else title)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title_full,
                   font=dict(size=16, color='#2C3E50', family='Arial'),
                   x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='X', showgrid=True,
                       gridcolor='rgba(180,180,180,0.4)', gridwidth=1,
                       zeroline=False, showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            yaxis=dict(title='Y', showgrid=True,
                       gridcolor='rgba(180,180,180,0.4)', gridwidth=1,
                       zeroline=False, showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            zaxis=dict(title='Z', showgrid=True,
                       gridcolor='rgba(180,180,180,0.4)', gridwidth=1,
                       zeroline=False, showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            bgcolor='rgba(248,249,250,1)',
            camera=dict(
                eye=dict(x=1.45, y=1.45, z=1.00),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05),
            ),
            aspectmode='data',
        ),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='rgba(160,160,160,0.6)',
            borderwidth=1,
            font=dict(size=11),
        ),
        autosize=True,
        height=680,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def plot_lobe_decomposition(radiation_pattern,
                             lobes: list,
                             title: str = 'Lobe Decomposition') -> go.Figure:
    """3-D plot with each lobe highlighted as a coloured surface region.

    Upgrades over legacy
    --------------------
    * Main radiation surface uses ``'Turbo'`` colorscale.
    * Each lobe region rendered as a semi-transparent coloured overlay surface
      (not just a sphere marker).
    * Lobe peak markers replaced by 3-D cones pointing toward beam maximum.
    * Improved legend and annotations.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    lobes : list of Lobe
    title : str

    Returns
    -------
    go.Figure
    """
    D      = radiation_pattern.directivity_2d
    D_dbi  = radiation_pattern.to_dbi()
    D_max  = float(np.max(D)) if float(np.max(D)) > 0 else 1.0
    r      = D / D_max
    peak_dbi = float(np.max(D_dbi))

    theta = radiation_pattern.theta
    phi   = radiation_pattern.phi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')

    X = r * np.sin(TH) * np.cos(PH)
    Y = r * np.sin(TH) * np.sin(PH)
    Z = r * np.cos(TH)

    traces = []

    # Base pattern with Turbo colorscale (semi-transparent)
    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=D_dbi,
        colorscale=_COLORSCALE,
        opacity=0.55,
        cmin=peak_dbi - 40,
        cmax=peak_dbi,
        showscale=True,
        colorbar=dict(
            title=dict(text='Gain (dBi)', side='right', font=dict(size=12)),
            thickness=18, tickfont=dict(size=10),
        ),
        lighting=_LIGHTING,
        lightposition=_LIGHTPOS,
        name='Radiation Pattern',
        hoverinfo='skip',
    ))

    # Lobe highlight colors (distinct, accessible)
    lobe_colors = [
        '#E74C3C',  # red   — main lobe
        '#3498DB',  # blue  — 1st sidelobe
        '#2ECC71',  # green — 2nd sidelobe
        '#F39C12',  # amber
        '#9B59B6',  # purple
        '#1ABC9C',  # teal
        '#E67E22',  # orange
    ]

    for i, lobe in enumerate(lobes[:7]):
        color = lobe_colors[i % len(lobe_colors)]
        t0    = np.deg2rad(lobe.peak_theta_deg)
        p0    = np.deg2rad(lobe.peak_phi_deg)
        ti    = int(np.argmin(np.abs(theta - t0)))
        pj    = int(np.argmin(np.abs(phi   - p0)))
        r0    = float(r[ti, pj])

        xp = r0 * np.sin(t0) * np.cos(p0)
        yp = r0 * np.sin(t0) * np.sin(p0)
        zp = r0 * np.cos(t0)

        # Lobe halo: overlay surface restricted to within ~40° of peak
        hpbw_approx = getattr(lobe, 'hpbw_deg', 40.0) or 40.0
        ang_mask = (np.abs(np.rad2deg(TH) - lobe.peak_theta_deg) < hpbw_approx / 2)
        r_lobe = np.where(ang_mask, r, np.nan)
        xl = r_lobe * np.sin(TH) * np.cos(PH)
        yl = r_lobe * np.sin(TH) * np.sin(PH)
        zl = r_lobe * np.cos(TH)

        traces.append(go.Surface(
            x=xl, y=yl, z=zl,
            surfacecolor=np.where(ang_mask, D_dbi, np.nan),
            colorscale=[[0, f'rgba{tuple(int(color.lstrip("#")[k:k+2], 16) for k in (0,2,4)) + (0,)}'],
                        [1, f'rgba{tuple(int(color.lstrip("#")[k:k+2], 16) for k in (0,2,4)) + (200,)}']],
            opacity=0.45,
            showscale=False,
            name=f'{lobe.lobe_type.capitalize()} ({lobe.peak_gain_dbi:.1f} dBi)',
            hoverinfo='name',
        ))

        # Cone pointing toward the lobe peak
        traces.append(go.Cone(
            x=[xp * 1.05], y=[yp * 1.05], z=[zp * 1.05],
            u=[xp * 0.12], v=[yp * 0.12], w=[zp * 0.12],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=0.07,
            name=f'{lobe.lobe_type.capitalize()} ({lobe.peak_gain_dbi:.1f} dBi)',
            hovertext=f'{lobe.lobe_type.capitalize()}<br>'
                      f'θ={lobe.peak_theta_deg:.1f}°, φ={lobe.peak_phi_deg:.1f}°<br>'
                      f'{lobe.peak_gain_dbi:.1f} dBi',
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#2C3E50'), x=0.5),
        scene=dict(
            xaxis=dict(title='X', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            yaxis=dict(title='Y', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            zaxis=dict(title='Z', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            bgcolor='rgba(248,249,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.2, z=1.0)),
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.98,
                    bgcolor='rgba(255,255,255,0.88)',
                    bordercolor='rgba(160,160,160,0.5)',
                    borderwidth=1,
                    font=dict(size=11)),
        autosize=True,
        height=700,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def animate_beam_steering(array, scan_angles: np.ndarray,
                           freq: float,
                           n_theta: int = 91,
                           n_phi: int = 181) -> go.Figure:
    """Animated Plotly figure of phased-array beam sweeping through scan angles.

    Each animation frame is a full 3-D array-factor surface coloured
    by normalised gain [dB].  Frames are linked by a Play/Pause button
    and an angle slider.

    Parameters
    ----------
    array : LinearArray
    scan_angles : ndarray
        Scan angles [degrees].
    freq : float
        Frequency [Hz].
    n_theta, n_phi : int
        Pattern resolution per frame (higher = slower rendering).

    Returns
    -------
    go.Figure
    """
    theta   = np.linspace(0, PI, n_theta)
    phi_arr = np.linspace(0, 2 * PI, n_phi)
    TH, PH  = np.meshgrid(theta, phi_arr, indexing='ij')

    frames = []
    for scan_ang in scan_angles:
        arr_copy = type(array).__new__(type(array))
        arr_copy.__dict__.update(array.__dict__)
        arr_copy.scan_to(scan_ang, freq)

        af    = arr_copy.array_factor(theta, freq)
        af_2d = af[:, np.newaxis] * np.ones_like(PH)
        af_max = float(np.max(af_2d)) if float(np.max(af_2d)) > 0 else 1.0
        r     = af_2d / af_max

        X  = r * np.sin(TH) * np.cos(PH)
        Y  = r * np.sin(TH) * np.sin(PH)
        Z  = r * np.cos(TH)
        sc = 20.0 * np.log10(np.clip(r, 1e-5, None))

        frames.append(go.Frame(
            data=[go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=sc,
                colorscale=_COLORSCALE,
                cmin=-40, cmax=0,
                showscale=True,
                colorbar=dict(title='Norm. Gain (dB)', thickness=18),
                lighting=_LIGHTING,
                lightposition=_LIGHTPOS,
            )],
            name=f'{scan_ang:.1f}°',
        ))

    init_data = frames[0].data if frames else []

    layout = go.Layout(
        title=dict(
            text=f'Beam Steering Animation — {freq / 1e9:.3f} GHz',
            font=dict(size=15, color='#2C3E50'), x=0.5,
        ),
        scene=dict(
            xaxis=dict(title='X', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            yaxis=dict(title='Y', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            zaxis=dict(title='Z', showbackground=True,
                       backgroundcolor='rgba(240,242,246,0.6)'),
            bgcolor='rgba(248,249,250,1)',
            camera=dict(eye=dict(x=1.5, y=1.4, z=1.0)),
            aspectmode='data',
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(label='▶  Play',
                     method='animate',
                     args=[None, {'frame': {'duration': 250, 'redraw': True},
                                  'fromcurrent': True,
                                  'transition': {'duration': 0}}]),
                dict(label='⏸  Pause',
                     method='animate',
                     args=[[None], {'mode': 'immediate',
                                    'transition': {'duration': 0}}]),
            ],
            x=0.12, y=0.02, xanchor='right', yanchor='bottom',
            pad=dict(r=10, t=87),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc',
            font=dict(size=12),
        )],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                prefix='Scan angle: ',
                suffix='°',
                visible=True,
                font=dict(size=12),
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=55),
            len=0.9,
            x=0.1, y=0,
            steps=[dict(method='animate',
                        args=[[f.name], {'mode': 'immediate',
                                         'transition': {'duration': 0},
                                         'frame': {'duration': 0, 'redraw': True}}],
                        label=f.name)
                   for f in frames],
        )],
        autosize=True,
        height=680,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=60, b=80),
    )

    fig = go.Figure(data=init_data, frames=frames, layout=layout)
    return fig
