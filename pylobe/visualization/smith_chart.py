"""Research-grade interactive Smith chart (Plotly).

Upgrades over legacy version
-----------------------------
* VSWR reference circles at 1.5 / 2 / 3 / 5 (dashed red, labelled).
* Constant-Q locus circles at Q = 0.5 / 1 / 2 / 5 (dashed purple, labelled).
* Impedance trajectory drawn with a colour gradient (low-freq → high-freq)
  **and** directional arrows to show the sweep direction.
* Frequency annotations at up to 5 evenly spaced points on the trajectory.
* Improved grid: r-circle labels repositioned so they never overlap arcs.
* Dark-bordered unit circle; real-axis line rendered precisely.
"""
import numpy as np
import plotly.graph_objects as go
from pylobe.analysis.smith import SmithChart as SmithCalc
from pylobe.constants import PI

# ── Grid settings ─────────────────────────────────────────────────────────────
_R_VALS      = [0, 0.2, 0.5, 1.0, 2.0, 5.0]
_X_VALS      = [0.2, 0.5, 1.0, 2.0, 5.0, -0.2, -0.5, -1.0, -2.0, -5.0]
_VSWR_LEVELS = [1.5, 2.0, 3.0, 5.0]
_Q_LEVELS    = [0.5, 1.0, 2.0, 5.0]


def _vswr_to_gamma(vswr: float) -> float:
    return (vswr - 1.0) / (vswr + 1.0)


def _q_locus(q: float, Z0: float = 50.0, n_pts: int = 400) -> np.ndarray:
    """Gamma locus for constant Q = Im(Z)/Re(Z) (both signs)."""
    R_arr = np.logspace(-3, 3, n_pts) * Z0
    Z_pos = R_arr * (1.0 + 1j * q)
    Z_neg = R_arr * (1.0 - 1j * q)
    g_pos = (Z_pos - Z0) / (Z_pos + Z0)
    g_neg = (Z_neg - Z0) / (Z_neg + Z0)
    return g_pos, g_neg


def plot_smith_chart(gamma_trace: np.ndarray = None,
                     impedance_trace: np.ndarray = None,
                     Z0: float = 50.0,
                     freq_labels: np.ndarray = None,
                     title: str = 'Smith Chart') -> go.Figure:
    """Full interactive Smith chart with impedance/reflection trajectory.

    Grid lines
    ----------
    * Constant-r circles : r = 0, 0.2, 0.5, 1, 2, 5
    * Constant-x arcs    : x = ±0.2, ±0.5, ±1, ±2, ±5
    * VSWR circles        : VSWR = 1.5, 2, 3, 5   (dashed red)
    * Constant-Q loci     : Q = ±0.5, ±1, ±2, ±5  (dashed purple)

    Trajectory
    ----------
    * Colour-coded blue → red (low-freq → high-freq).
    * Directional arrows every ~20% of the trajectory length.
    * Up to 5 frequency annotations at evenly spaced points.

    Parameters
    ----------
    gamma_trace : ndarray of complex or None
    impedance_trace : ndarray of complex or None
    Z0 : float
    freq_labels : ndarray or None  [Hz]
    title : str

    Returns
    -------
    go.Figure
    """
    sc     = SmithCalc(Z0=Z0)
    traces = []

    # ── Unit circle (boundary) ────────────────────────────────────────────────
    phi = np.linspace(0, 2 * PI, 500)
    traces.append(go.Scattergl(
        x=np.cos(phi), y=np.sin(phi),
        mode='lines',
        line=dict(color='#1A252F', width=2.0),
        showlegend=False, hoverinfo='none',
    ))

    # ── Constant-r circles ────────────────────────────────────────────────────
    for r_val in _R_VALS:
        gamma = sc.constant_r_circle(r_val)
        inside = np.abs(gamma) <= 1.002
        traces.append(go.Scattergl(
            x=gamma[inside].real,
            y=gamma[inside].imag,
            mode='lines',
            line=dict(color='rgba(120,130,140,0.55)', width=0.8),
            showlegend=False, hoverinfo='none',
        ))
        # Label slightly above the r-circle centre on the real axis
        g_center = float(r_val) / (float(r_val) + 1.0) if r_val < 1e8 else 1.0
        traces.append(go.Scattergl(
            x=[g_center], y=[0.04],
            mode='text',
            text=[f'r={r_val}'],
            textfont=dict(size=8, color='#555E6A'),
            showlegend=False, hoverinfo='none',
        ))

    # ── Constant-x arcs ───────────────────────────────────────────────────────
    for x_val in _X_VALS:
        gamma = sc.constant_x_arc(x_val)
        inside = np.abs(gamma) <= 1.002
        traces.append(go.Scattergl(
            x=gamma[inside].real,
            y=gamma[inside].imag,
            mode='lines',
            line=dict(color='rgba(120,130,140,0.45)', width=0.8),
            showlegend=False, hoverinfo='none',
        ))
        # Label near the outer rim
        if inside.any():
            rim_idx = int(np.argmax(np.abs(gamma[inside])))
            gp = gamma[inside][rim_idx]
            traces.append(go.Scattergl(
                x=[float(gp.real)], y=[float(gp.imag)],
                mode='text',
                text=[f'j{x_val:+.1f}' if abs(x_val) != int(abs(x_val))
                      else f'j{int(x_val):+d}'],
                textfont=dict(size=7.5, color='#555E6A'),
                showlegend=False, hoverinfo='none',
            ))

    # ── Real axis ─────────────────────────────────────────────────────────────
    traces.append(go.Scattergl(
        x=[-1, 1], y=[0, 0],
        mode='lines',
        line=dict(color='#2C3E50', width=0.9),
        showlegend=False, hoverinfo='none',
    ))

    # ── VSWR reference circles ────────────────────────────────────────────────
    for vswr in _VSWR_LEVELS:
        gm = _vswr_to_gamma(vswr)
        circle_phi = np.linspace(0, 2 * PI, 300)
        xc = gm * np.cos(circle_phi)
        yc = gm * np.sin(circle_phi)
        traces.append(go.Scattergl(
            x=xc, y=yc,
            mode='lines',
            line=dict(color='rgba(192,57,43,0.35)', width=1.0, dash='dot'),
            showlegend=False, hoverinfo='none',
        ))
        # Label at top of circle
        traces.append(go.Scattergl(
            x=[0.0], y=[gm + 0.04],
            mode='text',
            text=[f'VSWR={vswr:.1g}'],
            textfont=dict(size=7.5, color='rgba(192,57,43,0.75)'),
            showlegend=False, hoverinfo='none',
        ))

    # ── Constant-Q loci ───────────────────────────────────────────────────────
    for q_val in _Q_LEVELS:
        g_pos, g_neg = _q_locus(q_val, Z0=Z0)
        inside_p = np.abs(g_pos) <= 1.002
        inside_n = np.abs(g_neg) <= 1.002
        for gq, inside, label_sign in [(g_pos, inside_p, '+'),
                                        (g_neg, inside_n, '−')]:
            if not inside.any():
                continue
            traces.append(go.Scattergl(
                x=gq[inside].real,
                y=gq[inside].imag,
                mode='lines',
                line=dict(color='rgba(142,68,173,0.28)',
                          width=0.9, dash='dashdot'),
                showlegend=False, hoverinfo='none',
            ))
        # Single label for each |Q|
        rim_p_idx = int(np.argmax(np.abs(g_pos[inside_p]))) if inside_p.any() else 0
        if inside_p.any():
            gq_label = g_pos[inside_p][rim_p_idx]
            traces.append(go.Scattergl(
                x=[float(gq_label.real)], y=[float(gq_label.imag) + 0.04],
                mode='text',
                text=[f'Q={q_val:.1g}'],
                textfont=dict(size=7.5, color='rgba(142,68,173,0.70)'),
                showlegend=False, hoverinfo='none',
            ))

    # ── Match point ───────────────────────────────────────────────────────────
    traces.append(go.Scattergl(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=8, color='#27AE60', symbol='circle',
                    line=dict(color='white', width=1.2)),
        name='Match (Γ = 0)',
        hoverinfo='name',
    ))

    # ── Impedance trajectory ──────────────────────────────────────────────────
    if impedance_trace is not None:
        gamma_trace = sc.impedance_to_gamma(np.asarray(impedance_trace))

    if gamma_trace is not None:
        gamma_trace = np.asarray(gamma_trace, dtype=complex)
        g_re = gamma_trace.real
        g_im = gamma_trace.imag
        n_pts = len(gamma_trace)
        color_arr = np.linspace(0, 1, n_pts)

        if freq_labels is not None and len(freq_labels) == n_pts:
            hover_text = [
                f'f = {freq_labels[i] / 1e9:.3f} GHz<br>'
                f'Γ = {gamma_trace[i].real:.3f} {gamma_trace[i].imag:+.3f}j<br>'
                f'Z = {sc.gamma_to_impedance(gamma_trace[i]).real:.1f}'
                f'{sc.gamma_to_impedance(gamma_trace[i]).imag:+.1f}j Ω<br>'
                f'VSWR = {sc.vswr(gamma_trace[i]):.2f}'
                for i in range(n_pts)
            ]
        else:
            hover_text = [
                f'Γ = {g.real:.3f} {g.imag:+.3f}j<br>'
                f'Z = {sc.gamma_to_impedance(g).real:.1f}'
                f'{sc.gamma_to_impedance(g).imag:+.1f}j Ω'
                for g in gamma_trace
            ]

        # Trajectory line
        traces.append(go.Scattergl(
            x=g_re, y=g_im,
            mode='lines',
            line=dict(color='#2C3E50', width=2.8),
            showlegend=False, hoverinfo='none',
        ))

        # Frequency-coloured marker set
        traces.append(go.Scattergl(
            x=g_re, y=g_im,
            mode='markers',
            marker=dict(
                size=5,
                color=color_arr,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title=dict(text='Frequency', font=dict(size=11)),
                    thickness=16,
                    tickvals=[0.0, 0.5, 1.0],
                    ticktext=['f_lo', 'f_c', 'f_hi'],
                    tickfont=dict(size=10),
                    len=0.5,
                    y=0.5,
                ),
            ),
            text=hover_text,
            hoverinfo='text',
            name='Impedance trajectory',
        ))

        # ── Directional arrows (every ~20% of trajectory) ─────────────────────
        arrow_indices = np.linspace(1, n_pts - 2, 5, dtype=int)
        for ai in arrow_indices:
            dx = float(g_re[ai + 1] - g_re[ai - 1])
            dy = float(g_im[ai + 1] - g_im[ai - 1])
            length = (dx ** 2 + dy ** 2) ** 0.5 + 1e-12
            scale = 0.04
            traces.append(go.Scattergl(
                x=[float(g_re[ai]), float(g_re[ai]) + scale * dx / length],
                y=[float(g_im[ai]), float(g_im[ai]) + scale * dy / length],
                mode='lines',
                line=dict(color='#2C3E50', width=3),
                showlegend=False, hoverinfo='none',
            ))

        # ── Frequency annotations at 5 evenly spaced points ──────────────────
        if freq_labels is not None and len(freq_labels) == n_pts:
            ann_indices = np.linspace(0, n_pts - 1, 5, dtype=int)
            for ai in ann_indices:
                f_ghz = float(freq_labels[ai]) / 1e9
                traces.append(go.Scattergl(
                    x=[float(g_re[ai])],
                    y=[float(g_im[ai])],
                    mode='markers+text',
                    marker=dict(size=9, color='white',
                                symbol='diamond',
                                line=dict(color='#2C3E50', width=1.4)),
                    text=[f'{f_ghz:.2f}G'],
                    textposition='top right',
                    textfont=dict(size=8, color='#2C3E50'),
                    showlegend=False,
                    hoverinfo='skip',
                ))

        # Start / end markers
        traces.append(go.Scattergl(
            x=[float(g_re[0])], y=[float(g_im[0])],
            mode='markers',
            marker=dict(size=13, color='#27AE60', symbol='circle',
                        line=dict(color='white', width=1.5)),
            name='Start (f_lo)',
        ))
        traces.append(go.Scattergl(
            x=[float(g_re[-1])], y=[float(g_im[-1])],
            mode='markers',
            marker=dict(size=13, color='#E74C3C', symbol='square',
                        line=dict(color='white', width=1.5)),
            name='End (f_hi)',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title,
                   font=dict(size=16, color='#2C3E50', family='Arial'),
                   x=0.5, xanchor='center'),
        xaxis=dict(
            range=[-1.12, 1.12],
            scaleanchor='y', scaleratio=1,
            zeroline=False, showgrid=False,
            title='Re(Γ)', tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[-1.12, 1.12],
            zeroline=False, showgrid=False,
            title='Im(Γ)', tickfont=dict(size=10),
        ),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='white',
        autosize=True,
        height=720,
        legend=dict(
            x=1.10, y=0.95,
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='rgba(160,160,160,0.5)',
            borderwidth=1,
            font=dict(size=11),
        ),
        font=dict(family='Arial', size=12),
        margin=dict(l=60, r=180, t=60, b=60),
    )
    return fig
