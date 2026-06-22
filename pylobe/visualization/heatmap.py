"""Research-grade gain heatmap, S11, VSWR, and impedance frequency-response plots.

Upgrades over legacy version
-----------------------------
* Gain heatmap: ``'Turbo'`` colorscale; −3 dB / −10 dB / −20 dB contour
  overlays; proper tick spacing; peak annotation with full θ/φ info.
* S11 plot: fractional-bandwidth annotation; multiple resonance detection;
  return-loss at each resonance annotated; group-delay stub.
* VSWR plot: matched-band fill; cleaner threshold annotation.
* Impedance plot: improved dual-axis; resonance + anti-resonance markers.
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pylobe.visualization.polar import _setup_style, PALETTE
from pylobe.constants import C0

# ── Colorscale ───────────────────────────────────────────────────────────────
_COLORSCALE = 'Turbo'   # replaces 'Jet'


def plot_gain_heatmap(radiation_pattern,
                      title: str = 'Gain Pattern (dBi)',
                      contour_levels: tuple = (-3, -10, -20)) -> go.Figure:
    """Research-grade interactive 2-D gain heatmap (θ vs φ).

    Upgrades
    --------
    * ``'Turbo'`` colorscale (perceptually uniform, replaces ``'Jet'``).
    * Contour overlays at ``contour_levels`` dB relative to peak (white lines).
    * Peak annotated with θ / φ / gain text.
    * Colour-bar tick labels show absolute dBi values.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    title : str
    contour_levels : tuple of float
        Gain offsets [dB] from peak at which to draw iso-gain contours.

    Returns
    -------
    go.Figure
    """
    D_dbi     = radiation_pattern.to_dbi()
    theta_deg = np.rad2deg(radiation_pattern.theta)
    phi_deg   = np.rad2deg(radiation_pattern.phi)
    peak_dbi  = float(np.max(D_dbi))

    TH_d, PH_d = np.meshgrid(theta_deg, phi_deg, indexing='ij')
    hover = np.array([
        [f'θ = {TH_d[i, j]:.1f}°, φ = {PH_d[i, j]:.1f}°<br>'
         f'Gain = {D_dbi[i, j]:.2f} dBi'
         for j in range(len(phi_deg))]
        for i in range(len(theta_deg))
    ])

    traces = []

    # ── Heatmap ──────────────────────────────────────────────────────────────
    traces.append(go.Heatmap(
        z=D_dbi,
        x=phi_deg,
        y=theta_deg,
        colorscale=_COLORSCALE,
        zmin=peak_dbi - 40,
        zmax=peak_dbi,
        text=hover,
        hoverinfo='text',
        colorbar=dict(
            title=dict(text='Gain (dBi)', side='right',
                       font=dict(size=13)),
            thickness=22,
            tickfont=dict(size=11),
            tickformat='.0f',
            nticks=9,
            outlinewidth=1,
            outlinecolor='#888888',
        ),
        zsmooth='best',
    ))

    # ── Contour overlays ─────────────────────────────────────────────────────
    contour_colors = {
        -3:  'rgba(255,255,255,0.95)',  # white
        -10: 'rgba(220,220,220,0.85)',  # light grey
        -20: 'rgba(180,180,180,0.70)',  # mid grey
    }
    for offset in contour_levels:
        level = peak_dbi + offset
        col   = contour_colors.get(offset, 'rgba(200,200,200,0.70)')
        dash  = 'solid' if offset == -3 else ('dash' if offset == -10 else 'dot')
        traces.append(go.Contour(
            z=D_dbi,
            x=phi_deg,
            y=theta_deg,
            contours=dict(
                start=level, end=level, size=0,
                coloring='none',
            ),
            line=dict(width=2.0, color=col, dash=dash),
            showscale=False,
            showlegend=True,
            name=f'{offset:+.0f} dB contour',
            hoverinfo='skip',
        ))

    # ── Peak marker ───────────────────────────────────────────────────────────
    peak_idx = np.unravel_index(np.argmax(D_dbi), D_dbi.shape)
    pi, pj   = peak_idx
    traces.append(go.Scatter(
        x=[phi_deg[pj]],
        y=[theta_deg[pi]],
        mode='markers+text',
        marker=dict(symbol='cross-thin-open', size=16,
                    color='white', line=dict(color='white', width=2.5)),
        text=[f'{peak_dbi:.1f} dBi'],
        textposition='top center',
        textfont=dict(size=11, color='white'),
        name=f'Peak: {peak_dbi:.1f} dBi @ θ={theta_deg[pi]:.1f}°, φ={phi_deg[pj]:.1f}°',
        hoverinfo='name',
    ))

    freq_ghz  = getattr(radiation_pattern, 'freq', None)
    title_str = (f'{title} @ {freq_ghz / 1e9:.3f} GHz' if freq_ghz else title)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title_str,
                   font=dict(size=16, color='#2C3E50', family='Arial'),
                   x=0.5, xanchor='center'),
        xaxis=dict(
            title='φ (degrees)', tickfont=dict(size=11),
            dtick=30, range=[float(phi_deg[0]), float(phi_deg[-1])],
            showgrid=False,
        ),
        yaxis=dict(
            title='θ (degrees)', tickfont=dict(size=11),
            autorange='reversed', dtick=15,
            range=[float(theta_deg[-1]), float(theta_deg[0])],
            showgrid=False,
        ),
        autosize=True,
        height=560,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial', size=12),
        legend=dict(
            x=1.14, y=0.98,
            bgcolor='rgba(255,255,255,0.90)',
            bordercolor='rgba(160,160,160,0.5)',
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )
    return fig


def plot_s11_vs_freq(freq: np.ndarray, S11_db: np.ndarray,
                     bandwidth_threshold: float = -10.0,
                     title: str = 'S₁₁ vs Frequency') -> plt.Figure:
    """Research-grade S₁₁ frequency response plot.

    Upgrades
    --------
    * Detects and annotates all resonances (local minima below threshold).
    * Fractional-bandwidth annotation for each band.
    * Upper bound annotation at 0 dB.

    Parameters
    ----------
    freq : ndarray [Hz]
    S11_db : ndarray [dB]
    bandwidth_threshold : float
        Threshold for bandwidth shading and BW annotation [dB]. Default −10.
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    freq_ghz = freq / 1e9
    ax.plot(freq_ghz, S11_db, color=PALETTE[0], linewidth=2.2, label='|S₁₁|',
            zorder=4)

    # ── Threshold line ────────────────────────────────────────────────────────
    ax.axhline(y=bandwidth_threshold, color='#E74C3C', linestyle='--',
               linewidth=1.4, label=f'{bandwidth_threshold:.0f} dB threshold',
               zorder=3)
    ax.axhline(y=0, color='#888888', linestyle=':', linewidth=0.8, alpha=0.5,
               zorder=2)

    # ── Bandwidth shading ─────────────────────────────────────────────────────
    below = S11_db < bandwidth_threshold
    if below.any():
        ax.fill_between(freq_ghz, S11_db, bandwidth_threshold,
                        where=below, alpha=0.14, color='#E74C3C',
                        label='Operating band', zorder=2)

    # ── Detect all resonances (local minima below threshold) ─────────────────
    from scipy.signal import argrelmin
    local_min_idx = argrelmin(S11_db, order=max(1, len(S11_db) // 50))[0]
    resonances = [i for i in local_min_idx if S11_db[i] < bandwidth_threshold]
    if not resonances:
        resonances = [int(np.argmin(S11_db))]

    y_span = float(np.max(S11_db) - np.min(S11_db)) or 10
    for n_res, idx_res in enumerate(resonances):
        f_res   = float(freq_ghz[idx_res])
        s11_res = float(S11_db[idx_res])
        offset_x = (float(freq_ghz[-1]) - float(freq_ghz[0])) * 0.07
        offset_y = y_span * 0.18 * (1 + 0.4 * n_res)

        ax.annotate(
            f'f_r = {f_res:.3f} GHz\nS₁₁ = {s11_res:.1f} dB',
            xy=(f_res, s11_res),
            xytext=(f_res + offset_x, s11_res + offset_y),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.4),
            fontsize=9.5, color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#BDC3C7', alpha=0.92),
            zorder=6,
        )

        # Bandwidth annotation for this resonance
        band_mask = below.copy()
        # Find contiguous band around this resonance
        lo, hi = idx_res, idx_res
        while lo > 0 and S11_db[lo - 1] < bandwidth_threshold:
            lo -= 1
        while hi < len(S11_db) - 1 and S11_db[hi + 1] < bandwidth_threshold:
            hi += 1
        if hi > lo:
            f_lo  = float(freq_ghz[lo])
            f_hi  = float(freq_ghz[hi])
            bw    = f_hi - f_lo
            fbw   = 200.0 * bw / (f_lo + f_hi + 1e-30)
            y_arr = bandwidth_threshold + 1.5
            ax.annotate('', xy=(f_hi, y_arr), xytext=(f_lo, y_arr),
                        arrowprops=dict(arrowstyle='<->', color='#C0392B',
                                        lw=1.6), zorder=5)
            ax.text((f_lo + f_hi) / 2, y_arr + 1.2,
                    f'BW = {bw * 1000:.0f} MHz ({fbw:.1f}%)',
                    ha='center', fontsize=9.5, color='#C0392B',
                    fontweight='bold', zorder=6)

    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('|S₁₁| (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_xlim([float(freq_ghz[0]), float(freq_ghz[-1])])
    ax.set_ylim([min(float(S11_db.min()) - 5, -40), 3])
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(framealpha=0.92, fontsize=10, loc='upper right')
    fig.tight_layout()
    return fig


def plot_vswr_vs_freq(freq: np.ndarray, vswr: np.ndarray,
                      threshold: float = 2.0) -> plt.Figure:
    """Research-grade VSWR vs frequency plot.

    Upgrades
    --------
    * Green fill for matched band (VSWR < threshold).
    * Secondary axis showing equivalent return loss.
    * Minimum VSWR annotated.

    Parameters
    ----------
    freq : ndarray [Hz]
    vswr : ndarray
    threshold : float
        VSWR threshold (default 2.0 → −9.54 dB return loss).

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.2))

    freq_ghz = freq / 1e9

    ax1.plot(freq_ghz, vswr, color=PALETTE[1], linewidth=2.2, label='VSWR',
             zorder=4)
    ax1.axhline(y=threshold, color='#E74C3C', linestyle='--', linewidth=1.4,
                label=f'VSWR = {threshold:.1f}', zorder=3)

    # Matched-band fill
    matched = vswr < threshold
    ax1.fill_between(freq_ghz, 1, vswr,
                     where=matched, alpha=0.15, color='#27AE60',
                     label=f'Matched band (VSWR < {threshold:.1f})', zorder=2)

    # Secondary axis: equivalent |S₁₁| in dB
    ax2 = ax1.twinx()
    gamma = (vswr - 1) / (vswr + 1)
    s11_db_equiv = 20.0 * np.log10(np.clip(gamma, 1e-6, 1.0))
    ax2.plot(freq_ghz, s11_db_equiv, color=PALETTE[2], linewidth=1.4,
             linestyle=':', alpha=0.7, label='|S₁₁| (dB)')
    ax2.set_ylabel('Equivalent |S₁₁| (dB)', fontsize=11, color=PALETTE[2])
    ax2.tick_params(axis='y', labelcolor=PALETTE[2], labelsize=9)
    ax2.set_ylim([float(np.min(s11_db_equiv)) - 3, 1])
    ax2.axhline(-10, color=PALETTE[2], linewidth=0.7, linestyle=':', alpha=0.4)

    # Annotate VSWR minimum
    idx_min  = int(np.argmin(vswr))
    f_vmin   = float(freq_ghz[idx_min])
    vswr_min = float(vswr[idx_min])
    ax1.annotate(
        f'VSWR_min = {vswr_min:.2f}\n@ {f_vmin:.3f} GHz',
        xy=(f_vmin, vswr_min),
        xytext=(f_vmin + (float(freq_ghz[-1]) - float(freq_ghz[0])) * 0.06,
                vswr_min + (threshold - 1.0) * 0.4),
        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.3),
        fontsize=9.5, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                  ec='#BDC3C7', alpha=0.92),
        zorder=6,
    )

    ax1.set_xlabel('Frequency (GHz)', fontsize=12)
    ax1.set_ylabel('VSWR', fontsize=12)
    ax1.set_title('VSWR vs Frequency', fontsize=14, fontweight='bold',
                  color='#2C3E50')
    ax1.set_ylim([1.0, min(float(np.max(vswr)) + 1.0, 20)])
    ax1.grid(True, alpha=0.35, linestyle='--')

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, framealpha=0.92, fontsize=10,
               loc='upper right')
    fig.tight_layout()
    return fig


def plot_impedance_vs_freq(freq: np.ndarray, Z: np.ndarray,
                           Z0: float = 50.0,
                           title: str = 'Input Impedance vs Frequency') -> plt.Figure:
    """Research-grade input impedance (R + jX) vs frequency.

    Upgrades
    --------
    * Both resonance (min |X|) and anti-resonance (max |X|) detected and
      annotated.
    * Shading between R curve and Z₀ reference.
    * Improved dual-axis layout.

    Parameters
    ----------
    freq : ndarray [Hz]
    Z : ndarray of complex [Ω]
    Z0 : float
        Reference impedance [Ω]. Default 50 Ω.
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    freq_ghz = freq / 1e9
    R = Z.real
    X = Z.imag

    # Resistance & reactance
    lR, = ax1.plot(freq_ghz, R, color=PALETTE[0], linewidth=2.2, label='R (Ω)')
    lX, = ax1.plot(freq_ghz, X, color=PALETTE[1], linewidth=2.2,
                   linestyle='--', label='X (Ω)')
    ax1.axhline(0,  color='#BDC3C7', linewidth=0.8, linestyle='--')
    ax1.axhline(Z0, color='#27AE60', linewidth=1.0, linestyle=':',
                alpha=0.65, label=f'Z₀ = {Z0:.0f} Ω')

    # Shade region between R and Z0
    ax1.fill_between(freq_ghz, R, Z0,
                     where=(R < Z0), alpha=0.06, color=PALETTE[0],
                     interpolate=True)
    ax1.fill_between(freq_ghz, R, Z0,
                     where=(R > Z0), alpha=0.06, color=PALETTE[1],
                     interpolate=True)

    # VSWR on secondary axis
    gamma  = (Z - Z0) / (Z + Z0)
    mag    = np.clip(np.abs(gamma), 0.0, 0.9999)
    vswr   = (1.0 + mag) / (1.0 - mag)
    lV, = ax2.plot(freq_ghz, vswr, color=PALETTE[2], linewidth=1.5,
                   linestyle=':', alpha=0.75, label='VSWR')
    ax2.axhline(2.0, color=PALETTE[2], linewidth=0.7, linestyle=':',
                alpha=0.45)
    ax2.set_ylabel('VSWR', fontsize=12, color=PALETTE[2])
    ax2.tick_params(axis='y', labelcolor=PALETTE[2])
    ax2.set_ylim([1.0, min(float(np.max(vswr)) * 1.15, 20)])

    # Resonance: minimum |X|
    idx_res = int(np.argmin(np.abs(X)))
    f_res   = float(freq_ghz[idx_res])
    ax1.axvline(f_res, color='#95A5A6', linewidth=1.0, linestyle=':',
                alpha=0.7)
    ax1.annotate(
        f'f_r = {f_res:.3f} GHz\nR = {R[idx_res]:.1f} Ω',
        xy=(f_res, float(R[idx_res])),
        xytext=(f_res + (float(freq_ghz[-1]) - float(freq_ghz[0])) * 0.05,
                float(np.max(R)) * 0.75),
        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2),
        fontsize=9.5, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.22', fc='white',
                  ec='#BDC3C7', alpha=0.92),
    )

    # Anti-resonance: maximum |X|
    idx_ares = int(np.argmax(np.abs(X)))
    f_ares   = float(freq_ghz[idx_ares])
    ax1.axvline(f_ares, color='#E67E22', linewidth=0.9, linestyle=':', alpha=0.55)
    ax1.annotate(
        f'f_ar = {f_ares:.3f} GHz',
        xy=(f_ares, float(X[idx_ares])),
        xytext=(f_ares - (float(freq_ghz[-1]) - float(freq_ghz[0])) * 0.12,
                float(np.max(R)) * 0.55),
        arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1.1),
        fontsize=9, color='#E67E22',
        bbox=dict(boxstyle='round,pad=0.20', fc='white',
                  ec='#E67E22', alpha=0.88),
    )

    ax1.set_xlabel('Frequency (GHz)', fontsize=12)
    ax1.set_ylabel('Impedance (Ω)', fontsize=12)
    ax1.set_xlim([float(freq_ghz[0]), float(freq_ghz[-1])])
    ax1.grid(True, alpha=0.35, linestyle='--')
    ax1.set_title(title, fontsize=14, fontweight='bold', color='#2C3E50')

    lines  = [lR, lX, lV]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, framealpha=0.92, fontsize=10, loc='upper left')

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib static gain heatmap  (for dashboards and 600 DPI export)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_heatmap_mpl(radiation_pattern,
                          ax=None,
                          dyn_range: float = 40,
                          contour_levels: tuple = (-3, -10, -20)) -> plt.Figure:
    """Matplotlib static gain heatmap (θ vs φ, colour = dBi).

    Suitable for 600 DPI export and dashboard embedding.  A Plotly
    interactive version is available as :func:`plot_gain_heatmap`.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    ax : matplotlib Axes or None
    dyn_range : float
        Colour range below peak [dB].
    contour_levels : tuple of float
        dB offsets from peak at which to draw iso-gain lines.

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    D_dbi     = radiation_pattern.to_dbi()
    theta_deg = np.rad2deg(radiation_pattern.theta)
    phi_deg   = np.rad2deg(radiation_pattern.phi)
    peak_dbi  = float(np.max(D_dbi))

    im = ax.pcolormesh(phi_deg, theta_deg, D_dbi,
                       cmap='turbo',
                       vmin=peak_dbi - dyn_range,
                       vmax=peak_dbi,
                       shading='gouraud', rasterized=True)

    # ── Contour overlays ──────────────────────────────────────────────────────
    _cont_colors  = {-3: 'white', -10: 'lightgrey', -20: 'grey'}
    _cont_dashes  = {-3: 'solid', -10: 'dashed',    -20: 'dotted'}
    _cont_widths  = {-3: 1.6,    -10: 1.1,           -20: 0.9}
    for offset in contour_levels:
        level = peak_dbi + offset
        try:
            cs = ax.contour(phi_deg, theta_deg, D_dbi,
                            levels=[level],
                            colors=[_cont_colors.get(offset, 'white')],
                            linewidths=[_cont_widths.get(offset, 1.0)],
                            linestyles=[_cont_dashes.get(offset, 'solid')])
            ax.clabel(cs, inline=True, fontsize=7.5,
                      fmt={level: f'{offset:+.0f} dB'})
        except Exception:
            pass

    # ── Peak marker ───────────────────────────────────────────────────────────
    pi, pj = np.unravel_index(np.argmax(D_dbi), D_dbi.shape)
    ax.plot(float(phi_deg[pj]), float(theta_deg[pi]),
            'w+', markersize=14, markeredgewidth=2.5, zorder=5)
    ax.text(float(phi_deg[pj]) + 3, float(theta_deg[pi]) - 2,
            f'{peak_dbi:.1f} dBi',
            color='white', fontsize=8.5, fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', fc='#2C3E50',
                      ec='white', alpha=0.82))

    # ── Colorbar ─────────────────────────────────────────────────────────────
    # Use make_axes_locatable only when standalone to avoid triggering
    # matplotlib's tight-layout-compatibility probe on figures that also
    # contain polar axes (e.g. dashboards).
    if standalone:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div  = make_axes_locatable(ax)
        cax  = div.append_axes('right', size='3%', pad=0.09)
        cbar = fig.colorbar(im, cax=cax)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label('Gain (dBi)', fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel('phi (degrees)', fontsize=11)
    ax.set_ylabel('theta (degrees)', fontsize=11)
    ax.invert_yaxis()   # theta = 0 deg at top (zenith)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(0, 181, 15))
    ax.tick_params(labelsize=9)

    freq_ghz = getattr(radiation_pattern, 'freq', None)
    title_str = (f'Gain Pattern (dBi) @ {freq_ghz / 1e9:.3f} GHz'
                 if freq_ghz else 'Gain Pattern (dBi)')
    ax.set_title(title_str, fontsize=12, fontweight='bold', color='#2C3E50')

    if standalone:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Frequency–Angle Gain Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(freq_ghz: np.ndarray,
                   theta_deg: np.ndarray,
                   patterns_db: np.ndarray,
                   phi_cut_deg: float = 0.0,
                   dyn_range: float = 40,
                   title: str = 'Frequency–Angle Gain Waterfall') -> plt.Figure:
    """Frequency × angle × gain waterfall (2-D colour map).

    Useful for understanding how the radiation pattern varies with
    frequency — shows beam narrowing/broadening, frequency-dependent
    SLL, and pattern scan.

    Parameters
    ----------
    freq_ghz : ndarray, shape (n_freq,)
        Frequency axis [GHz].
    theta_deg : ndarray, shape (n_theta,)
        Angle axis [degrees].
    patterns_db : ndarray, shape (n_freq, n_theta)
        Normalised gain at each (frequency, angle) point [dB].
        Each row should be normalised to its own peak (0 dB) or to the
        global peak — both are informative.
    phi_cut_deg : float
        Azimuthal cut angle [degrees] (for title annotation).
    dyn_range : float
        Colour range below the maximum [dB].
    title : str

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from pylobe.visualization.heatmap import plot_waterfall
    >>> theta = np.linspace(-90, 90, 181)
    >>> freqs = np.linspace(2.0, 3.0, 11)
    >>> pats  = np.random.uniform(-40, 0, (11, 181))
    >>> fig   = plot_waterfall(freqs, theta, pats)
    """
    _setup_style()
    freq_ghz   = np.asarray(freq_ghz,   dtype=float)
    theta_deg  = np.asarray(theta_deg,  dtype=float)
    patterns_db = np.asarray(patterns_db, dtype=float)

    peak_db = float(np.max(patterns_db))
    vmin    = peak_db - dyn_range

    fig, (ax_main, ax_side) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.08},
    )

    # ── Waterfall heatmap ─────────────────────────────────────────────────────
    im = ax_main.pcolormesh(theta_deg, freq_ghz, patterns_db,
                             cmap='turbo',
                             vmin=vmin, vmax=peak_db,
                             shading='gouraud', rasterized=True)

    # Contour overlays at -3, -10 dB
    try:
        cs3 = ax_main.contour(theta_deg, freq_ghz, patterns_db,
                               levels=[peak_db - 3],
                               colors=['white'], linewidths=[1.5])
        ax_main.clabel(cs3, inline=True, fontsize=7.5, fmt='−3 dB')
    except Exception:
        pass
    try:
        cs10 = ax_main.contour(theta_deg, freq_ghz, patterns_db,
                                levels=[peak_db - 10],
                                colors=['lightgrey'], linewidths=[1.0],
                                linestyles=['dashed'])
        ax_main.clabel(cs10, inline=True, fontsize=7, fmt='−10 dB')
    except Exception:
        pass

    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div  = make_axes_locatable(ax_main)
    cax  = div.append_axes('right', size='2.5%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Norm. Gain (dB)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax_main.set_xlabel(f'θ (degrees)   [φ = {phi_cut_deg:.0f}°]', fontsize=11)
    ax_main.set_ylabel('Frequency (GHz)', fontsize=11)
    ax_main.set_title(f'{title}', fontsize=12, fontweight='bold', color='#2C3E50')
    ax_main.tick_params(labelsize=9)

    # ── Side panel: peak gain vs frequency ───────────────────────────────────
    peak_per_freq = np.max(patterns_db, axis=1)
    peak_angle    = theta_deg[np.argmax(patterns_db, axis=1)]

    ax_side.plot(peak_per_freq, freq_ghz,
                 color=PALETTE[0], linewidth=2.0)
    ax_side.fill_betweenx(freq_ghz, peak_per_freq, vmin,
                           alpha=0.15, color=PALETTE[0])
    ax_side.set_xlabel('Peak Gain (dB)', fontsize=10)
    ax_side.set_xlim([vmin, peak_db + 1])
    ax_side.set_ylim([float(freq_ghz[0]), float(freq_ghz[-1])])
    ax_side.yaxis.set_ticklabels([])
    ax_side.grid(True, alpha=0.35, linestyle='--')
    ax_side.set_title('Peak vs Freq', fontsize=10, fontweight='bold',
                       color='#2C3E50')
    ax_side.tick_params(labelsize=9)

    # Twin axis: peak angle
    ax_side2 = ax_side.twiny()
    ax_side2.plot(peak_angle, freq_ghz,
                  color=PALETTE[1], linewidth=1.8, linestyle='--',
                  label='Peak angle')
    ax_side2.set_xlabel('Peak θ (°)', fontsize=9, color=PALETTE[1])
    ax_side2.tick_params(axis='x', labelcolor=PALETTE[1], labelsize=8)

    fig.suptitle(f'φ = {phi_cut_deg:.0f}° cut | '
                 f'{freq_ghz[0]:.3f}–{freq_ghz[-1]:.3f} GHz',
                 fontsize=11, color='#2C3E50', y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Return-loss vs frequency (separate from S11)
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_loss_vs_freq(freq: np.ndarray, S11_db: np.ndarray,
                              title: str = 'Return Loss vs Frequency') -> plt.Figure:
    """Return loss (positive dB convention) vs frequency.

    Return loss RL = −S₁₁ [dB] (positive; higher = better matched).

    Parameters
    ----------
    freq : ndarray [Hz]
    S11_db : ndarray [dB]  (negative values)
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    freq_ghz = freq / 1e9
    rl_db = -np.asarray(S11_db, dtype=float)   # flip sign → positive

    ax.plot(freq_ghz, rl_db, color=PALETTE[0], linewidth=2.2, label='Return Loss')
    ax.axhline(10, color='#27AE60', linestyle='--', linewidth=1.4,
               label='10 dB threshold', alpha=0.85)
    ax.fill_between(freq_ghz, rl_db, 10,
                    where=(rl_db >= 10), alpha=0.13, color='#27AE60',
                    label='Matched band (RL > 10 dB)')

    # Annotate maximum return loss
    idx_max = int(np.argmax(rl_db))
    ax.annotate(
        f'RL_max = {rl_db[idx_max]:.1f} dB\n@ {freq_ghz[idx_max]:.3f} GHz',
        xy=(float(freq_ghz[idx_max]), float(rl_db[idx_max])),
        xytext=(float(freq_ghz[idx_max]) + (float(freq_ghz[-1]) -
                float(freq_ghz[0])) * 0.07,
                float(rl_db[idx_max]) - 4),
        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.3),
        fontsize=9.5, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                  ec='#BDC3C7', alpha=0.92),
    )

    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Return Loss (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_xlim([float(freq_ghz[0]), float(freq_ghz[-1])])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(framealpha=0.92, fontsize=10)
    fig.tight_layout()
    return fig
