"""Research-grade polar radiation pattern plots (Matplotlib)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.patches import FancyArrowPatch
from pylobe.analysis.metrics import beamwidth_hpbw, side_lobe_level
from pylobe.visualization.style import set_style as _set_global_style

# ── Colorblind-friendly research palette (Wong 2011) ─────────────────────────
PALETTE = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermilion
    '#009E73',  # Bluish green
    '#CC79A7',  # Reddish purple
    '#56B4E9',  # Sky blue
    '#E69F00',  # Orange
    '#F0E442',  # Yellow
    '#000000',  # Black
]

_HPBW_COLOR = '#C0392B'
_SLL_COLOR  = '#E67E22'
_NULL_COLOR = '#27AE60'


def _setup_style():
    if plt.rcParams.get('figure.dpi', 100) == 100:
        _set_global_style('default')


def _find_nulls(pattern_db: np.ndarray, angles_deg: np.ndarray,
                threshold_below_peak: float = 15.0):
    """Return angles of local minima that are at least threshold_below_peak dB below peak."""
    peak = np.max(pattern_db)
    nulls = []
    n = len(pattern_db)
    for i in range(1, n - 1):
        if (pattern_db[i] <= pattern_db[i - 1] and
                pattern_db[i] <= pattern_db[i + 1] and
                pattern_db[i] < peak - threshold_below_peak):
            nulls.append(angles_deg[i])
    return np.array(nulls)


def _find_sidelobe_peaks(pattern_db: np.ndarray, peak_idx: int):
    """Indices of sidelobe peaks, sorted by descending gain."""
    peaks = []
    n = len(pattern_db)
    for i in range(1, n - 1):
        if i == peak_idx:
            continue
        if (pattern_db[i] > pattern_db[i - 1] and
                pattern_db[i] > pattern_db[i + 1] and
                pattern_db[i] > np.max(pattern_db) - 45):
            peaks.append(i)
    return sorted(peaks, key=lambda i: -pattern_db[i])


def _configure_polar_ax(ax, peak_db: float, r_min: float, dyn_range: float):
    """Apply consistent research-grade polar axes styling."""
    ax.set_rmin(r_min)
    ax.set_rmax(peak_db + 3)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    tick_step = 10 if dyn_range >= 30 else 5
    r_ticks = np.arange(r_min, peak_db + 1, tick_step)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels(
        [f'{int(v)} dB' for v in r_ticks],
        fontsize=7.5, color='#444444'
    )
    ax.set_rlabel_position(42)

    ax.yaxis.grid(True, linestyle='--', alpha=0.35, color='#888888', linewidth=0.6)
    ax.xaxis.grid(True, linestyle=':',  alpha=0.30, color='#888888', linewidth=0.5)
    ax.set_thetagrids(np.arange(0, 360, 30), fontsize=9, color='#2C3E50')
    ax.set_facecolor('#F8F9FA')
    ax.spines['polar'].set_linewidth(1.2)
    ax.spines['polar'].set_color('#555555')


def plot_polar(pattern_db: np.ndarray, angles_deg: np.ndarray,
               label: str = '', ax=None,
               dyn_range: float = 40,
               color: str = None,
               annotate: bool = True) -> plt.Figure:
    """Research-grade polar radiation pattern on dB scale.

    Upgrades over legacy version
    ----------------------------
    * Colorblind-friendly palette.
    * Legend placed *below* the polar disc — never overlaps the pattern.
    * Main-lobe sector highlighted with stronger fill (> −10 dB region).
    * HPBW annotation: dash-dot −3 dB arc + double-headed arrow + labelled box.
    * SLL annotation: dashed ring at the first sidelobe level.
    * Null markers: small green triangles at each null.
    * First-sidelobe peak marked with an open circle.
    * ``constrained_layout``-aware figure sizing.

    Parameters
    ----------
    pattern_db : ndarray
        Radiation pattern [dB], normalised so peak = 0 dB (or absolute dBi).
    angles_deg : ndarray
        Corresponding angles [degrees].
    label : str
        Legend label / frequency string.
    ax : matplotlib polar Axes or None
        Existing polar axes to draw on. Created if *None*.
    dyn_range : float
        Radial depth [dB] displayed. Default 40 dB.
    color : str or None
        Trace colour. Falls back to palette[0].
    annotate : bool
        Draw HPBW, SLL, null, and sidelobe markers.

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(7.5, 8.2))
        ax  = fig.add_axes([0.08, 0.10, 0.84, 0.84], projection='polar')
    else:
        fig = ax.get_figure()

    peak_db  = float(np.max(pattern_db))
    r_min    = peak_db - dyn_range
    angles_r = np.deg2rad(angles_deg)
    r_data   = np.clip(pattern_db, r_min, peak_db)
    color    = color or PALETTE[0]
    peak_idx = int(np.argmax(pattern_db))

    # ── Pattern line ──────────────────────────────────────────────────────────
    ax.plot(angles_r, r_data, color=color, label=label,
            linewidth=2.0, zorder=4, solid_capstyle='round')

    # ── Two-layer fill: full-range light + main-lobe accent ──────────────────
    ax.fill(angles_r, r_data, alpha=0.07, color=color, zorder=2)
    # Highlight region above −10 dB (main-lobe body)
    ml_r = np.where(r_data >= peak_db - 10, r_data, r_min)
    ax.fill(angles_r, ml_r, alpha=0.17, color=color, zorder=3)

    # ── Axes ─────────────────────────────────────────────────────────────────
    _configure_polar_ax(ax, peak_db, r_min, dyn_range)

    # ── Annotations ──────────────────────────────────────────────────────────
    if annotate:
        linear = 10.0 ** ((pattern_db - peak_db) / 10.0)
        hpbw   = beamwidth_hpbw(linear, angles_deg)

        # HPBW
        if not np.isnan(hpbw) and hpbw > 0:
            peak_angle = float(angles_deg[peak_idx])
            half_lo    = np.deg2rad(peak_angle - hpbw / 2)
            half_hi    = np.deg2rad(peak_angle + hpbw / 2)
            hp_level   = peak_db - 3.0

            # Dash-dot arc at −3 dB
            arc_a = np.linspace(half_lo, half_hi, 90)
            ax.plot(arc_a, np.full(90, hp_level),
                    color=_HPBW_COLOR, linewidth=1.8,
                    linestyle='-.', alpha=0.9, zorder=5)

            # Double-headed arrow spanning the beam
            ax.annotate('', xy=(half_hi, hp_level),
                        xytext=(half_lo, hp_level),
                        arrowprops=dict(arrowstyle='<->',
                                        color=_HPBW_COLOR, lw=1.7,
                                        mutation_scale=14),
                        zorder=6)

            # Labelled text box
            ax.text(np.deg2rad(peak_angle), hp_level - 3.0,
                    f'HPBW = {hpbw:.1f}°',
                    ha='center', va='top',
                    fontsize=8.5, color=_HPBW_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.28', fc='white',
                              ec=_HPBW_COLOR, alpha=0.93, linewidth=0.9),
                    zorder=7)

        # SLL dashed ring
        sll = side_lobe_level(pattern_db, angles_deg)
        if not np.isinf(sll) and sll < -5:
            sll_level = peak_db + sll
            ring_a    = np.linspace(0, 2 * np.pi, 500)
            ax.plot(ring_a, np.full(500, sll_level),
                    color=_SLL_COLOR, linestyle='--',
                    linewidth=1.3, alpha=0.75, zorder=5)
            ax.text(np.deg2rad(68), sll_level + 1.2,
                    f'SLL = {sll:.1f} dB',
                    fontsize=8, color=_SLL_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white',
                              ec=_SLL_COLOR, alpha=0.92, linewidth=0.8),
                    zorder=7)

        # Null markers (green downward triangles at the noise floor)
        nulls = _find_nulls(pattern_db, angles_deg)
        for null_ang in nulls[:12]:
            ax.plot(np.deg2rad(null_ang), r_min + 1.0,
                    'v', color=_NULL_COLOR, markersize=5.5,
                    alpha=0.85, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.5)

        # First sidelobe peak (open circle)
        sl_peaks = _find_sidelobe_peaks(pattern_db, peak_idx)
        if sl_peaks:
            fsl = sl_peaks[0]
            ax.plot(angles_r[fsl], r_data[fsl],
                    'o', color=_SLL_COLOR, markersize=6,
                    markerfacecolor='white', markeredgewidth=1.5,
                    zorder=6)

    # ── Title ─────────────────────────────────────────────────────────────────
    freq_label = f' — {label}' if label else ''
    ax.set_title(f'Radiation Pattern{freq_label}',
                 pad=20, fontsize=12, fontweight='bold', color='#2C3E50')

    # ── Legend: placed below the polar disc, never overlapping ───────────────
    if label and standalone:
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, lbls,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.01),
                       ncol=min(len(lbls), 4),
                       framealpha=0.93, fontsize=9,
                       edgecolor='#cccccc')

    return fig


def plot_polar_compare(theta_list, patterns: list,
                       labels: list = None,
                       title: str = 'Radiation Pattern Comparison',
                       dyn_range: float = 40) -> plt.Figure:
    """Overlay multiple radiation patterns on the same polar axes.

    Upgrades: research-grade color palette, legend placed *outside* the polar
    disc (below), consistent axes styling via ``_configure_polar_ax``.

    Parameters
    ----------
    theta_list : list of ndarray or single ndarray
        Angles [radians] for each pattern, or a single shared array.
    patterns : list of ndarray
        List of 1-D patterns (linear, normalised to 1).
    labels : list of str or None
        Legend labels.
    title : str
    dyn_range : float

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig = plt.figure(figsize=(8.0, 8.8))
    ax  = fig.add_axes([0.08, 0.10, 0.84, 0.84], projection='polar')

    if labels is None:
        labels = [f'Pattern {i + 1}' for i in range(len(patterns))]

    if not isinstance(theta_list, (list, tuple)):
        theta_list = [theta_list] * len(patterns)

    dbs = []
    for pat in patterns:
        pat   = np.asarray(pat, dtype=float)
        p_max = float(np.max(pat)) if np.max(pat) > 0 else 1.0
        dbs.append(10.0 * np.log10(np.clip(pat / p_max, 1e-10, None)))

    peak_all = float(max(np.max(d) for d in dbs))
    r_min    = peak_all - dyn_range

    for i, (theta, pat_db, lbl) in enumerate(zip(theta_list, dbs, labels)):
        theta  = np.asarray(theta, dtype=float)
        r_data = np.clip(pat_db, r_min, peak_all)
        color  = PALETTE[i % len(PALETTE)]
        ax.plot(theta, r_data, color=color, label=lbl,
                linewidth=2.0, zorder=3 + i)
        ax.fill(theta, r_data, alpha=0.06, color=color, zorder=2)

    _configure_polar_ax(ax, peak_all, r_min, dyn_range)

    ax.set_title(title, pad=20, fontsize=12, fontweight='bold', color='#2C3E50')

    # Legend below polar disc — never touches the pattern
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=min(len(lbls), 4),
                   framealpha=0.93, fontsize=9,
                   edgecolor='#cccccc')

    return fig


def plot_e_h_plane(radiation_pattern, dyn_range: float = 40) -> plt.Figure:
    """Side-by-side E-plane and H-plane polar plots.

    Upgrades: shared figure legend below both panels, accent fill for
    main-lobe region, consistent research-grade styling.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    dyn_range : float

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    fig = plt.figure(figsize=(14, 8.5))
    ax_e = fig.add_axes([0.04, 0.10, 0.44, 0.82], projection='polar')
    ax_h = fig.add_axes([0.52, 0.10, 0.44, 0.82], projection='polar')

    theta_deg = np.rad2deg(radiation_pattern.theta)

    def _prepare_360(cut_lin):
        p_max = float(np.max(cut_lin)) if np.max(cut_lin) > 0 else 1.0
        db    = 10.0 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
        a360  = np.concatenate([theta_deg, theta_deg + 180.0])
        d360  = np.concatenate([db, db[::-1]])
        return a360, d360

    e_cut_lin         = radiation_pattern.phi_cut(0.0)
    angles_360, e_360 = _prepare_360(e_cut_lin)
    plot_polar(e_360, angles_360, label='E-plane (φ=0°)',
               ax=ax_e, dyn_range=dyn_range, color=PALETTE[0])
    ax_e.set_title('E-Plane Pattern', pad=20, fontsize=12, fontweight='bold',
                   color='#2C3E50')

    h_cut_lin         = radiation_pattern.phi_cut(90.0)
    angles_360, h_360 = _prepare_360(h_cut_lin)
    plot_polar(h_360, angles_360, label='H-plane (φ=90°)',
               ax=ax_h, dyn_range=dyn_range, color=PALETTE[1])
    ax_h.set_title('H-Plane Pattern', pad=20, fontsize=12, fontweight='bold',
                   color='#2C3E50')

    fig.suptitle('E-Plane & H-Plane Radiation Patterns',
                 fontsize=14, fontweight='bold', color='#2C3E50', y=0.99)

    # Shared legend below both panels
    handles, lbls = [], []
    for ax in [ax_e, ax_h]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h[:1])  # first line handle only
        lbls.extend(l[:1])
    if handles:
        fig.legend(handles, lbls,
                   loc='lower center', bbox_to_anchor=(0.5, 0.01),
                   ncol=2, framealpha=0.93, fontsize=10, edgecolor='#cccccc')

    return fig


def plot_phase_pattern(radiation_pattern,
                       phi_deg: float = 0.0,
                       component: str = 'theta',
                       ax=None) -> plt.Figure:
    """Phase of the radiation pattern at a given φ cut.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    phi_deg : float
        Azimuthal angle [degrees].
    component : str
        'theta' or 'phi'.
    ax : matplotlib Axes or None

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
    else:
        fig = ax.get_figure()

    phase_2d = radiation_pattern.phase_pattern(component=component)
    phi_rad  = np.deg2rad(phi_deg) % (2.0 * np.pi)
    idx      = int(np.argmin(np.abs(radiation_pattern.phi - phi_rad)))
    phase_cut = phase_2d[:, idx]
    theta_deg = np.rad2deg(radiation_pattern.theta)

    ax.plot(theta_deg, phase_cut, color=PALETTE[2], linewidth=2.0)

    # Shade ±90° regions
    ax.axhspan(90, 180, alpha=0.04, color=PALETTE[0])
    ax.axhspan(-180, -90, alpha=0.04, color=PALETTE[1])

    ax.set_xlabel('θ (degrees)', fontsize=12)
    ax.set_ylabel('Phase (degrees)', fontsize=12)
    ax.set_xlim([float(theta_deg[0]), float(theta_deg[-1])])
    ax.set_ylim([-195, 195])
    ax.set_yticks(range(-180, 181, 45))
    ax.axhline(0, color='#bdc3c7', linewidth=0.8, linestyle='--')
    ax.grid(True, alpha=0.35, linestyle='--')

    freq_str = f'{radiation_pattern.freq / 1e9:.3f} GHz'
    ax.set_title(
        f'Phase Pattern — E_{component} | φ = {phi_deg:.0f}° | {freq_str}',
        fontsize=12, fontweight='bold', color='#2C3E50',
    )

    if standalone:
        fig.tight_layout()
    return fig
