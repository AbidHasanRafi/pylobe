"""Research-grade Cartesian (rectangular) radiation pattern plots.

Upgrades over legacy version
-----------------------------
* Gradient fill distinguishing main-lobe body (above −10 dB) from
  sidelobe region.
* −3 dB and −10 dB reference lines labelled inline.
* Peak gain value annotated on the plot.
* First sidelobe peak marked with an open-circle marker.
* Null positions shown as small downward triangles on the floor.
* Colourblind-safe palette consistent with ``polar.py``.
"""
import numpy as np
import matplotlib.pyplot as plt
from pylobe.visualization.polar import (
    _setup_style, PALETTE,
    _find_nulls, _find_sidelobe_peaks,
)

_NULL_COLOR = '#27AE60'
_SLL_COLOR  = '#E67E22'
_HPBW_COLOR = '#C0392B'


def plot_pattern_cartesian(pattern_db: np.ndarray, angles_deg: np.ndarray,
                           label: str = '', ax=None,
                           dyn_range: float = 40,
                           color: str = None,
                           freq: float = None,
                           annotate: bool = True) -> plt.Figure:
    """Research-grade Cartesian radiation pattern (dB vs angle).

    Upgrades
    --------
    * Two-level fill: full-range light + main-lobe (> −10 dB) accent.
    * −3 dB and −10 dB reference lines with inline labels.
    * Peak gain annotated (dBi or normalised dB).
    * First sidelobe peak marked with an open circle.
    * Null positions marked with triangles on the dynamic-range floor.

    Parameters
    ----------
    pattern_db : ndarray
        Pattern [dB].
    angles_deg : ndarray
        Angles [degrees].
    label : str
    ax : matplotlib Axes or None
    dyn_range : float
    color : str or None
    freq : float or None  [Hz]
    annotate : bool

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
    else:
        fig = ax.get_figure()

    color    = color or PALETTE[0]
    peak     = float(np.max(pattern_db))
    floor    = peak - dyn_range
    peak_idx = int(np.argmax(pattern_db))

    # ── Main line ──────────────────────────────────────────────────────────────
    ax.plot(angles_deg, pattern_db, color=color, linewidth=2.0,
            label=label if label else '_nolegend_', zorder=4)

    # ── Two-level fill ────────────────────────────────────────────────────────
    ax.fill_between(angles_deg, pattern_db, floor,
                    alpha=0.08, color=color, zorder=2)
    # Accent fill for main lobe (> −10 dB)
    ml_db = np.where(pattern_db >= peak - 10, pattern_db, floor)
    ax.fill_between(angles_deg, ml_db, floor,
                    alpha=0.18, color=color, zorder=3)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(y=peak - 3, color=_HPBW_COLOR, linestyle='--',
               linewidth=1.2, alpha=0.9, zorder=3)
    ax.axhline(y=peak - 10, color=_SLL_COLOR, linestyle=':',
               linewidth=1.1, alpha=0.85, zorder=3)

    # Inline reference labels on the right edge
    x_right = float(angles_deg[-1])
    ax.text(x_right, peak - 3 + 0.5, '−3 dB', va='bottom', ha='right',
            fontsize=8, color=_HPBW_COLOR, fontweight='bold')
    ax.text(x_right, peak - 10 + 0.5, '−10 dB', va='bottom', ha='right',
            fontsize=8, color=_SLL_COLOR, fontweight='bold')

    # ── Annotations ───────────────────────────────────────────────────────────
    if annotate:
        # Peak gain label
        ax.annotate(
            f'{peak:.1f} dB',
            xy=(float(angles_deg[peak_idx]), peak),
            xytext=(float(angles_deg[peak_idx]), peak + dyn_range * 0.08),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
            ha='center', fontsize=9, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec=color, alpha=0.90, linewidth=0.8),
            zorder=6,
        )

        # First sidelobe peak
        sl_peaks = _find_sidelobe_peaks(pattern_db, peak_idx)
        if sl_peaks:
            fsl = sl_peaks[0]
            ax.plot(float(angles_deg[fsl]), float(pattern_db[fsl]),
                    'o', color=_SLL_COLOR, markersize=7,
                    markerfacecolor='white', markeredgewidth=1.8,
                    zorder=5, label=f'1st SL: {pattern_db[fsl]:.1f} dB')
            ax.text(float(angles_deg[fsl]),
                    float(pattern_db[fsl]) + dyn_range * 0.04,
                    f'{pattern_db[fsl]:.1f} dB',
                    ha='center', fontsize=8, color=_SLL_COLOR,
                    fontweight='bold', zorder=6)

        # Null markers
        nulls = _find_nulls(pattern_db, angles_deg)
        ax.plot(nulls, np.full(len(nulls), floor + 0.5),
                'v', color=_NULL_COLOR, markersize=5,
                alpha=0.85, zorder=5, label='Nulls' if len(nulls) else '_')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim([float(angles_deg[0]), float(angles_deg[-1])])
    ax.set_ylim([floor - 2, peak + dyn_range * 0.12])
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalised Gain (dB)', fontsize=12)
    ax.grid(True, alpha=0.35, linestyle='--')

    freq_str = f' @ {freq / 1e9:.3f} GHz' if freq is not None else ''
    ax.set_title(f'Radiation Pattern (Cartesian){freq_str}',
                 fontsize=13, fontweight='bold', color='#2C3E50')

    if label:
        ax.legend(framealpha=0.92, fontsize=9, loc='upper right')
    fig.tight_layout()
    return fig


def plot_array_factor(array, freq: float,
                      theta_range: tuple = (0, 180)) -> plt.Figure:
    """Plot normalised array factor vs theta with enhanced annotations.

    Upgrades
    --------
    * Two-level fill for main-lobe accent.
    * −3 dB label inline.
    * Grating-lobe positions annotated (if any exist > −3 dB).

    Parameters
    ----------
    array : LinearArray
    freq : float  [Hz]
    theta_range : tuple  (lo_deg, hi_deg)

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    theta_deg = np.linspace(*theta_range, 361)
    theta_rad = np.deg2rad(theta_deg)
    af        = array.array_factor(theta_rad, freq)
    af_db     = 20.0 * np.log10(np.clip(af, 1e-5, None))
    peak      = float(np.max(af_db))
    floor_db  = -40.0

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(theta_deg, af_db, color=PALETTE[0], linewidth=2.0, zorder=4)

    # Two-level fill
    ax.fill_between(theta_deg, af_db, floor_db,
                    alpha=0.08, color=PALETTE[0], zorder=2)
    ml_af = np.where(af_db >= peak - 10, af_db, floor_db)
    ax.fill_between(theta_deg, ml_af, floor_db,
                    alpha=0.18, color=PALETTE[0], zorder=3)

    # Reference lines
    ax.axhline(peak - 3, color=_HPBW_COLOR, linestyle='--',
               linewidth=1.2, alpha=0.9, zorder=3)
    ax.text(float(theta_deg[-1]), peak - 3 + 0.6,
            '−3 dB', va='bottom', ha='right',
            fontsize=8, color=_HPBW_COLOR, fontweight='bold')

    # Detect grating lobes (peaks above −3 dB other than main lobe)
    peak_idx = int(np.argmax(af_db))
    sl_peaks = _find_sidelobe_peaks(af_db, peak_idx)
    grating  = [i for i in sl_peaks if af_db[i] > peak - 3]
    for gi in grating:
        ax.axvline(float(theta_deg[gi]), color='#E74C3C',
                   linestyle=':', linewidth=1.2, alpha=0.8)
        ax.text(float(theta_deg[gi]), peak - 5,
                f'GL\n{theta_deg[gi]:.0f}°', ha='center', fontsize=7.5,
                color='#E74C3C', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='#E74C3C', alpha=0.85))

    ax.set_xlabel('θ (degrees)', fontsize=12)
    ax.set_ylabel('|AF| (dB)', fontsize=12)
    ax.set_xlim([float(theta_deg[0]), float(theta_deg[-1])])
    ax.set_ylim([floor_db - 1, peak + 3])
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.set_title(
        f'Array Factor — N = {array.N}, d = {array.d * 1e3:.1f} mm'
        f' @ {freq / 1e9:.3f} GHz',
        fontsize=13, fontweight='bold', color='#2C3E50',
    )
    fig.tight_layout()
    return fig
