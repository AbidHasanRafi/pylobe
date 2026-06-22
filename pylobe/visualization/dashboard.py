"""Comprehensive multi-panel visualization dashboards for antenna design.

All figures save at 600 DPI (controlled by ``style.py``).

Public API
----------
plot_pattern_dashboard(radiation_pattern, ...)
    3 × 3 panel: E/H polar + E/H cartesian + gain heatmap +
    diagonal cuts + metrics table.

plot_frequency_dashboard(freqs, S11_complex, Z_complex, ...)
    3 × 3 panel: S11 + VSWR + return loss + R(Z) + X(Z) +
    |Z|/∠Z + group delay + S-parameter table.

plot_design_dashboard(design)
    Combined pattern + frequency + structure figure (returns a tuple
    of matplotlib Figures and Plotly Figures).

plot_compare_patterns(rp_list, labels, ...)
    Overlay multiple antennas' E/H patterns and gain bar chart.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylobe.analysis.metrics import (
    beamwidth_hpbw, beamwidth_fnbw, side_lobe_level, front_to_back_ratio,
)
from pylobe.visualization.polar import (
    _setup_style, _configure_polar_ax, _find_nulls, _find_sidelobe_peaks,
    PALETTE, _HPBW_COLOR, _SLL_COLOR, _NULL_COLOR,
)
from pylobe.constants import PI

# ── Colour constants ──────────────────────────────────────────────────────────
_TITLE_BG  = '#2C3E50'
_ROW_EVEN  = '#EBF5FB'
_ROW_ODD   = '#FDFEFE'
_HDR_BG    = '#2C3E50'
_HDR_FG    = 'white'


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_db(linear: np.ndarray, ref: float = None) -> np.ndarray:
    ref = ref or float(np.max(linear)) or 1.0
    return 10.0 * np.log10(np.clip(linear / ref, 1e-10, None))


def _plot_polar_on_ax(ax, pattern_db: np.ndarray, angles_deg: np.ndarray,
                      dyn_range: float = 40, color: str = None,
                      label: str = '', annotate: bool = True):
    """Draw a polar pattern on an *existing* polar axes (no new figure)."""
    color    = color or PALETTE[0]
    peak_db  = float(np.max(pattern_db))
    r_min    = peak_db - dyn_range
    angles_r = np.deg2rad(angles_deg)
    r_data   = np.clip(pattern_db, r_min, peak_db)
    peak_idx = int(np.argmax(pattern_db))

    ax.plot(angles_r, r_data, color=color, linewidth=1.8,
            label=label, zorder=4, solid_capstyle='round')
    ax.fill(angles_r, r_data, alpha=0.07, color=color, zorder=2)
    ml_r = np.where(r_data >= peak_db - 10, r_data, r_min)
    ax.fill(angles_r, ml_r, alpha=0.17, color=color, zorder=3)
    _configure_polar_ax(ax, peak_db, r_min, dyn_range)

    if annotate:
        linear = 10.0 ** ((pattern_db - peak_db) / 10.0)
        hpbw   = beamwidth_hpbw(linear, angles_deg)
        if not np.isnan(hpbw) and hpbw > 0:
            peak_angle = float(angles_deg[peak_idx])
            hp_level   = peak_db - 3.0
            half_lo    = np.deg2rad(peak_angle - hpbw / 2)
            half_hi    = np.deg2rad(peak_angle + hpbw / 2)
            arc_a      = np.linspace(half_lo, half_hi, 80)
            ax.plot(arc_a, np.full(80, hp_level),
                    color=_HPBW_COLOR, linewidth=1.6,
                    linestyle='-.', alpha=0.9, zorder=5)
            ax.annotate('', xy=(half_hi, hp_level),
                        xytext=(half_lo, hp_level),
                        arrowprops=dict(arrowstyle='<->',
                                        color=_HPBW_COLOR, lw=1.5,
                                        mutation_scale=12), zorder=6)
            ax.text(np.deg2rad(peak_angle), hp_level - 2.8,
                    f'HPBW={hpbw:.1f}°', ha='center', va='top',
                    fontsize=7.5, color=_HPBW_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white',
                              ec=_HPBW_COLOR, alpha=0.92, linewidth=0.8),
                    zorder=7)
        sll = side_lobe_level(pattern_db, angles_deg)
        if not np.isinf(sll) and sll < -5:
            sll_level = peak_db + sll
            ax.plot(np.linspace(0, 2 * np.pi, 400),
                    np.full(400, sll_level),
                    color=_SLL_COLOR, linestyle='--',
                    linewidth=1.2, alpha=0.75, zorder=5)
            ax.text(np.deg2rad(65), sll_level + 1.0,
                    f'SLL={sll:.1f} dB', fontsize=7, color=_SLL_COLOR,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec=_SLL_COLOR, alpha=0.90, linewidth=0.7),
                    zorder=7)
        nulls = _find_nulls(pattern_db, angles_deg)
        for null_ang in nulls[:10]:
            ax.plot(np.deg2rad(null_ang), r_min + 0.8, 'v',
                    color=_NULL_COLOR, markersize=5,
                    alpha=0.85, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.4)


def _plot_cartesian_on_ax(ax, pattern_db: np.ndarray, angles_deg: np.ndarray,
                           dyn_range: float = 40, color: str = None,
                           label: str = '', annotate: bool = True):
    """Draw a Cartesian pattern on an existing Cartesian axes."""
    color    = color or PALETTE[0]
    peak_db  = float(np.max(pattern_db))
    floor_db = peak_db - dyn_range
    peak_idx = int(np.argmax(pattern_db))

    ax.plot(angles_deg, pattern_db, color=color, linewidth=1.8,
            label=label if label else '_nolegend_', zorder=4)
    ax.fill_between(angles_deg, pattern_db, floor_db,
                    alpha=0.07, color=color, zorder=2)
    ml_db = np.where(pattern_db >= peak_db - 10, pattern_db, floor_db)
    ax.fill_between(angles_deg, ml_db, floor_db,
                    alpha=0.17, color=color, zorder=3)

    # Reference lines
    ax.axhline(peak_db - 3,  color=_HPBW_COLOR, linestyle='--',
               linewidth=1.1, alpha=0.9, zorder=3)
    ax.axhline(peak_db - 10, color=_SLL_COLOR,  linestyle=':',
               linewidth=1.0, alpha=0.85, zorder=3)

    x_right = float(angles_deg[-1])
    ax.text(x_right, peak_db - 3 + 0.4, '−3 dB', va='bottom', ha='right',
            fontsize=7.5, color=_HPBW_COLOR, fontweight='bold')
    ax.text(x_right, peak_db - 10 + 0.4, '−10 dB', va='bottom', ha='right',
            fontsize=7.5, color=_SLL_COLOR, fontweight='bold')

    ax.set_xlim([float(angles_deg[0]), float(angles_deg[-1])])
    ax.set_ylim([floor_db - 1, peak_db + dyn_range * 0.10])
    ax.grid(True, alpha=0.35, linestyle='--')


def _draw_metrics_table(ax, rp, dyn_range: float = 40):
    """Render a styled metrics table on a plain Axes."""
    ax.axis('off')
    sm = rp.summary()

    # Compute extra metrics
    theta_deg = np.rad2deg(rp.theta)
    e_cut     = rp.phi_cut(0.0)
    h_cut     = rp.phi_cut(90.0)
    fnbw_e    = beamwidth_fnbw(e_cut, theta_deg)
    fnbw_h    = beamwidth_fnbw(h_cut, theta_deg)

    # Beam solid angle: Ω_A = ∫ (F/F_max) dΩ = 4π / D_max
    beam_solid_angle = 4.0 * PI / (rp.peak_directivity_linear + 1e-30)

    # Axial ratio at peak direction
    pi_pk, pj_pk = np.unravel_index(
        np.argmax(rp.directivity_2d), rp.directivity_2d.shape)
    from pylobe.analysis.metrics import axial_ratio
    ar = axial_ratio(
        complex(rp.E_theta[pi_pk, pj_pk]),
        complex(rp.E_phi[pi_pk, pj_pk]),
    )
    ar_db = 20.0 * np.log10(max(ar, 1.0))

    rows = [
        ('Frequency',         f'{sm.freq_ghz:.4f} GHz'),
        ('Peak Gain',         f'{sm.peak_gain_dbi:.2f} dBi'),
        ('Peak Direction',    f'θ={sm.theta_max_deg:.1f}°, φ={sm.phi_max_deg:.1f}°'),
        ('HPBW  E-plane',     f'{sm.hpbw_e_deg:.1f}°'),
        ('HPBW  H-plane',     f'{sm.hpbw_h_deg:.1f}°'),
        ('FNBW  E-plane',     f'{fnbw_e:.1f}°'),
        ('FNBW  H-plane',     f'{fnbw_h:.1f}°'),
        ('SLL',               f'{sm.sll_db:.1f} dB'),
        ('F/B Ratio',         f'{sm.fbr_db:.1f} dB'),
        ('Beam Solid Angle',  f'{beam_solid_angle:.4f} sr'),
        ('Axial Ratio @ peak',f'{ar:.2f}  ({ar_db:.1f} dB)'),
    ]

    col_labels  = ['Parameter', 'Value']
    cell_text   = [[r[0], r[1]] for r in rows]
    row_colours = [(_ROW_EVEN if i % 2 == 0 else _ROW_ODD)
                   for i in range(len(rows))]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center', cellLoc='left',
        colWidths=[0.52, 0.48],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.85)

    # Header styling
    for j in range(2):
        cell = tbl[(0, j)]
        cell.set_facecolor(_HDR_BG)
        cell.get_text().set_color(_HDR_FG)
        cell.get_text().set_fontweight('bold')

    # Row alternate colouring
    for i, fc in enumerate(row_colours, start=1):
        for j in range(2):
            tbl[(i, j)].set_facecolor(fc)
            tbl[(i, j)].set_edgecolor('#D0D3D4')

    ax.set_title('Antenna Metrics', fontsize=10, fontweight='bold',
                 color='#2C3E50', pad=6)


def _draw_sparam_table(ax, freqs: np.ndarray, S11_complex: np.ndarray,
                        Z: np.ndarray, Z0: float = 50.0):
    """Render S-parameter summary table on a plain Axes."""
    ax.axis('off')
    S11_db  = 20.0 * np.log10(np.clip(np.abs(S11_complex), 1e-10, None))
    vswr    = (1 + np.abs(S11_complex)) / (1 - np.abs(S11_complex) + 1e-30)
    rl_db   = -S11_db

    idx_res = int(np.argmin(S11_db))
    f_res   = float(freqs[idx_res]) / 1e9
    s11_min = float(S11_db[idx_res])
    vswr_at_res = float(vswr[idx_res])
    rl_at_res   = float(rl_db[idx_res])
    Z_at_res    = complex(Z[idx_res])

    # -10 dB bandwidth
    below = S11_db < -10
    bw_mhz, fbw_pct = 0.0, 0.0
    if below.any():
        lo = int(np.argmax(below))
        hi = len(below) - 1 - int(np.argmax(below[::-1]))
        bw_mhz  = (float(freqs[hi]) - float(freqs[lo])) / 1e6
        fc_band = (float(freqs[lo]) + float(freqs[hi])) / 2e9
        fbw_pct = 200.0 * bw_mhz / (fc_band * 1e3 + 1e-30)

    rows = [
        ('Resonant Frequency',   f'{f_res:.4f} GHz'),
        ('S₁₁ at Resonance',     f'{s11_min:.1f} dB'),
        ('Return Loss',          f'{rl_at_res:.1f} dB'),
        ('VSWR at Resonance',    f'{vswr_at_res:.2f}'),
        ('10-dB Bandwidth',      f'{bw_mhz:.1f} MHz'),
        ('Fractional Bandwidth', f'{fbw_pct:.2f} %'),
        ('Z @ resonance',        f'{Z_at_res.real:.1f} + j{Z_at_res.imag:.1f} Ω'),
        ('Ref. Impedance Z₀',    f'{Z0:.0f} Ω'),
    ]

    cell_text   = [[r[0], r[1]] for r in rows]
    row_colours = [(_ROW_EVEN if i % 2 == 0 else _ROW_ODD)
                   for i in range(len(rows))]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=['Parameter', 'Value'],
        loc='center', cellLoc='left',
        colWidths=[0.55, 0.45],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.85)
    for j in range(2):
        cell = tbl[(0, j)]
        cell.set_facecolor(_HDR_BG)
        cell.get_text().set_color(_HDR_FG)
        cell.get_text().set_fontweight('bold')
    for i, fc in enumerate(row_colours, start=1):
        for j in range(2):
            tbl[(i, j)].set_facecolor(fc)
            tbl[(i, j)].set_edgecolor('#D0D3D4')

    ax.set_title('S-Parameter Summary', fontsize=10, fontweight='bold',
                 color='#2C3E50', pad=6)


# ─────────────────────────────────────────────────────────────────────────────
# Public: Pattern dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_pattern_dashboard(radiation_pattern,
                            dyn_range: float = 40,
                            title: str = '') -> plt.Figure:
    """Comprehensive 2-D radiation pattern dashboard.

    Layout (4 rows × 3 cols)
    -------------------------
    Row 0 : E-plane polar  |  H-plane polar  |  Metrics table
    Row 1 : E-plane cart.  |  H-plane cart.  |  Phase pattern (E-plane)
    Row 2 : Gain heatmap (θ vs φ) — spanning all 3 columns
    Row 3 : φ = 45° cut (polar)  |  φ = 135° cut (polar)  |  Diagonal cuts cartesian

    All annotated with HPBW, SLL, nulls, −3/−10 dB references.
    Saved at 600 DPI via ``export_fig()``.

    Parameters
    ----------
    radiation_pattern : RadiationPattern
    dyn_range : float
        Polar/Cartesian dynamic range [dB].
    title : str
        Extra title suffix (e.g. antenna name / frequency).

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    rp        = radiation_pattern
    theta_deg = np.rad2deg(rp.theta)
    sm        = rp.summary()
    freq_str  = f'{rp.freq / 1e9:.3f} GHz'
    main_title = (f'Radiation Pattern Dashboard — {title}  @  {freq_str}'
                  if title else f'Radiation Pattern Dashboard — {freq_str}')

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor('white')

    # Super-title
    fig.suptitle(main_title, fontsize=14, fontweight='bold',
                 color='#2C3E50', y=0.995)

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.52, wspace=0.38,
        top=0.975, bottom=0.03, left=0.06, right=0.97,
        height_ratios=[1.6, 1.0, 1.0, 1.4],
    )

    # ── Row 0: E-polar, H-polar, Metrics ─────────────────────────────────────
    ax_ep = fig.add_subplot(gs[0, 0], projection='polar')
    ax_hp = fig.add_subplot(gs[0, 1], projection='polar')
    ax_mt = fig.add_subplot(gs[0, 2])

    def _cut_360(cut_lin):
        p_max = float(np.max(cut_lin)) or 1.0
        db    = 10.0 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
        a360  = np.concatenate([theta_deg, theta_deg + 180.0])
        d360  = np.concatenate([db, db[::-1]])
        return a360, d360

    e_ang360, e_db360 = _cut_360(rp.phi_cut(0.0))
    h_ang360, h_db360 = _cut_360(rp.phi_cut(90.0))

    _plot_polar_on_ax(ax_ep, e_db360, e_ang360, dyn_range,
                      PALETTE[0], 'E-plane (φ=0°)')
    ax_ep.set_title('E-Plane Pattern', pad=16, fontsize=11,
                    fontweight='bold', color='#2C3E50')

    _plot_polar_on_ax(ax_hp, h_db360, h_ang360, dyn_range,
                      PALETTE[1], 'H-plane (φ=90°)')
    ax_hp.set_title('H-Plane Pattern', pad=16, fontsize=11,
                    fontweight='bold', color='#2C3E50')

    _draw_metrics_table(ax_mt, rp, dyn_range)

    # ── Row 1: E-cart, H-cart, Phase ─────────────────────────────────────────
    ax_ec = fig.add_subplot(gs[1, 0])
    ax_hc = fig.add_subplot(gs[1, 1])
    ax_ph = fig.add_subplot(gs[1, 2])

    e_db_h = rp.theta_cut_db(0.0)
    h_db_h = rp.theta_cut_db(90.0)

    _plot_cartesian_on_ax(ax_ec, e_db_h, theta_deg, dyn_range,
                           PALETTE[0], 'E-plane')
    ax_ec.set_xlabel('θ (degrees)', fontsize=10)
    ax_ec.set_ylabel('Norm. Gain (dB)', fontsize=10)
    ax_ec.set_title('E-Plane (Cartesian)', fontsize=11,
                    fontweight='bold', color='#2C3E50')

    _plot_cartesian_on_ax(ax_hc, h_db_h, theta_deg, dyn_range,
                           PALETTE[1], 'H-plane')
    ax_hc.set_xlabel('θ (degrees)', fontsize=10)
    ax_hc.set_ylabel('Norm. Gain (dB)', fontsize=10)
    ax_hc.set_title('H-Plane (Cartesian)', fontsize=11,
                    fontweight='bold', color='#2C3E50')

    # Phase pattern at E-plane
    phase_2d  = rp.phase_pattern('theta')
    phi_rad_0 = 0.0
    idx_phi0  = int(np.argmin(np.abs(rp.phi - phi_rad_0)))
    phase_cut = phase_2d[:, idx_phi0]
    ax_ph.plot(theta_deg, phase_cut, color=PALETTE[2], linewidth=1.8)
    ax_ph.fill_between(theta_deg, phase_cut, 0,
                        where=(phase_cut >= 0),
                        alpha=0.12, color=PALETTE[0], interpolate=True)
    ax_ph.fill_between(theta_deg, phase_cut, 0,
                        where=(phase_cut < 0),
                        alpha=0.12, color=PALETTE[1], interpolate=True)
    ax_ph.axhline(0, color='#95A5A6', linewidth=0.8, linestyle=':')
    ax_ph.set_ylim([-195, 195])
    ax_ph.set_yticks(range(-180, 181, 45))
    ax_ph.set_xlabel('θ (degrees)', fontsize=10)
    ax_ph.set_ylabel('Phase (degrees)', fontsize=10)
    ax_ph.grid(True, alpha=0.35, linestyle='--')
    ax_ph.set_title('Phase Pattern  Eθ | φ=0°', fontsize=11,
                    fontweight='bold', color='#2C3E50')

    # ── Row 2: Gain heatmap (full width) ──────────────────────────────────────
    ax_hm = fig.add_subplot(gs[2, :])
    from pylobe.visualization.heatmap import plot_gain_heatmap_mpl
    plot_gain_heatmap_mpl(rp, ax=ax_hm, dyn_range=dyn_range)

    # ── Row 3: Diagonal cuts ──────────────────────────────────────────────────
    ax_d0  = fig.add_subplot(gs[3, 0], projection='polar')
    ax_d1  = fig.add_subplot(gs[3, 1], projection='polar')
    ax_dc  = fig.add_subplot(gs[3, 2])

    for phi_cut, ax_d, col, label in [
        (45.0,  ax_d0, PALETTE[3], 'φ=45°'),
        (135.0, ax_d1, PALETTE[4], 'φ=135°'),
    ]:
        cut_lin      = rp.phi_cut(phi_cut)
        a360, d360   = _cut_360(cut_lin)
        _plot_polar_on_ax(ax_d, d360, a360, dyn_range, col, label,
                          annotate=False)
        ax_d.set_title(f'φ = {phi_cut:.0f}° Cut', pad=14,
                       fontsize=10, fontweight='bold', color='#2C3E50')

    # All cuts overlaid in Cartesian
    for phi_cut, col, label in [
        (0.0,  PALETTE[0], 'E-plane (0°)'),
        (45.0, PALETTE[3], 'φ=45°'),
        (90.0, PALETTE[1], 'H-plane (90°)'),
        (135.0,PALETTE[4], 'φ=135°'),
    ]:
        cut_lin = rp.phi_cut(phi_cut)
        p_max   = float(np.max(cut_lin)) or 1.0
        db_cut  = 10.0 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
        ax_dc.plot(theta_deg, db_cut, color=col, linewidth=1.6, label=label)

    ax_dc.axhline(-3,  color=_HPBW_COLOR, linestyle='--',
                  linewidth=1.0, alpha=0.8)
    ax_dc.axhline(-10, color=_SLL_COLOR, linestyle=':',
                  linewidth=0.9, alpha=0.75)
    ax_dc.set_ylim([-dyn_range - 1, 2])
    ax_dc.set_xlabel('θ (degrees)', fontsize=10)
    ax_dc.set_ylabel('Norm. Gain (dB)', fontsize=10)
    ax_dc.set_title('Multi-Cut Overlay (Cartesian)', fontsize=11,
                    fontweight='bold', color='#2C3E50')
    ax_dc.legend(framealpha=0.9, fontsize=8.5, loc='lower center',
                 ncol=2, bbox_to_anchor=(0.5, -0.25))
    ax_dc.grid(True, alpha=0.35, linestyle='--')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: Frequency-response dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_frequency_dashboard(freqs: np.ndarray,
                              S11_complex: np.ndarray,
                              Z_complex: np.ndarray,
                              Z0: float = 50.0,
                              title: str = '') -> plt.Figure:
    """Comprehensive frequency-response dashboard.

    Layout (3 rows × 3 cols)
    -------------------------
    Row 0 : S₁₁ (dB)  |  VSWR  |  Return Loss
    Row 1 : R(Z)       |  X(Z)  |  |Z| and ∠Z
    Row 2 : Group delay  |  Phase of S₁₁  |  S-parameter table

    Parameters
    ----------
    freqs : ndarray [Hz]
    S11_complex : ndarray of complex
    Z_complex : ndarray of complex [Ω]
    Z0 : float  [Ω]
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    freqs        = np.asarray(freqs,       dtype=float)
    S11_complex  = np.asarray(S11_complex, dtype=complex)
    Z_complex    = np.asarray(Z_complex,   dtype=complex)

    freq_ghz = freqs / 1e9
    S11_db   = 20.0 * np.log10(np.clip(np.abs(S11_complex), 1e-10, None))
    vswr     = (1 + np.abs(S11_complex)) / np.clip(1 - np.abs(S11_complex), 1e-6, None)
    rl_db    = -S11_db
    R        = Z_complex.real
    X        = Z_complex.imag
    Z_mag    = np.abs(Z_complex)
    Z_ang    = np.angle(Z_complex, deg=True)
    gamma    = (Z_complex - Z0) / (Z_complex + Z0)
    S11_ph   = np.angle(gamma, deg=True)

    # Group delay from S11 phase
    omega    = 2 * np.pi * freqs
    gd_ns    = -np.gradient(np.unwrap(np.angle(S11_complex)), omega) * 1e9

    main_title = f'Frequency Response Dashboard — {title}' if title else \
                 'Frequency Response Dashboard'

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor('white')
    fig.suptitle(main_title, fontsize=14, fontweight='bold',
                 color='#2C3E50', y=0.998)

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.38,
        top=0.975, bottom=0.05, left=0.07, right=0.97,
        height_ratios=[1.0, 1.0, 1.1],
    )

    def _style_ax(ax, xlabel='Frequency (GHz)', ylabel=''):
        ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim([float(freq_ghz[0]), float(freq_ghz[-1])])
        ax.grid(True, alpha=0.35, linestyle='--')
        ax.tick_params(labelsize=9)

    # ── Row 0: S11, VSWR, Return loss ────────────────────────────────────────
    ax_s11 = fig.add_subplot(gs[0, 0])
    ax_s11.plot(freq_ghz, S11_db, color=PALETTE[0], linewidth=2.0)
    ax_s11.axhline(-10, color='#E74C3C', linestyle='--', linewidth=1.2)
    ax_s11.fill_between(freq_ghz, S11_db, -10,
                         where=(S11_db < -10), alpha=0.14, color='#E74C3C')
    idx_min = int(np.argmin(S11_db))
    ax_s11.annotate(
        f'{S11_db[idx_min]:.1f} dB\n@ {freq_ghz[idx_min]:.3f} GHz',
        xy=(float(freq_ghz[idx_min]), float(S11_db[idx_min])),
        xytext=(float(freq_ghz[idx_min]) + (freq_ghz[-1] - freq_ghz[0]) * 0.07,
                float(S11_db[idx_min]) + 6),
        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2),
        fontsize=8.5, color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                  ec='#BDC3C7', alpha=0.92),
    )
    ax_s11.set_ylim([min(float(S11_db.min()) - 4, -40), 2])
    ax_s11.set_title('|S₁₁| vs Frequency', fontsize=11,
                     fontweight='bold', color='#2C3E50')
    _style_ax(ax_s11, ylabel='|S₁₁| (dB)')

    ax_vswr = fig.add_subplot(gs[0, 1])
    ax_vswr.plot(freq_ghz, vswr, color=PALETTE[1], linewidth=2.0)
    ax_vswr.axhline(2.0, color='#E74C3C', linestyle='--', linewidth=1.2)
    ax_vswr.fill_between(freq_ghz, 1, vswr,
                          where=(vswr < 2.0), alpha=0.13, color='#27AE60')
    ax_vswr.set_ylim([1.0, min(float(vswr.max()) + 1, 15)])
    ax_vswr.set_title('VSWR vs Frequency', fontsize=11,
                      fontweight='bold', color='#2C3E50')
    _style_ax(ax_vswr, ylabel='VSWR')

    ax_rl = fig.add_subplot(gs[0, 2])
    ax_rl.plot(freq_ghz, rl_db, color=PALETTE[2], linewidth=2.0)
    ax_rl.axhline(10, color='#27AE60', linestyle='--', linewidth=1.2)
    ax_rl.fill_between(freq_ghz, rl_db, 10,
                        where=(rl_db >= 10), alpha=0.13, color='#27AE60')
    ax_rl.set_ylim(bottom=0)
    ax_rl.set_title('Return Loss vs Frequency', fontsize=11,
                    fontweight='bold', color='#2C3E50')
    _style_ax(ax_rl, ylabel='Return Loss (dB)')

    # ── Row 1: R(Z), X(Z), |Z| & ∠Z ─────────────────────────────────────────
    ax_R = fig.add_subplot(gs[1, 0])
    ax_R.plot(freq_ghz, R, color=PALETTE[0], linewidth=2.0, label='R (Ω)')
    ax_R.axhline(Z0, color='#27AE60', linestyle=':', linewidth=1.0,
                 alpha=0.7, label=f'Z₀ = {Z0:.0f} Ω')
    ax_R.fill_between(freq_ghz, R, Z0, where=(R < Z0),
                       alpha=0.07, color=PALETTE[0], interpolate=True)
    ax_R.fill_between(freq_ghz, R, Z0, where=(R > Z0),
                       alpha=0.07, color=PALETTE[1], interpolate=True)
    # Resonance marker
    idx_res = int(np.argmin(np.abs(X)))
    ax_R.axvline(float(freq_ghz[idx_res]), color='#95A5A6',
                 linewidth=0.9, linestyle=':', alpha=0.7)
    ax_R.text(float(freq_ghz[idx_res]), float(ax_R.get_ylim()[1]) * 0.92,
              f'f_r={freq_ghz[idx_res]:.3f}G', fontsize=7.5,
              color='#555', ha='center',
              bbox=dict(boxstyle='round,pad=0.15', fc='white',
                        ec='#95A5A6', alpha=0.85))
    ax_R.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
    ax_R.set_title('Resistance R(Z)', fontsize=11,
                   fontweight='bold', color='#2C3E50')
    _style_ax(ax_R, ylabel='R (Ω)')

    ax_X = fig.add_subplot(gs[1, 1])
    ax_X.plot(freq_ghz, X, color=PALETTE[1], linewidth=2.0, label='X (Ω)')
    ax_X.axhline(0, color='#BDC3C7', linewidth=0.8, linestyle='--')
    ax_X.axvline(float(freq_ghz[idx_res]), color='#95A5A6',
                 linewidth=0.9, linestyle=':', alpha=0.7)
    ax_X.fill_between(freq_ghz, X, 0, where=(X >= 0),
                       alpha=0.10, color=PALETTE[0], interpolate=True,
                       label='Inductive')
    ax_X.fill_between(freq_ghz, X, 0, where=(X < 0),
                       alpha=0.10, color=PALETTE[1], interpolate=True,
                       label='Capacitive')
    ax_X.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
    ax_X.set_title('Reactance X(Z)', fontsize=11,
                   fontweight='bold', color='#2C3E50')
    _style_ax(ax_X, ylabel='X (Ω)')

    ax_Zmag = fig.add_subplot(gs[1, 2])
    ax_Zang = ax_Zmag.twinx()
    lZ, = ax_Zmag.plot(freq_ghz, Z_mag, color=PALETTE[0],
                        linewidth=2.0, label='|Z| (Ω)')
    la, = ax_Zang.plot(freq_ghz, Z_ang, color=PALETTE[3],
                        linewidth=1.6, linestyle='--', label='∠Z (°)')
    ax_Zmag.set_ylabel('|Z| (Ω)', fontsize=10)
    ax_Zang.set_ylabel('∠Z (degrees)', fontsize=10, color=PALETTE[3])
    ax_Zang.tick_params(axis='y', labelcolor=PALETTE[3], labelsize=9)
    h1, l1 = ax_Zmag.get_legend_handles_labels()
    h2, l2 = ax_Zang.get_legend_handles_labels()
    ax_Zmag.legend(h1 + h2, l1 + l2, fontsize=8.5, framealpha=0.9)
    ax_Zmag.set_title('|Z| and ∠Z vs Frequency', fontsize=11,
                      fontweight='bold', color='#2C3E50')
    _style_ax(ax_Zmag, ylabel='|Z| (Ω)')

    # ── Row 2: Group delay, Phase S11, Table ─────────────────────────────────
    ax_gd = fig.add_subplot(gs[2, 0])
    ax_gd.plot(freq_ghz, gd_ns, color=PALETTE[0], linewidth=2.0)
    ax_gd.axhline(0, color='#95A5A6', linewidth=0.8, linestyle=':')
    ax_gd.set_title('Group Delay  (−dφ/dω)', fontsize=11,
                    fontweight='bold', color='#2C3E50')
    _style_ax(ax_gd, ylabel='Group Delay (ns)')

    ax_ph = fig.add_subplot(gs[2, 1])
    ax_ph.plot(freq_ghz, S11_ph, color=PALETTE[2], linewidth=2.0)
    ax_ph.fill_between(freq_ghz, S11_ph, 0,
                        where=(S11_ph >= 0), alpha=0.10, color=PALETTE[0],
                        interpolate=True)
    ax_ph.fill_between(freq_ghz, S11_ph, 0,
                        where=(S11_ph < 0), alpha=0.10, color=PALETTE[1],
                        interpolate=True)
    ax_ph.axhline(0, color='#95A5A6', linewidth=0.8, linestyle=':')
    ax_ph.set_ylim([-195, 195])
    ax_ph.set_yticks(range(-180, 181, 45))
    ax_ph.set_title('Phase of S₁₁', fontsize=11,
                    fontweight='bold', color='#2C3E50')
    _style_ax(ax_ph, ylabel='Phase S₁₁ (degrees)')

    ax_tbl = fig.add_subplot(gs[2, 2])
    _draw_sparam_table(ax_tbl, freqs, S11_complex, Z_complex, Z0)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public: Combined design dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_design_dashboard(design,
                           dyn_range: float = 40,
                           Z0: float = 50.0) -> tuple:
    """Create all dashboard figures for a solved ``AntennaDesign``.

    Returns a tuple ``(fig_pattern, fig_freq)`` of matplotlib figures.
    Both are at 600 DPI when exported via ``export_fig()``.

    Parameters
    ----------
    design : AntennaDesign  (must be solved)
    dyn_range : float
    Z0 : float  [Ω]

    Returns
    -------
    tuple (fig_pattern : plt.Figure, fig_freq : plt.Figure or None)
    """
    design._require_solved()
    geom_name = type(design.geometry).__name__

    fig_pat = plot_pattern_dashboard(
        design._radiation_pattern,
        dyn_range=dyn_range,
        title=geom_name,
    )

    fig_freq = None
    if design._s11_data is not None:
        freqs, S11 = design._s11_data
        _, Z        = design._z_data
        fig_freq = plot_frequency_dashboard(
            freqs, S11, Z, Z0=Z0, title=geom_name,
        )

    return fig_pat, fig_freq


# ─────────────────────────────────────────────────────────────────────────────
# Public: Multi-antenna pattern comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_compare_patterns(rp_list: list,
                           labels: list = None,
                           dyn_range: float = 40,
                           title: str = 'Antenna Pattern Comparison') -> plt.Figure:
    """Overlay E/H patterns and gain bar chart for multiple antennas.

    Layout (2 rows × 3 cols)
    -------------------------
    Row 0 : E-plane polar overlay  |  H-plane polar overlay  |  Gain bar chart
    Row 1 : E-plane cartesian overlay  |  H-plane cartesian overlay  |  Metrics table

    Parameters
    ----------
    rp_list : list of RadiationPattern
    labels : list of str or None
    dyn_range : float
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    if labels is None:
        labels = [f'Antenna {i + 1}' for i in range(len(rp_list))]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=14, fontweight='bold',
                 color='#2C3E50', y=0.998)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.48, wspace=0.38,
        top=0.970, bottom=0.06, left=0.06, right=0.97,
        height_ratios=[1.6, 1.0],
    )

    ax_ep = fig.add_subplot(gs[0, 0], projection='polar')
    ax_hp = fig.add_subplot(gs[0, 1], projection='polar')
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_ec  = fig.add_subplot(gs[1, 0])
    ax_hc  = fig.add_subplot(gs[1, 1])
    ax_cmp = fig.add_subplot(gs[1, 2])

    peak_gains, hpbw_e_list, hpbw_h_list, sll_list = [], [], [], []

    def _cut_360(rp, phi):
        cut_lin = rp.phi_cut(phi)
        p_max   = float(np.max(cut_lin)) or 1.0
        db      = 10.0 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
        theta_d = np.rad2deg(rp.theta)
        a360    = np.concatenate([theta_d, theta_d + 180.0])
        d360    = np.concatenate([db, db[::-1]])
        return a360, d360, theta_d

    # Compute peak_db across all patterns to share colour range
    all_peak = max(float(rp.peak_directivity_dbi) for rp in rp_list)

    for i, (rp, lbl) in enumerate(zip(rp_list, labels)):
        col = PALETTE[i % len(PALETTE)]
        sm  = rp.summary()
        peak_gains.append(sm.peak_gain_dbi)
        hpbw_e_list.append(sm.hpbw_e_deg)
        hpbw_h_list.append(sm.hpbw_h_deg)
        sll_list.append(sm.sll_db)

        e_ang360, e_db360, theta_d = _cut_360(rp, 0.0)
        h_ang360, h_db360, _       = _cut_360(rp, 90.0)

        _plot_polar_on_ax(ax_ep, e_db360, e_ang360, dyn_range,
                          col, lbl, annotate=(i == 0))
        _plot_polar_on_ax(ax_hp, h_db360, h_ang360, dyn_range,
                          col, lbl, annotate=(i == 0))

        e_db_h = rp.theta_cut_db(0.0)
        h_db_h = rp.theta_cut_db(90.0)
        ax_ec.plot(theta_d, e_db_h, color=col, linewidth=1.8, label=lbl)
        ax_hc.plot(theta_d, h_db_h, color=col, linewidth=1.8, label=lbl)

    ax_ep.set_title('E-Plane Overlay', pad=16, fontsize=11,
                    fontweight='bold', color='#2C3E50')
    ax_hp.set_title('H-Plane Overlay', pad=16, fontsize=11,
                    fontweight='bold', color='#2C3E50')

    # Polar legends below discs
    for ax in [ax_ep, ax_hp]:
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, lbls,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.00),
                       ncol=min(len(lbls), 4),
                       framealpha=0.92, fontsize=9)

    # Cartesian axes
    for ax, title_str in [(ax_ec, 'E-Plane (Cartesian)'),
                           (ax_hc, 'H-Plane (Cartesian)')]:
        ax.axhline(-3,  color=_HPBW_COLOR, linestyle='--', linewidth=1.0, alpha=0.8)
        ax.axhline(-10, color=_SLL_COLOR,  linestyle=':',  linewidth=0.9, alpha=0.75)
        ax.set_ylim([-dyn_range - 1, 2])
        ax.set_xlabel('θ (degrees)', fontsize=10)
        ax.set_ylabel('Norm. Gain (dB)', fontsize=10)
        ax.set_title(title_str, fontsize=11, fontweight='bold', color='#2C3E50')
        ax.legend(framealpha=0.9, fontsize=8.5, loc='lower center',
                  ncol=min(len(labels), 3), bbox_to_anchor=(0.5, -0.28))
        ax.grid(True, alpha=0.35, linestyle='--')
        ax.tick_params(labelsize=9)

    # Bar chart: Peak gain comparison
    x     = np.arange(len(labels))
    bars  = ax_bar.bar(x, peak_gains, color=PALETTE[:len(labels)],
                       edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
    for bar, val in zip(bars, peak_gains):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#2C3E50')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
    ax_bar.set_ylabel('Peak Gain (dBi)', fontsize=10)
    ax_bar.set_title('Peak Gain Comparison', fontsize=11,
                     fontweight='bold', color='#2C3E50')
    ax_bar.grid(True, axis='y', alpha=0.35, linestyle='--')
    ax_bar.tick_params(labelsize=9)

    # Metrics comparison table
    ax_cmp.axis('off')
    col_labels = ['Antenna'] + labels
    rows = [
        ['Peak Gain (dBi)']   + [f'{v:.2f}' for v in peak_gains],
        ['HPBW E (°)']        + [f'{v:.1f}' for v in hpbw_e_list],
        ['HPBW H (°)']        + [f'{v:.1f}' for v in hpbw_h_list],
        ['SLL (dB)']          + [f'{v:.1f}' for v in sll_list],
    ]
    n_col = len(col_labels)
    tbl = ax_cmp.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center', cellLoc='center',
        colWidths=[0.28] + [0.72 / len(labels)] * len(labels),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 2.0)
    for j in range(n_col):
        tbl[(0, j)].set_facecolor(_HDR_BG)
        tbl[(0, j)].get_text().set_color(_HDR_FG)
        tbl[(0, j)].get_text().set_fontweight('bold')
    for i in range(1, len(rows) + 1):
        fc = _ROW_EVEN if i % 2 == 0 else _ROW_ODD
        for j in range(n_col):
            tbl[(i, j)].set_facecolor(fc)
            tbl[(i, j)].set_edgecolor('#D0D3D4')
    ax_cmp.set_title('Metrics Comparison', fontsize=11,
                     fontweight='bold', color='#2C3E50', pad=8)

    return fig
