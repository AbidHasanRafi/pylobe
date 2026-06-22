"""
Comprehensive 2-D Pattern Visualization
=========================================
Deep-dive into every pattern visualization available in PyLobe,
demonstrating research-grade output suitable for conference papers
and theses.  All static exports are at 600 DPI.

Visualizations covered
-----------------------
1.  E/H polar patterns with HPBW, SLL, null annotations
2.  E/H Cartesian patterns with -3/-10 dB reference lines
3.  Phase pattern (E_theta, H_theta)
4.  Full gain heatmap — interactive Plotly + static matplotlib
5.  Comprehensive 4x3 pattern dashboard
6.  Diagonal phi-cut polar overlays  (phi = 0/45/90/135 deg)
7.  Multi-cut Cartesian overlay
8.  3-D radiation pattern with E/H cuts + iso-gain rings
9.  Lobe decomposition (Plotly 3-D)
10. Normalized pattern on IEEE polar axes (publication style)
11. Pattern comparison overlay: two patch antennas side-by-side
12. Axial-ratio 2-D map (colormap over theta-phi)
13. Waterfall: E-plane pattern vs frequency (21 points)

Antennas used
-------------
- Rectangular patch 2.4 GHz on FR4  (main antenna)
- Rectangular patch 5.8 GHz on RT5880  (comparison)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pylobe.geometry.patch   import RectangularPatch
from pylobe.geometry.base    import FR4, RT5880, COPPER, PEC
from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
from pylobe.analysis.metrics import axial_ratio_2d, beamwidth_hpbw, side_lobe_level
from pylobe.visualization.style   import set_style, export_fig
from pylobe.visualization.polar   import (
    plot_polar, plot_polar_compare, plot_e_h_plane, plot_phase_pattern,
)
from pylobe.visualization.cartesian   import plot_pattern_cartesian
from pylobe.visualization.heatmap     import (
    plot_gain_heatmap, plot_gain_heatmap_mpl, plot_waterfall,
)
from pylobe.visualization.lobe3d      import (
    plot_3d_radiation, plot_lobe_decomposition,
)
from pylobe.visualization.dashboard   import (
    plot_pattern_dashboard, plot_compare_patterns,
)

set_style('default')

# ─────────────────────────────────────────────────────────────────────────────
# Solve both antennas
# ─────────────────────────────────────────────────────────────────────────────
print("Solving antennas...")

patch24 = RectangularPatch(freq=2.4e9, substrate_material=FR4,
                            h=1.6e-3, patch_material=COPPER,
                            ground_material=PEC, inset_feed=True)
solver24 = PatchAnalyticalSolver(patch24, freq=2.4e9)
rp24     = solver24.radiation_pattern(n_theta=181, n_phi=73)
sm24     = rp24.summary()

patch58 = RectangularPatch(freq=5.8e9, substrate_material=RT5880,
                            h=0.787e-3, patch_material=COPPER,
                            ground_material=PEC, inset_feed=True)
solver58 = PatchAnalyticalSolver(patch58, freq=5.8e9)
rp58     = solver58.radiation_pattern(n_theta=181, n_phi=73)

theta_deg = np.rad2deg(rp24.theta)
print(f"Patch 2.4 GHz: peak={sm24.peak_gain_dbi:.2f} dBi  "
      f"HPBW-E={sm24.hpbw_e_deg:.1f}deg  SLL={sm24.sll_db:.1f}dB")


# ─────────────────────────────────────────────────────────────────────────────
# 1. E/H polar — annotated, standalone
# ─────────────────────────────────────────────────────────────────────────────
fig_eh = plot_e_h_plane(rp24, dyn_range=40)
export_fig(fig_eh, 'dash_eh_polar.png', fmt='png')
plt.close('all')
print("Saved: dash_eh_polar.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. E/H Cartesian — using the pattern_cartesian function directly
# ─────────────────────────────────────────────────────────────────────────────
e_cut_lin = rp24.e_plane_cut()
h_cut_lin = rp24.h_plane_cut()
e_max = float(np.max(e_cut_lin)) or 1.0
h_max = float(np.max(h_cut_lin)) or 1.0
e_db  = 10 * np.log10(np.clip(e_cut_lin / e_max, 1e-10, None))
h_db  = 10 * np.log10(np.clip(h_cut_lin / h_max, 1e-10, None))

fig_cart, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(14, 5.5),
                                        sharey=True)
plot_pattern_cartesian(e_db, theta_deg, label='E-plane',
                        ax=ax_e, color='#0072B2',
                        freq=2.4e9, annotate=True)
ax_e.set_title('E-Plane (phi=0 deg)', fontsize=12,
                fontweight='bold', color='#2C3E50')

plot_pattern_cartesian(h_db, theta_deg, label='H-plane',
                        ax=ax_h, color='#D55E00',
                        freq=2.4e9, annotate=True)
ax_h.set_title('H-Plane (phi=90 deg)', fontsize=12,
                fontweight='bold', color='#2C3E50')

fig_cart.suptitle('Rectangular Patch 2.4 GHz FR4 -- E/H Patterns',
                   fontsize=13, fontweight='bold', color='#2C3E50')
fig_cart.tight_layout()
export_fig(fig_cart, 'dash_eh_cartesian.png', fmt='png')
plt.close('all')
print("Saved: dash_eh_cartesian.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase pattern — E_theta at phi=0, H_theta at phi=90
# ─────────────────────────────────────────────────────────────────────────────
fig_phase = plot_phase_pattern(rp24, component='theta')
export_fig(fig_phase, 'dash_phase_pattern.png', fmt='png')
plt.close('all')
print("Saved: dash_phase_pattern.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gain heatmap — interactive Plotly + static matplotlib 600 DPI
# ─────────────────────────────────────────────────────────────────────────────
fig_hm_plotly = plot_gain_heatmap(rp24, title='Patch 2.4 GHz -- Gain Heatmap')
fig_hm_plotly.write_html('dash_gain_heatmap_interactive.html')
print("Saved: dash_gain_heatmap_interactive.html")

fig_hm_mpl = plot_gain_heatmap_mpl(rp24, dyn_range=40)
export_fig(fig_hm_mpl, 'dash_gain_heatmap_600dpi.png', fmt='png')
plt.close('all')
print("Saved: dash_gain_heatmap_600dpi.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comprehensive 4x3 pattern dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig_dash = plot_pattern_dashboard(rp24, dyn_range=40,
                                   title='Patch 2.4 GHz FR4')
export_fig(fig_dash, 'dash_full_dashboard.png', fmt='png')
plt.close('all')
print("Saved: dash_full_dashboard.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diagonal phi-cut overlay — 4 cuts in one Cartesian plot
# ─────────────────────────────────────────────────────────────────────────────
phi_cuts   = [0.0, 45.0, 90.0, 135.0]
cut_labels = ['E-plane (0 deg)', 'phi=45 deg', 'H-plane (90 deg)', 'phi=135 deg']
cut_colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']

fig_cuts, ax_cuts = plt.subplots(figsize=(11, 5.5))
for phi, lbl, col in zip(phi_cuts, cut_labels, cut_colors):
    cut_lin = rp24.phi_cut(phi)
    p_max   = float(np.max(cut_lin)) or 1.0
    cut_db  = 10 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
    ax_cuts.plot(theta_deg, cut_db, color=col, linewidth=1.9, label=lbl)

ax_cuts.axhline(-3,  color='#C0392B', linestyle='--', linewidth=1.0, alpha=0.8)
ax_cuts.axhline(-10, color='#E67E22', linestyle=':',  linewidth=0.9, alpha=0.8)
ax_cuts.text(theta_deg[-1], -3 + 0.4,  '-3 dB',  ha='right', fontsize=8,
             color='#C0392B', fontweight='bold')
ax_cuts.text(theta_deg[-1], -10 + 0.4, '-10 dB', ha='right', fontsize=8,
             color='#E67E22', fontweight='bold')
ax_cuts.set_xlim([0, 180])
ax_cuts.set_ylim([-42, 2])
ax_cuts.set_xlabel('theta (degrees)', fontsize=12)
ax_cuts.set_ylabel('Normalized Gain (dB)', fontsize=12)
ax_cuts.set_title('Multi-Cut Pattern Overlay -- Patch 2.4 GHz FR4',
                   fontsize=13, fontweight='bold', color='#2C3E50')
ax_cuts.legend(framealpha=0.92, fontsize=10, loc='upper right')
ax_cuts.grid(True, alpha=0.35, linestyle='--')
fig_cuts.tight_layout()
export_fig(fig_cuts, 'dash_multi_cut_overlay.png', fmt='png')
plt.close('all')
print("Saved: dash_multi_cut_overlay.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 3-D radiation pattern
# ─────────────────────────────────────────────────────────────────────────────
fig_3d = plot_3d_radiation(rp24, show_e_h_cuts=True, show_isogain_rings=True,
                            title='Patch 2.4 GHz FR4')
fig_3d.write_html('dash_3d_pattern.html')
print("Saved: dash_3d_pattern.html")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Lobe decomposition (3-D)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pylobe.analysis.lobe import LobeAnalyzer
    lobes    = LobeAnalyzer(rp24).find_lobes()
    fig_lobe = plot_lobe_decomposition(rp24, lobes,
                                        title='Patch 2.4 GHz -- Lobe Decomposition')
    fig_lobe.write_html('dash_lobe_decomposition.html')
    print("Saved: dash_lobe_decomposition.html")
except Exception as exc:
    print(f"Lobe decomposition skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Publication-style polar (IEEE preset, single-column width)
# ─────────────────────────────────────────────────────────────────────────────
set_style('ieee')

e_db_180 = rp24.theta_cut_db(0.0)    # E-plane dB vs theta 0..180
theta_360 = np.concatenate([theta_deg, theta_deg + 180.0])
e_db_360  = np.concatenate([e_db_180, e_db_180[::-1]])

fig_ieee = plot_polar(e_db_360, theta_360,
                       label='E-plane 2.4 GHz',
                       dyn_range=40, color='#0072B2',
                       annotate=True)
export_fig(fig_ieee, 'dash_ieee_polar.png', fmt='png')
plt.close('all')
print("Saved: dash_ieee_polar.png  [IEEE style, 600 DPI]")

set_style('default')   # restore


# ─────────────────────────────────────────────────────────────────────────────
# 10. Pattern comparison: patch 2.4 GHz vs patch 5.8 GHz
# ─────────────────────────────────────────────────────────────────────────────
fig_cmp = plot_compare_patterns(
    rp_list=[rp24, rp58],
    labels=['Patch 2.4 GHz (FR4)', 'Patch 5.8 GHz (RT5880)'],
    dyn_range=40,
    title='Patch Antenna Pattern Comparison',
)
export_fig(fig_cmp, 'dash_pattern_comparison.png', fmt='png')
plt.close('all')
print("Saved: dash_pattern_comparison.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Axial-ratio 2-D map (Stokes-parameter method)
# ─────────────────────────────────────────────────────────────────────────────
AR_2d  = axial_ratio_2d(rp24.E_theta, rp24.E_phi)
AR_dB  = 20.0 * np.log10(np.minimum(AR_2d, 100.0))   # clip inf for display
phi_deg = np.rad2deg(rp24.phi)

fig_ar, ax_ar = plt.subplots(figsize=(12, 5.5))
im = ax_ar.pcolormesh(phi_deg, theta_deg, AR_dB,
                       cmap='RdYlGn_r',
                       vmin=0, vmax=20,
                       shading='gouraud', rasterized=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
div = make_axes_locatable(ax_ar)
cax = div.append_axes('right', size='3%', pad=0.08)
cbar = fig_ar.colorbar(im, cax=cax)
cbar.set_label('Axial Ratio (dB)', fontsize=10)

# 3-dB CP boundary (AR < 3 dB ~ good CP)
try:
    cs = ax_ar.contour(phi_deg, theta_deg, AR_dB,
                        levels=[3.0], colors=['white'], linewidths=[2.0])
    ax_ar.clabel(cs, inline=True, fontsize=8, fmt='3 dB CP')
except Exception:
    pass

ax_ar.set_xlabel('phi (degrees)', fontsize=11)
ax_ar.set_ylabel('theta (degrees)', fontsize=11)
ax_ar.set_title('Axial Ratio Map (Stokes params) -- Patch 2.4 GHz FR4',
                 fontsize=12, fontweight='bold', color='#2C3E50')
ax_ar.invert_yaxis()
ax_ar.set_xticks(np.arange(0, 361, 30))
ax_ar.set_yticks(np.arange(0, 181, 15))
fig_ar.tight_layout()
export_fig(fig_ar, 'dash_axial_ratio_map.png', fmt='png')
plt.close('all')
print("Saved: dash_axial_ratio_map.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Waterfall: E-plane pattern vs frequency over the impedance band
# ─────────────────────────────────────────────────────────────────────────────
freqs_z, Z = solver24.impedance_sweep(n_freq=300)
f_ghz_all  = freqs_z / 1e9

# Sample 21 frequency points spanning the impedance band
idx        = np.round(np.linspace(0, len(f_ghz_all) - 1, 21)).astype(int)
f_ghz_sub  = f_ghz_all[idx]

n_theta_wf = 181
theta_wf   = np.linspace(0, 180, n_theta_wf)
pats_wf    = []
for f_ghz in f_ghz_sub:
    solver_f = PatchAnalyticalSolver(patch24, freq=f_ghz * 1e9)
    rp_f     = solver_f.radiation_pattern(n_theta=n_theta_wf, n_phi=1)
    cut      = rp_f.e_plane_cut()
    p_max    = float(np.max(cut)) or 1.0
    pats_wf.append(10 * np.log10(np.clip(cut / p_max, 1e-10, None)))

fig_wf = plot_waterfall(
    freq_ghz    = f_ghz_sub,
    theta_deg   = theta_wf,
    patterns_db = np.array(pats_wf),
    phi_cut_deg = 0.0,
    dyn_range   = 40,
    title       = 'Patch 2.4 GHz FR4 -- E-Plane Gain vs Frequency',
)
export_fig(fig_wf, 'dash_waterfall_e_plane.png', fmt='png')
plt.close('all')
print("Saved: dash_waterfall_e_plane.png  [600 DPI]")


print("\nAll pattern visualization examples complete.")
