"""
Rectangular patch antenna complete design flow at 2.4 GHz on FR4.

Demonstrates:
  - Balanis cavity-model design (W, L, eps_eff, feed inset y0)
  - AntennaDesign high-level workflow object
  - Comprehensive 2-D pattern dashboard  (plot_2d_patterns)
  - Comprehensive frequency-response dashboard  (plot_frequency_response)
  - Combined design dashboard  (plot_dashboard)
  - 3-D interactive radiation pattern (Plotly)
  - Interactive Smith chart with VSWR and constant-Q loci
  - Gain heatmap (interactive Plotly + static matplotlib at 600 DPI)
  - All figures exported at 600 DPI via export_all()
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylobe.geometry.patch import RectangularPatch
from pylobe.geometry.base import FR4, RT5880, COPPER, PEC
from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
from pylobe.design import AntennaDesign
from pylobe.visualization.style import set_style, export_fig
from pylobe.visualization.polar import plot_e_h_plane
from pylobe.visualization.lobe3d import plot_3d_radiation
from pylobe.visualization.smith_chart import plot_smith_chart
from pylobe.visualization.heatmap import (
    plot_gain_heatmap, plot_gain_heatmap_mpl,
    plot_s11_vs_freq, plot_vswr_vs_freq, plot_impedance_vs_freq,
    plot_return_loss_vs_freq,
)
from pylobe.visualization.dashboard import (
    plot_pattern_dashboard, plot_frequency_dashboard,
)

FREQ   = 2.4e9
Z_FEED = 50.0

print("=" * 60)
print("  Rectangular Patch Antenna -- 2.4 GHz on FR4")
print("=" * 60)

set_style('default')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometry and analytical design
# ─────────────────────────────────────────────────────────────────────────────
patch = RectangularPatch(
    freq=FREQ,
    substrate_material=FR4,   # eps_r=4.4, tan_d=0.02
    h=1.6e-3,
    patch_material=COPPER,
    ground_material=PEC,
    inset_feed=True,
)

print(f"\nSubstrate : {FR4.name}  eps_r={FR4.eps_r}  tan_d={FR4.loss_tangent}")
print(f"Patch W   : {patch.W*1e3:.3f} mm")
print(f"Patch L   : {patch.L*1e3:.3f} mm")
print(f"eps_eff   : {patch.eps_eff:.4f}")
print(f"dL        : {patch.delta_L*1e3:.4f} mm")
print(f"Inset y0  : {patch.y0*1e3:.3f} mm")
print(f"f_r       : {patch.resonant_frequency/1e9:.4f} GHz")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Solve via low-level solver (shows solver API)
# ─────────────────────────────────────────────────────────────────────────────
solver = PatchAnalyticalSolver(patch, freq=FREQ)
rp     = solver.radiation_pattern(n_theta=181, n_phi=73)
sm     = rp.summary()    # returns PatternSummary dataclass

print(f"\n--- Radiation pattern summary ---")
print(f"Peak gain   : {sm.peak_gain_dbi:.2f} dBi")
print(f"HPBW  E     : {sm.hpbw_e_deg:.1f} deg")
print(f"HPBW  H     : {sm.hpbw_h_deg:.1f} deg")
print(f"SLL         : {sm.sll_db:.1f} dB")
print(f"F/B ratio   : {sm.fbr_db:.1f} dB")

# ─────────────────────────────────────────────────────────────────────────────
# 3. E/H polar pattern (classic, standalone)
# ─────────────────────────────────────────────────────────────────────────────
fig_eh = plot_e_h_plane(rp)
export_fig(fig_eh, 'patch_eh_plane.png', fmt='png')
plt.close('all')
print("\nSaved: patch_eh_plane.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 2-D pattern dashboard (E/H polar + cartesian + heatmap + phase + cuts)
# ─────────────────────────────────────────────────────────────────────────────
fig_pat_dash = plot_pattern_dashboard(
    rp,
    dyn_range=40,
    title='2.4 GHz Patch FR4',
)
export_fig(fig_pat_dash, 'patch_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: patch_pattern_dashboard.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 3-D interactive pattern
# ─────────────────────────────────────────────────────────────────────────────
fig_3d = plot_3d_radiation(rp, show_e_h_cuts=True, show_isogain_rings=True)
fig_3d.write_html('patch_3d_pattern.html')
print("Saved: patch_3d_pattern.html")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Gain heatmap — interactive Plotly + static matplotlib
# ─────────────────────────────────────────────────────────────────────────────
fig_hm_plotly = plot_gain_heatmap(rp, title='Patch 2.4 GHz — Gain Heatmap')
fig_hm_plotly.write_html('patch_gain_heatmap.html')
print("Saved: patch_gain_heatmap.html")

fig_hm_mpl = plot_gain_heatmap_mpl(rp, dyn_range=40)
export_fig(fig_hm_mpl, 'patch_gain_heatmap_600dpi.png', fmt='png')
plt.close('all')
print("Saved: patch_gain_heatmap_600dpi.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Impedance & S-parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────
freqs, s11_complex = solver.s11(Z0=Z_FEED, n_freq=300)
freqs_z, Z         = solver.impedance_sweep(n_freq=300)
s11_db             = 20 * np.log10(np.abs(s11_complex) + 1e-20)

fig_s11 = plot_s11_vs_freq(freqs, s11_db,
                            bandwidth_threshold=-10,
                            title='Patch 2.4 GHz — |S11|')
export_fig(fig_s11, 'patch_s11.png', fmt='png')
plt.close('all')
print("Saved: patch_s11.png  [600 DPI]")

vswr = (1 + np.abs(s11_complex)) / np.clip(1 - np.abs(s11_complex), 1e-6, None)
fig_vswr = plot_vswr_vs_freq(freqs, vswr, threshold=2.0)
export_fig(fig_vswr, 'patch_vswr.png', fmt='png')
plt.close('all')
print("Saved: patch_vswr.png  [600 DPI]")

fig_rl = plot_return_loss_vs_freq(freqs, s11_db,
                                   title='Patch 2.4 GHz — Return Loss')
export_fig(fig_rl, 'patch_return_loss.png', fmt='png')
plt.close('all')
print("Saved: patch_return_loss.png  [600 DPI]")

fig_Z = plot_impedance_vs_freq(freqs_z, Z,
                                title='Patch 2.4 GHz — Input Impedance')
export_fig(fig_Z, 'patch_impedance.png', fmt='png')
plt.close('all')
print("Saved: patch_impedance.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Frequency dashboard (S11 + VSWR + RL + R + X + |Z|/angle + GD + table)
# ─────────────────────────────────────────────────────────────────────────────
fig_freq_dash = plot_frequency_dashboard(
    freqs, s11_complex, Z,
    Z0=Z_FEED,
    title='2.4 GHz Patch FR4',
)
export_fig(fig_freq_dash, 'patch_frequency_dashboard.png', fmt='png')
plt.close('all')
print("Saved: patch_frequency_dashboard.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Interactive Smith chart
# ─────────────────────────────────────────────────────────────────────────────
fig_sm = plot_smith_chart(
    impedance_trace=Z,
    freq_labels=freqs_z,
    title='Patch 2.4 GHz — Smith Chart',
)
fig_sm.write_html('patch_smith.html')
print("Saved: patch_smith.html")

# ─────────────────────────────────────────────────────────────────────────────
# 10. AntennaDesign workflow — high-level API
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- AntennaDesign high-level API ---")
design = AntennaDesign(patch, freq=FREQ)
design.solve(n_theta=181, n_phi=73, n_freq=300)

print(design.summary())

# One-line dashboards via design object
fig_d_pat, fig_d_freq = design.plot_dashboard()
export_fig(fig_d_pat,  'patch_design_pattern_dash.png',  fmt='png')
export_fig(fig_d_freq, 'patch_design_freq_dash.png',     fmt='png')
plt.close('all')
print("Saved: patch_design_pattern_dash.png  [600 DPI]")
print("Saved: patch_design_freq_dash.png     [600 DPI]")

# Export everything at once
exported = design.export_all(
    prefix='patch_export',
    fmt='png',
    dpi=600,
    include_interactive=True,
)
print(f"\nexport_all() wrote {len(exported)} files:")
for f in exported:
    print(f"  {f}")

print("\nDone.")
