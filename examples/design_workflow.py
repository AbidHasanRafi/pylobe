"""
Complete AntennaDesign Workflow
================================
End-to-end demonstration of the high-level ``AntennaDesign`` API —
the fastest path from geometry to fully-documented results.

What this example covers
------------------------
1. Creating three antennas: patch (FR4), patch (RT5880), half-wave dipole
2. Solving with AntennaDesign.solve()
3. Reading the rich text summary (gain, HPBW, FNBW, SLL, FBR,
   beam solid angle, axial ratio, S-parameters, VSWR, bandwidth)
4. Generating comprehensive 2-D pattern dashboards (plot_2d_patterns)
5. Generating frequency-response dashboards (plot_frequency_response)
6. One-line combined dashboard (plot_dashboard)
7. Interactive Plotly visuals: 3-D pattern, Smith chart, gain heatmap
8. Waterfall plot: how the pattern changes vs frequency
9. Bulk export at 600 DPI with export_all()

All figures that go to disk are exported at >= 600 DPI.

Output files (prefix = 'workflow_')
-------------------------------------
workflow_patch24_pattern_dashboard.png   -- 4x3 panel, 600 DPI
workflow_patch24_freq_dashboard.png      -- 3x3 panel, 600 DPI
workflow_patch24_pattern_3d.html         -- Plotly interactive
workflow_patch24_smith.html              -- Plotly interactive
workflow_patch24_gain_heatmap.html       -- Plotly interactive
workflow_patch58_pattern_dashboard.png
workflow_patch58_freq_dashboard.png
workflow_dipole_pattern_dashboard.png
workflow_waterfall_patch24.png           -- freq x angle waterfall
workflow_waterfall_dipole.png
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylobe.geometry.patch   import RectangularPatch
from pylobe.geometry.dipole  import HalfWaveDipole
from pylobe.geometry.base    import FR4, RT5880, COPPER, PEC
from pylobe.design           import AntennaDesign
from pylobe.visualization.style import set_style, export_fig

set_style('default')

# ─────────────────────────────────────────────────────────────────────────────
# Helper — print a section banner
# ─────────────────────────────────────────────────────────────────────────────
def banner(text):
    print("\n" + "=" * 62)
    print(f"  {text}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# ANTENNA 1 — Rectangular patch at 2.4 GHz on FR4
# ─────────────────────────────────────────────────────────────────────────────
banner("ANTENNA 1: Rectangular Patch 2.4 GHz on FR4")

patch24 = RectangularPatch(
    freq=2.4e9,
    substrate_material=FR4,
    h=1.6e-3,
    patch_material=COPPER,
    ground_material=PEC,
    inset_feed=True,
)

design24 = AntennaDesign(patch24, freq=2.4e9)
design24.solve(n_theta=181, n_phi=73, n_freq=300)

# Rich text summary — includes FNBW, beam solid angle, axial ratio, bandwidth
print(design24.summary())

# 2-D pattern dashboard: polar + cartesian + phase + heatmap + multi-cut + table
fig_pat24 = design24.plot_2d_patterns(dyn_range=40)
export_fig(fig_pat24, 'workflow_patch24_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: workflow_patch24_pattern_dashboard.png  [600 DPI]")

# Frequency response: S11, VSWR, RL, R, X, |Z|/angle, group delay, S-table
fig_freq24 = design24.plot_frequency_response(Z0=50.0)
export_fig(fig_freq24, 'workflow_patch24_freq_dashboard.png', fmt='png')
plt.close('all')
print("Saved: workflow_patch24_freq_dashboard.png  [600 DPI]")

# 3-D radiation pattern with E/H cuts and iso-gain rings
fig_3d_24 = design24.plot_pattern(plane='3d')
fig_3d_24.write_html('workflow_patch24_pattern_3d.html')
print("Saved: workflow_patch24_pattern_3d.html")

# Smith chart — impedance trajectory with VSWR circles and constant-Q loci
fig_sm_24 = design24.plot_smith()
fig_sm_24.write_html('workflow_patch24_smith.html')
print("Saved: workflow_patch24_smith.html")

# Interactive gain heatmap (Plotly)
fig_hm_24 = design24.plot_gain_heatmap(interactive=True)
fig_hm_24.write_html('workflow_patch24_gain_heatmap.html')
print("Saved: workflow_patch24_gain_heatmap.html")

# Static gain heatmap at 600 DPI (matplotlib, embeds well in reports)
fig_hm_mpl24 = design24.plot_gain_heatmap(interactive=False)
export_fig(fig_hm_mpl24, 'workflow_patch24_gain_heatmap_600dpi.png', fmt='png')
plt.close('all')
print("Saved: workflow_patch24_gain_heatmap_600dpi.png  [600 DPI]")

# Waterfall: E-plane pattern vs frequency across the impedance band
fig_wf24 = design24.plot_waterfall(phi_cut_deg=0.0)
export_fig(fig_wf24, 'workflow_waterfall_patch24.png', fmt='png')
plt.close('all')
print("Saved: workflow_waterfall_patch24.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# ANTENNA 2 — Rectangular patch at 5.8 GHz on RT/duroid 5880
# ─────────────────────────────────────────────────────────────────────────────
banner("ANTENNA 2: Rectangular Patch 5.8 GHz on RT/duroid5880")

patch58 = RectangularPatch(
    freq=5.8e9,
    substrate_material=RT5880,   # eps_r=2.2, tan_d=0.0009
    h=0.787e-3,
    patch_material=COPPER,
    ground_material=PEC,
    inset_feed=True,
)

design58 = AntennaDesign(patch58, freq=5.8e9)
design58.solve(n_theta=181, n_phi=73, n_freq=300)
print(design58.summary())

fig_pat58 = design58.plot_2d_patterns(dyn_range=40)
export_fig(fig_pat58, 'workflow_patch58_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: workflow_patch58_pattern_dashboard.png  [600 DPI]")

fig_freq58 = design58.plot_frequency_response(Z0=50.0)
export_fig(fig_freq58, 'workflow_patch58_freq_dashboard.png', fmt='png')
plt.close('all')
print("Saved: workflow_patch58_freq_dashboard.png  [600 DPI]")

fig_sm_58 = design58.plot_smith()
fig_sm_58.write_html('workflow_patch58_smith.html')
print("Saved: workflow_patch58_smith.html")


# ─────────────────────────────────────────────────────────────────────────────
# ANTENNA 3 — Half-wave dipole at 300 MHz
# ─────────────────────────────────────────────────────────────────────────────
banner("ANTENNA 3: Half-Wave Dipole 300 MHz")

dipole = HalfWaveDipole(freq=300e6, length_factor=0.47, conductor_material=COPPER)
design_dip = AntennaDesign(dipole, freq=300e6)
design_dip.solve(n_theta=181, n_phi=73, n_freq=200)
print(design_dip.summary())

fig_pat_dip = design_dip.plot_2d_patterns(dyn_range=40)
export_fig(fig_pat_dip, 'workflow_dipole_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: workflow_dipole_pattern_dashboard.png  [600 DPI]")

fig_freq_dip = design_dip.plot_frequency_response(Z0=73.1)
if fig_freq_dip:
    export_fig(fig_freq_dip, 'workflow_dipole_freq_dashboard.png', fmt='png')
    plt.close('all')
    print("Saved: workflow_dipole_freq_dashboard.png  [600 DPI]")

fig_wf_dip = design_dip.plot_waterfall(phi_cut_deg=0.0)
export_fig(fig_wf_dip, 'workflow_waterfall_dipole.png', fmt='png')
plt.close('all')
print("Saved: workflow_waterfall_dipole.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# Bulk export — export_all() in one call
# ─────────────────────────────────────────────────────────────────────────────
banner("Bulk export: export_all()")

print("\nExporting all plots for Patch 2.4 GHz...")
exported24 = design24.export_all(
    prefix='workflow_bulk_patch24',
    fmt='png',
    dpi=600,
    include_interactive=True,
)
print(f"  {len(exported24)} files written:")
for f in exported24:
    print(f"    {f}")

print("\nExporting all plots for Dipole 300 MHz...")
exported_dip = design_dip.export_all(
    prefix='workflow_bulk_dipole',
    fmt='png',
    dpi=600,
    include_interactive=True,
)
print(f"  {len(exported_dip)} files written:")
for f in exported_dip:
    print(f"    {f}")

banner("Done")
