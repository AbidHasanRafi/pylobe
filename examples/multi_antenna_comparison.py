"""
Multi-Antenna Comparison and Analysis
=======================================
Shows how to compare multiple antenna designs in one workflow using
the new comparison tools in PyLobe.

Techniques demonstrated
-----------------------
1.  Side-by-side E/H polar and Cartesian overlays (plot_compare_patterns)
2.  Peak-gain bar chart and tabular metrics comparison
3.  Gain heatmap comparison — matplotlib side-by-side at 600 DPI
4.  Frequency-response overlay: S11 of all three antennas
5.  VSWR comparison
6.  Return-loss overlay
7.  Impedance-magnitude overlay
8.  Waterfall (E-plane vs frequency) for each antenna
9.  Axial-ratio profile along E-plane for each antenna
10. Beam-solid-angle summary bar chart

Antennas compared
-----------------
A: Rectangular patch  2.4 GHz  FR4  h=1.6 mm     (WiFi band)
B: Rectangular patch  2.4 GHz  RT5880  h=0.79 mm  (lower-loss substrate)
C: Half-wave dipole   2.4 GHz  copper

All exports: 600 DPI PNG + Plotly HTML where interactive.
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
from pylobe.solver.analytical.patch_solver  import PatchAnalyticalSolver
from pylobe.solver.analytical.dipole_solver import DipoleSolver
from pylobe.analysis.metrics import (
    axial_ratio_2d, beam_solid_angle, beamwidth_hpbw, beamwidth_fnbw,
    side_lobe_level, front_to_back_ratio,
)
from pylobe.visualization.style   import set_style, export_fig
from pylobe.visualization.heatmap import (
    plot_gain_heatmap_mpl, plot_waterfall,
    plot_s11_vs_freq, plot_vswr_vs_freq, plot_return_loss_vs_freq,
)
from pylobe.visualization.dashboard import plot_compare_patterns

set_style('default')

PALETTE = ['#0072B2', '#D55E00', '#009E73']
FREQ    = 2.4e9

# ─────────────────────────────────────────────────────────────────────────────
# Solve all three antennas
# ─────────────────────────────────────────────────────────────────────────────
print("Solving antennas...")

# A — Patch on FR4
pA = RectangularPatch(freq=FREQ, substrate_material=FR4,
                       h=1.6e-3, patch_material=COPPER,
                       ground_material=PEC, inset_feed=True)
sA = PatchAnalyticalSolver(pA, freq=FREQ)
rA = sA.radiation_pattern(n_theta=181, n_phi=73)

# B — Patch on RT5880
pB = RectangularPatch(freq=FREQ, substrate_material=RT5880,
                       h=0.787e-3, patch_material=COPPER,
                       ground_material=PEC, inset_feed=True)
sB = PatchAnalyticalSolver(pB, freq=FREQ)
rB = sB.radiation_pattern(n_theta=181, n_phi=73)

# C — Half-wave dipole
dC  = HalfWaveDipole(freq=FREQ, length_factor=0.47, conductor_material=COPPER)
sC  = DipoleSolver(dC, freq=FREQ)
rC  = sC.radiation_pattern(n_theta=181, n_phi=73)

rp_list = [rA, rB, rC]
labels  = ['Patch FR4 1.6mm', 'Patch RT5880 0.79mm', 'Dipole 0.47lam']
print("Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Comprehensive pattern comparison dashboard
# ─────────────────────────────────────────────────────────────────────────────
fig_cmp = plot_compare_patterns(
    rp_list=rp_list,
    labels=labels,
    dyn_range=40,
    title='Antenna Comparison -- 2.4 GHz',
)
export_fig(fig_cmp, 'cmp_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: cmp_pattern_dashboard.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gain heatmap side-by-side (matplotlib, 600 DPI)
# ─────────────────────────────────────────────────────────────────────────────
fig_hm, axes_hm = plt.subplots(1, 3, figsize=(20, 6))
fig_hm.suptitle('Gain Heatmap Comparison -- 2.4 GHz',
                 fontsize=13, fontweight='bold', color='#2C3E50')
for rp, ax_hm, lbl in zip(rp_list, axes_hm, labels):
    plot_gain_heatmap_mpl(rp, ax=ax_hm, dyn_range=40)
    ax_hm.set_title(lbl, fontsize=11, fontweight='bold', color='#2C3E50', pad=6)
fig_hm.tight_layout()
export_fig(fig_hm, 'cmp_gain_heatmap.png', fmt='png')
plt.close('all')
print("Saved: cmp_gain_heatmap.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Frequency-response overlays (S11, VSWR, return loss, |Z|)
#    — one combined figure, 2x2 subplots
# ─────────────────────────────────────────────────────────────────────────────
print("Computing frequency sweeps...")

# S11 sweeps
freq_A, s11A = sA.s11(Z0=50.0, n_freq=300)
freq_B, s11B = sB.s11(Z0=50.0, n_freq=300)
# Dipole ref impedance ≈ 73.1 Ohm, use 50 Ohm port for comparison
freq_C = np.linspace(2.0e9, 2.8e9, 300)
Zin_C  = np.array([DipoleSolver(HalfWaveDipole(freq=f, length_factor=0.47),
                                  freq=f).input_impedance()
                    for f in freq_C])
s11C   = (Zin_C - 50.0) / (Zin_C + 50.0)

s11A_db = 20 * np.log10(np.abs(s11A) + 1e-20)
s11B_db = 20 * np.log10(np.abs(s11B) + 1e-20)
s11C_db = 20 * np.log10(np.abs(s11C) + 1e-20)
vswr_A  = (1 + np.abs(s11A)) / np.clip(1 - np.abs(s11A), 1e-6, None)
vswr_B  = (1 + np.abs(s11B)) / np.clip(1 - np.abs(s11B), 1e-6, None)
vswr_C  = (1 + np.abs(s11C)) / np.clip(1 - np.abs(s11C), 1e-6, None)

# Impedance
_, ZA = sA.impedance_sweep(n_freq=300)
_, ZB = sB.impedance_sweep(n_freq=300)
ZC    = Zin_C

fig_freq, axes_f = plt.subplots(2, 2, figsize=(16, 11))
fig_freq.suptitle('Frequency Response Comparison -- 2.4 GHz',
                   fontsize=13, fontweight='bold', color='#2C3E50')

ax_s11, ax_vswr, ax_rl, ax_zmag = (axes_f[0, 0], axes_f[0, 1],
                                    axes_f[1, 0], axes_f[1, 1])

for (fq, s11_db, lbl, col) in [
    (freq_A, s11A_db, labels[0], PALETTE[0]),
    (freq_B, s11B_db, labels[1], PALETTE[1]),
    (freq_C, s11C_db, labels[2], PALETTE[2]),
]:
    fghz = fq / 1e9
    ax_s11.plot(fghz, s11_db, color=col, lw=2.0, label=lbl)
    ax_rl.plot(fghz, -s11_db, color=col, lw=2.0, label=lbl)

for (fq, vswr, lbl, col) in [
    (freq_A, vswr_A, labels[0], PALETTE[0]),
    (freq_B, vswr_B, labels[1], PALETTE[1]),
    (freq_C, vswr_C, labels[2], PALETTE[2]),
]:
    ax_vswr.plot(fq / 1e9, vswr, color=col, lw=2.0, label=lbl)

for (fq, Z, lbl, col) in [
    (freq_A, ZA, labels[0], PALETTE[0]),
    (freq_B, ZB, labels[1], PALETTE[1]),
    (freq_C, ZC, labels[2], PALETTE[2]),
]:
    ax_zmag.plot(fq / 1e9, np.abs(Z), color=col, lw=2.0, label=lbl)

# Style
ax_s11.axhline(-10, color='#E74C3C', ls='--', lw=1.1, alpha=0.8)
ax_s11.set_ylim([min(s11A_db.min(), s11B_db.min(), s11C_db.min()) - 4, 2])
ax_s11.set_title('|S11| vs Frequency', fontsize=11, fontweight='bold', color='#2C3E50')
ax_s11.set_ylabel('|S11| (dB)', fontsize=10)

ax_vswr.axhline(2.0, color='#E74C3C', ls='--', lw=1.1, alpha=0.8)
ax_vswr.set_ylim([1.0, 10.0])
ax_vswr.set_title('VSWR vs Frequency', fontsize=11, fontweight='bold', color='#2C3E50')
ax_vswr.set_ylabel('VSWR', fontsize=10)

ax_rl.axhline(10.0, color='#27AE60', ls='--', lw=1.1, alpha=0.8)
ax_rl.set_ylim(bottom=0)
ax_rl.set_title('Return Loss vs Frequency', fontsize=11, fontweight='bold', color='#2C3E50')
ax_rl.set_ylabel('Return Loss (dB)', fontsize=10)

ax_zmag.axhline(50.0, color='#27AE60', ls=':', lw=1.0, alpha=0.7, label='Z0=50 Ohm')
ax_zmag.set_title('|Z| vs Frequency', fontsize=11, fontweight='bold', color='#2C3E50')
ax_zmag.set_ylabel('|Z| (Ohm)', fontsize=10)

for ax in axes_f.flat:
    ax.set_xlabel('Frequency (GHz)', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.tick_params(labelsize=9)
    ax.set_xlim([2.0, 2.8])

fig_freq.tight_layout()
export_fig(fig_freq, 'cmp_frequency_response.png', fmt='png')
plt.close('all')
print("Saved: cmp_frequency_response.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Waterfall comparison — E-plane vs frequency for each antenna
# ─────────────────────────────────────────────────────────────────────────────
print("Computing waterfalls...")

def _compute_waterfall(solver, patch_or_dipole, freq_ghz_list, n_theta=181):
    """Compute E-plane gain waterfall for a sequence of frequencies."""
    theta_deg = np.linspace(0, 180, n_theta)
    pats = []
    for f_ghz in freq_ghz_list:
        try:
            if isinstance(patch_or_dipole, RectangularPatch):
                sv = PatchAnalyticalSolver(patch_or_dipole, freq=f_ghz * 1e9)
            else:
                dip_f = HalfWaveDipole(freq=f_ghz * 1e9, length_factor=0.47)
                sv    = DipoleSolver(dip_f, freq=f_ghz * 1e9)
            rp_f  = sv.radiation_pattern(n_theta=n_theta, n_phi=1)
            cut   = rp_f.e_plane_cut()
        except Exception:
            cut = np.ones(n_theta)
        p_max = float(np.max(cut)) or 1.0
        pats.append(10 * np.log10(np.clip(cut / p_max, 1e-10, None)))
    return theta_deg, np.array(pats)

f_ghz_sweep = np.linspace(2.0, 2.8, 17)

for geom, ant_label, fname_suffix in [
    (pA, labels[0], 'patch_fr4'),
    (pB, labels[1], 'patch_rt5880'),
    (dC, labels[2], 'dipole'),
]:
    theta_wf, pats_wf = _compute_waterfall(None, geom, f_ghz_sweep)
    fig_wf = plot_waterfall(
        freq_ghz    = f_ghz_sweep,
        theta_deg   = theta_wf,
        patterns_db = pats_wf,
        phi_cut_deg = 0.0,
        dyn_range   = 40,
        title       = f'E-Plane Waterfall -- {ant_label}',
    )
    fname = f'cmp_waterfall_{fname_suffix}.png'
    export_fig(fig_wf, fname, fmt='png')
    plt.close('all')
    print(f"Saved: {fname}  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Axial-ratio profile along E-plane (theta scan at phi=0)
# ─────────────────────────────────────────────────────────────────────────────
fig_ar, ax_ar = plt.subplots(figsize=(11, 5.5))
theta_deg = np.rad2deg(rA.theta)

for rp, lbl, col in zip(rp_list, labels, PALETTE):
    # AR along phi=0 cut (E-plane)
    idx_phi0 = int(np.argmin(np.abs(rp.phi)))   # phi closest to 0
    Et_cut   = rp.E_theta[:, idx_phi0]
    Ep_cut   = rp.E_phi[:, idx_phi0]
    AR_cut   = np.array([axial_ratio_2d(
                    Et_cut[i:i+1, np.newaxis],
                    Ep_cut[i:i+1, np.newaxis]
                ).flat[0] for i in range(len(Et_cut))])
    AR_dB = 20.0 * np.log10(np.minimum(AR_cut, 100.0))
    ax_ar.plot(theta_deg, AR_dB, color=col, lw=2.0, label=lbl)

ax_ar.axhline(3.0, color='#27AE60', ls='--', lw=1.2, alpha=0.85,
               label='3 dB CP threshold')
ax_ar.fill_between(theta_deg, 0, 3.0, alpha=0.08, color='#27AE60',
                    label='Good CP region')
ax_ar.set_xlim([0, 180])
ax_ar.set_ylim([0, 25])
ax_ar.set_xlabel('theta (degrees)', fontsize=12)
ax_ar.set_ylabel('Axial Ratio (dB)', fontsize=12)
ax_ar.set_title('E-Plane Axial Ratio Profile (Stokes params) -- 2.4 GHz',
                 fontsize=13, fontweight='bold', color='#2C3E50')
ax_ar.legend(framealpha=0.92, fontsize=10, loc='upper right')
ax_ar.grid(True, alpha=0.35, linestyle='--')
fig_ar.tight_layout()
export_fig(fig_ar, 'cmp_axial_ratio_profile.png', fmt='png')
plt.close('all')
print("Saved: cmp_axial_ratio_profile.png  [600 DPI]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metrics summary — beam solid angle bar chart + printed table
# ─────────────────────────────────────────────────────────────────────────────
print("\nMetrics summary:")

metrics = []
for rp, lbl in zip(rp_list, labels):
    sm    = rp.summary()
    omega = beam_solid_angle(rp.E_theta, rp.E_phi, rp.theta, rp.phi)
    e_lin = rp.e_plane_cut()
    h_lin = rp.h_plane_cut()
    th_d  = np.rad2deg(rp.theta)
    fnbw_e = beamwidth_fnbw(e_lin, th_d)
    fnbw_h = beamwidth_fnbw(h_lin, th_d)
    metrics.append({
        'label':       lbl,
        'gain_dbi':    sm.peak_gain_dbi,
        'hpbw_e':      sm.hpbw_e_deg,
        'hpbw_h':      sm.hpbw_h_deg,
        'fnbw_e':      fnbw_e,
        'sll_db':      sm.sll_db,
        'fbr_db':      sm.fbr_db,
        'beam_sr':     omega,
    })
    print(f"  {lbl:<25}  gain={sm.peak_gain_dbi:.2f} dBi  "
          f"HPBW-E={sm.hpbw_e_deg:.1f}  FNBW-E={fnbw_e:.1f}  "
          f"SLL={sm.sll_db:.1f}  FBR={sm.fbr_db:.1f}  "
          f"Omega_A={omega:.4f} sr")

# Bar chart for beam solid angle
fig_bsa, ax_bsa = plt.subplots(figsize=(9, 5))
x     = np.arange(len(metrics))
bsas  = [m['beam_sr'] for m in metrics]
bars  = ax_bsa.bar(x, bsas, color=PALETTE[:len(metrics)],
                    edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
for bar, val in zip(bars, bsas):
    ax_bsa.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f} sr', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#2C3E50')
ax_bsa.set_xticks(x)
ax_bsa.set_xticklabels([m['label'] for m in metrics],
                        rotation=15, ha='right', fontsize=10)
ax_bsa.set_ylabel('Beam Solid Angle (sr)', fontsize=12)
ax_bsa.set_title('Beam Solid Angle Comparison -- 2.4 GHz',
                  fontsize=13, fontweight='bold', color='#2C3E50')
ax_bsa.grid(True, axis='y', alpha=0.35, linestyle='--')
fig_bsa.tight_layout()
export_fig(fig_bsa, 'cmp_beam_solid_angle.png', fmt='png')
plt.close('all')
print("\nSaved: cmp_beam_solid_angle.png  [600 DPI]")

# Peak gain comparison bar chart
fig_gain, ax_gain = plt.subplots(figsize=(9, 5))
gains = [m['gain_dbi'] for m in metrics]
bars2 = ax_gain.bar(x, gains, color=PALETTE[:len(metrics)],
                     edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
for bar, val in zip(bars2, gains):
    ax_gain.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.05,
                  f'{val:.2f} dBi', ha='center', va='bottom',
                  fontsize=9.5, fontweight='bold', color='#2C3E50')
ax_gain.set_xticks(x)
ax_gain.set_xticklabels([m['label'] for m in metrics],
                          rotation=15, ha='right', fontsize=10)
ax_gain.set_ylabel('Peak Gain (dBi)', fontsize=12)
ax_gain.set_title('Peak Gain Comparison -- 2.4 GHz',
                   fontsize=13, fontweight='bold', color='#2C3E50')
ax_gain.grid(True, axis='y', alpha=0.35, linestyle='--')
fig_gain.tight_layout()
export_fig(fig_gain, 'cmp_peak_gain.png', fmt='png')
plt.close('all')
print("Saved: cmp_peak_gain.png  [600 DPI]")

print("\nAll multi-antenna comparison examples complete.")
