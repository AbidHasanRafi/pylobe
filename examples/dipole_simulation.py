"""
Half-wave dipole: analytical solver vs Method of Moments comparison.

Demonstrates:
  - DipoleSolver (Balanis closed-form)
  - WireMoMSolver (Pocklington integral, pulse basis / point matching)
  - Current distribution I(z) — analytical vs MoM
  - Far-field E-plane pattern overlay (polar + cartesian)
  - Input impedance R + jX vs frequency sweep
  - S11 and VSWR vs frequency
  - All plots exported at 600 DPI

Output files
------------
dipole_current.png            — amplitude and phase of current I(z)
dipole_pattern_compare.png    — E-plane pattern: analytical vs MoM
dipole_pattern_cartesian.png  — Cartesian overlay (dB)
dipole_impedance_sweep.png    — R + jX vs frequency (200-400 MHz)
dipole_s11.png                — |S11| vs frequency (73.1 Ohm ref)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylobe.geometry.dipole import HalfWaveDipole
from pylobe.solver.analytical.dipole_solver import DipoleSolver
from pylobe.solver.mom.wire import WireMoMSolver
from pylobe.visualization.polar import plot_polar_compare, plot_polar
from pylobe.visualization.cartesian import plot_pattern_cartesian
from pylobe.visualization.current_plot import plot_current_1d
from pylobe.visualization.heatmap import plot_s11_vs_freq, plot_vswr_vs_freq
from pylobe.visualization.style import set_style, export_fig
from pylobe.constants import C0, PI

FREQ = 300e6   # 300 MHz

print("=" * 60)
print("  Half-Wave Dipole: Analytical vs MoM")
print("=" * 60)

set_style('default')

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
dipole = HalfWaveDipole(freq=FREQ, length_factor=0.47, N_segments=21)
lam    = C0 / FREQ
print(f"\nf = {FREQ/1e6:.0f} MHz   lam = {lam*1e2:.2f} cm   "
      f"L = {dipole.L_total*1e2:.2f} cm  ({dipole.length_factor} lambda)")

# ─────────────────────────────────────────────────────────────────────────────
# Analytical solver
# ─────────────────────────────────────────────────────────────────────────────
an_solver = DipoleSolver(dipole, freq=FREQ)
Zan       = an_solver.input_impedance()
print(f"\nAnalytical:")
print(f"  Directivity  = {an_solver.directivity:.4f} ({an_solver.directivity_dbi:.2f} dBi)")
print(f"  Rr           = {an_solver.radiation_resistance:.2f} Ohm")
print(f"  Zin          = {Zan.real:.1f} + j{Zan.imag:.1f} Ohm")

# ─────────────────────────────────────────────────────────────────────────────
# MoM solver
# ─────────────────────────────────────────────────────────────────────────────
mom_solver = WireMoMSolver(dipole, freq=FREQ, N_segments=21)
mom_solver.solve()
Zmom = mom_solver.input_impedance()
print(f"\nMoM (N=21):")
print(f"  Zin          = {Zmom.real:.1f} + j{Zmom.imag:.1f} Ohm")

# ─────────────────────────────────────────────────────────────────────────────
# Current distribution — MoM (plot_current_1d handles the figure internally)
# ─────────────────────────────────────────────────────────────────────────────
z_seg  = np.linspace(-dipole.L_total / 2, dipole.L_total / 2, mom_solver.N)
I_mom  = mom_solver._currents

fig_I  = plot_current_1d(z_seg, I_mom, freq=FREQ)

# Add the analytical sinusoidal envelope on the amplitude axes (axes[0])
I_analytic = np.cos(PI * z_seg / dipole.L_total)
I_analytic = I_analytic / np.max(np.abs(I_analytic))   # normalise to 1
z_mm       = z_seg * 1e3

ax_amp = fig_I.axes[0]   # amplitude panel
ax_amp.plot(z_mm, np.abs(I_analytic),
            'k--', linewidth=1.5, label='Analytic sinusoid')
ax_amp.legend(fontsize=9, loc='upper right')

export_fig(fig_I, 'dipole_current.png', fmt='png')
plt.close('all')
print("\nSaved: dipole_current.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# Far-field pattern comparison — polar
# ─────────────────────────────────────────────────────────────────────────────
theta = np.linspace(1e-4, PI - 1e-4, 361)

F_an  = an_solver.element_factor(theta)
F_an  = F_an / np.max(F_an)

E_theta, _ = mom_solver.far_field(theta, phi=np.array([0.0]))
F_mom_cut  = np.abs(E_theta[:, 0])
F_mom_n    = F_mom_cut / (np.max(F_mom_cut) or 1.0)

theta_deg_half = np.rad2deg(theta)
theta_360      = np.concatenate([theta_deg_half, theta_deg_half + 180.0])
F_an_360       = np.concatenate([F_an,  F_an[::-1]])
F_mom_360      = np.concatenate([F_mom_n, F_mom_n[::-1]])

fig_polar = plot_polar_compare(
    [theta_360, theta_360],
    [F_an_360, F_mom_360],
    labels=['Analytical (Balanis)', f'MoM N={mom_solver.N}'],
    title='Half-Wave Dipole E-Plane Pattern Comparison',
    dyn_range=40,
)
export_fig(fig_polar, 'dipole_pattern_compare.png', fmt='png')
plt.close('all')
print("Saved: dipole_pattern_compare.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# Far-field comparison — Cartesian (dB)
# ─────────────────────────────────────────────────────────────────────────────
F_an_db  = 20 * np.log10(np.clip(F_an,   1e-10, None))
F_mom_db = 20 * np.log10(np.clip(F_mom_n, 1e-10, None))

fig_cart, ax_cart = plt.subplots(figsize=(10, 5.2))
ax_cart.plot(theta_deg_half, F_an_db,  color='#0072B2', lw=2.0,
             label='Analytical (Balanis)')
ax_cart.plot(theta_deg_half, F_mom_db, color='#D55E00', lw=1.8,
             linestyle='--', label=f'MoM N={mom_solver.N}')
ax_cart.axhline(-3,  color='#C0392B', linestyle=':', linewidth=1.0, alpha=0.8)
ax_cart.axhline(-10, color='#E67E22', linestyle=':', linewidth=1.0, alpha=0.8)
ax_cart.set_xlim([0, 180])
ax_cart.set_ylim([-42, 2])
ax_cart.set_xlabel('theta (degrees)', fontsize=12)
ax_cart.set_ylabel('Normalized Gain (dB)', fontsize=12)
ax_cart.set_title('E-Plane Pattern — Analytical vs MoM (Cartesian)',
                  fontsize=13, fontweight='bold', color='#2C3E50')
ax_cart.legend(framealpha=0.92, fontsize=10)
ax_cart.grid(True, alpha=0.35, linestyle='--')
fig_cart.tight_layout()
export_fig(fig_cart, 'dipole_pattern_cartesian.png', fmt='png')
plt.close('all')
print("Saved: dipole_pattern_cartesian.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# Impedance sweep  200 – 400 MHz
# ─────────────────────────────────────────────────────────────────────────────
freqs_sweep = np.linspace(200e6, 400e6, 101)
R_an, X_an  = [], []
for f in freqs_sweep:
    dip_f = HalfWaveDipole(freq=f, length_factor=0.47)
    slv_f = DipoleSolver(dip_f, freq=f)
    Zf    = slv_f.input_impedance()
    R_an.append(Zf.real)
    X_an.append(Zf.imag)
R_an = np.array(R_an)
X_an = np.array(X_an)
freqs_mhz = freqs_sweep / 1e6

fig_Z, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
ax1.plot(freqs_mhz, R_an, color='#0072B2', lw=2.2, label='R (Ohm)')
ax1.axhline(73.1, color='#27AE60', linestyle='--', lw=1.2, alpha=0.8,
            label='Self-resonance ~73 Ohm')
ax1.axvline(FREQ / 1e6, color='#95A5A6', linestyle=':', lw=0.9)
ax1.set_xlabel('Frequency (MHz)', fontsize=11)
ax1.set_ylabel('Resistance Ra (Ohm)', fontsize=11)
ax1.set_title('Input Resistance vs Frequency',
              fontsize=12, fontweight='bold', color='#2C3E50')
ax1.legend(fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.35, linestyle='--')

ax2.plot(freqs_mhz, X_an, color='#D55E00', lw=2.2, label='X (Ohm)')
ax2.axhline(0, color='k', lw=0.7, linestyle=':')
ax2.axvline(FREQ / 1e6, color='#95A5A6', linestyle=':', lw=0.9)
ax2.fill_between(freqs_mhz, X_an, 0, where=(X_an >= 0),
                 alpha=0.10, color='#0072B2', interpolate=True, label='Inductive')
ax2.fill_between(freqs_mhz, X_an, 0, where=(X_an < 0),
                 alpha=0.10, color='#D55E00', interpolate=True, label='Capacitive')
ax2.set_xlabel('Frequency (MHz)', fontsize=11)
ax2.set_ylabel('Reactance Xa (Ohm)', fontsize=11)
ax2.set_title('Input Reactance vs Frequency',
              fontsize=12, fontweight='bold', color='#2C3E50')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.35, linestyle='--')

fig_Z.tight_layout()
export_fig(fig_Z, 'dipole_impedance_sweep.png', fmt='png')
plt.close('all')
print("Saved: dipole_impedance_sweep.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# S11 and VSWR at the analytical self-resonance (73.1 Ohm ref)
# ─────────────────────────────────────────────────────────────────────────────
Z0_dipole = 73.1
s11_cplx   = (np.array(R_an) + 1j * np.array(X_an) - Z0_dipole) / \
             (np.array(R_an) + 1j * np.array(X_an) + Z0_dipole)
s11_db     = 20 * np.log10(np.abs(s11_cplx) + 1e-20)
vswr_d     = (1 + np.abs(s11_cplx)) / np.clip(1 - np.abs(s11_cplx), 1e-6, None)

fig_s11d = plot_s11_vs_freq(freqs_sweep, s11_db,
                             bandwidth_threshold=-10,
                             title=f'Dipole 300 MHz -- |S11| (ref {Z0_dipole} Ohm)')
export_fig(fig_s11d, 'dipole_s11.png', fmt='png')
plt.close('all')
print("Saved: dipole_s11.png  [600 DPI]")

fig_vswr_d = plot_vswr_vs_freq(freqs_sweep, vswr_d, threshold=2.0)
export_fig(fig_vswr_d, 'dipole_vswr.png', fmt='png')
plt.close('all')
print("Saved: dipole_vswr.png  [600 DPI]")

print("\nDone.")
