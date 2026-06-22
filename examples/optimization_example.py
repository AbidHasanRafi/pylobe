"""
Multi-algorithm optimization of a 5.8 GHz patch antenna.

Demonstrates:
  - Genetic Algorithm (SBX + polynomial mutation)
  - Particle Swarm Optimizer
  - Differential Evolution
  - Bayesian Optimization (Matern GP + EI)
  - Convergence curve comparison
  - Best design re-run with full 2-D dashboard and export at 600 DPI
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
from pylobe.optimization.genetic import GeneticAlgorithm
from pylobe.optimization.pso import ParticleSwarmOptimizer
from pylobe.optimization.differential_evolution import DifferentialEvolution
from pylobe.optimization.bayesian import BayesianOptimizer
from pylobe.design import AntennaDesign
from pylobe.visualization.style import set_style, export_fig
from pylobe.visualization.polar import plot_e_h_plane
from pylobe.visualization.lobe3d import plot_3d_radiation
from pylobe.visualization.dashboard import (
    plot_pattern_dashboard, plot_frequency_dashboard,
)

TARGET_FREQ = 5.8e9
Z0          = 50.0

# ─────────────────────────────────────────────────────────────────────────────
# Objective function — minimise −directivity (dBi) at 5.8 GHz
# ─────────────────────────────────────────────────────────────────────────────
def antenna_objective(params: dict) -> float:
    try:
        from pylobe.geometry.base import Material
        sub = Material('opt_sub',
                       eps_r=params['eps_r'],
                       loss_tangent=params.get('tan_d', 0.02))
        patch  = RectangularPatch(
            freq=TARGET_FREQ,
            substrate_material=sub,
            h=params['h'],
            patch_material=COPPER,
            ground_material=PEC,
            inset_feed=True,
        )
        solver = PatchAnalyticalSolver(patch, freq=TARGET_FREQ)
        return -solver.directivity_dbi   # minimise → maximise directivity
    except Exception:
        return 1e9


BOUNDS = {
    'eps_r': (2.0, 10.0),
    'h':     (0.5e-3, 3.0e-3),
}

print("=" * 60)
print("  Multi-Algorithm Patch Optimization -- 5.8 GHz")
print("=" * 60)

set_style('default')
results = {}

# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Genetic Algorithm...")
ga = GeneticAlgorithm(antenna_objective, BOUNDS, pop_size=40, n_generations=60)
results['GA']  = ga.run()
print(f"  Best directivity: {-results['GA'].best_fitness:.3f} dBi"
      f"  params={results['GA'].best_params}")

print("[2/4] Particle Swarm Optimizer...")
pso = ParticleSwarmOptimizer(antenna_objective, BOUNDS,
                              n_particles=40, n_iterations=60)
results['PSO'] = pso.run()
print(f"  Best directivity: {-results['PSO'].best_fitness:.3f} dBi"
      f"  params={results['PSO'].best_params}")

print("[3/4] Differential Evolution...")
de = DifferentialEvolution(antenna_objective, BOUNDS,
                            pop_size=30, n_generations=60)
results['DE']  = de.run()
print(f"  Best directivity: {-results['DE'].best_fitness:.3f} dBi"
      f"  params={results['DE'].best_params}")

print("[4/4] Bayesian Optimizer (GP + EI)...")
bo = BayesianOptimizer(antenna_objective, BOUNDS,
                        n_initial=10, n_iterations=40)
results['BO']  = bo.run()
print(f"  Best directivity: {-results['BO'].best_fitness:.3f} dBi"
      f"  params={results['BO'].best_params}")

# ─────────────────────────────────────────────────────────────────────────────
# Convergence curves
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
fig_conv, ax_conv = plt.subplots(figsize=(10, 5.5))
for (name, res), col in zip(results.items(), PALETTE):
    curve = -res.convergence_curve()   # flip sign -> directivity [dBi]
    ax_conv.plot(curve, lw=2.2, label=name, color=col)
    ax_conv.axhline(curve[-1], color=col, linestyle=':', lw=0.8, alpha=0.55)

ax_conv.set_xlabel('Iteration / Generation', fontsize=12)
ax_conv.set_ylabel('Best Directivity (dBi)', fontsize=12)
ax_conv.set_title('Optimization Convergence — 5.8 GHz Patch',
                  fontsize=13, fontweight='bold', color='#2C3E50')
ax_conv.legend(framealpha=0.92, fontsize=11)
ax_conv.grid(True, alpha=0.35, linestyle='--')
fig_conv.tight_layout()
export_fig(fig_conv, 'opt_convergence.png', fmt='png')
plt.close('all')
print("\nSaved: opt_convergence.png  [600 DPI]")

# ─────────────────────────────────────────────────────────────────────────────
# Best result — full re-run using AntennaDesign
# ─────────────────────────────────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k].best_fitness)
best      = results[best_name]
bp        = best.best_params
print(f"\nGlobal best: {best_name} ->  {-best.best_fitness:.3f} dBi")
print(f"  eps_r = {bp['eps_r']:.4f}")
print(f"  h     = {bp['h']*1e3:.4f} mm")

from pylobe.geometry.base import Material
opt_sub   = Material('opt_sub', eps_r=bp['eps_r'], loss_tangent=0.02)
opt_patch = RectangularPatch(
    freq=TARGET_FREQ,
    substrate_material=opt_sub,
    h=bp['h'],
    patch_material=COPPER,
    ground_material=PEC,
    inset_feed=True,
)
design = AntennaDesign(opt_patch, freq=TARGET_FREQ)
design.solve(n_theta=181, n_phi=73, n_freq=300)

print(design.summary())

# Pattern metrics from PatternSummary dataclass
sm = design.pattern_summary()
print(f"\nOptimised antenna at 5.8 GHz:")
print(f"  Peak gain    : {sm.peak_gain_dbi:.2f} dBi")
print(f"  HPBW E       : {sm.hpbw_e_deg:.1f} deg")
print(f"  HPBW H       : {sm.hpbw_h_deg:.1f} deg")
print(f"  SLL          : {sm.sll_db:.1f} dB")
print(f"  F/B ratio    : {sm.fbr_db:.1f} dB")

# E/H plane plot
fig_eh = plot_e_h_plane(design.radiation_pattern)
export_fig(fig_eh, 'opt_patch_eh_plane.png', fmt='png')
plt.close('all')
print("\nSaved: opt_patch_eh_plane.png  [600 DPI]")

# Comprehensive 2-D pattern dashboard
fig_pat_dash = plot_pattern_dashboard(
    design.radiation_pattern,
    dyn_range=40,
    title=f'Opt. Patch 5.8 GHz | eps_r={bp["eps_r"]:.2f}  h={bp["h"]*1e3:.2f}mm',
)
export_fig(fig_pat_dash, 'opt_patch_pattern_dashboard.png', fmt='png')
plt.close('all')
print("Saved: opt_patch_pattern_dashboard.png  [600 DPI]")

# 3-D interactive
fig_3d = plot_3d_radiation(design.radiation_pattern,
                            show_e_h_cuts=True, show_isogain_rings=True)
fig_3d.write_html('opt_patch_3d.html')
print("Saved: opt_patch_3d.html")

# Frequency dashboard
fig_freq_dash = design.plot_frequency_response(Z0=Z0)
if fig_freq_dash:
    export_fig(fig_freq_dash, 'opt_patch_freq_dashboard.png', fmt='png')
    plt.close('all')
    print("Saved: opt_patch_freq_dashboard.png  [600 DPI]")

# Export everything via export_all
exported = design.export_all(
    prefix='opt_patch_export',
    fmt='png',
    dpi=600,
    include_interactive=True,
)
print(f"\nexport_all() wrote {len(exported)} files:")
for f in exported:
    print(f"  {f}")

print("\nDone.")
