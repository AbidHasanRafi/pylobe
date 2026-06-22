"""
Full-wave FDTD simulation of a rectangular patch antenna at 2.4 GHz.

Demonstrates:
  - Auto-sized Yee grid with UPML absorbing boundaries
  - Gaussian pulse excitation
  - Time-domain probe -> S11 via FFT
  - Near-to-far field transformation -> 3-D pattern
  - S11 and VSWR vs frequency plots
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylobe.solver.fdtd import FDTDSimulation
from pylobe.solver.fdtd.sources import modulated_gaussian
from pylobe.geometry.patch import RectangularPatch
from pylobe.visualization.heatmap import plot_s11_vs_freq, plot_vswr_vs_freq
from pylobe.visualization.lobe3d import plot_3d_radiation
from pylobe.visualization.style import export_fig

print("=" * 60)
print("  FDTD Simulation -- 2.4 GHz Rectangular Patch")
print("=" * 60)

FREQ      = 2.4e9
BANDWIDTH = 2e9

# -- Patch geometry -----------------------------------------------------------
patch = RectangularPatch(freq=FREQ, eps_r=4.4, h=1.6e-3)
print(f"\nPatch: W={patch.W*1e3:.3f} mm, L={patch.L*1e3:.3f} mm")

# -- FDTD simulation setup ----------------------------------------------------
sim = FDTDSimulation(
    freq_center=FREQ,
    freq_span=BANDWIDTH,
    cells_per_wavelength=15,
    pml_cells=10,
)
sim.add_geometry(patch)

# Modulated Gaussian source centred at FREQ
t0       = 5.0 / BANDWIDTH          # pulse delay [s]
waveform = lambda t: modulated_gaussian(t, FREQ, t0, BANDWIDTH)
sim.add_source('soft', tuple(patch.feed_point), waveform)

print(f"\nGrid: {sim.grid.Nx}x{sim.grid.Ny}x{sim.grid.Nz} cells")
print(f"dt   = {sim.dt*1e12:.4f} ps")

print("\nRunning FDTD (auto n_steps)...")
sim.run(verbose=True)

# -- S-parameters -------------------------------------------------------------
freqs, S11 = sim.get_s_parameters()
s11_db     = 20 * np.log10(np.abs(S11) + 1e-20)

print(f"\nMinimum S11 = {np.min(s11_db):.1f} dB at {freqs[np.argmin(s11_db)]/1e9:.3f} GHz")

fig_s11 = plot_s11_vs_freq(freqs, s11_db)
export_fig(fig_s11, 'fdtd_patch_s11.png', fmt='png')
print("Saved: fdtd_patch_s11.png")
plt.close('all')

vswr = (1 + np.abs(S11)) / (1 - np.abs(S11) + 1e-12)
fig_vswr = plot_vswr_vs_freq(freqs, vswr)
export_fig(fig_vswr, 'fdtd_patch_vswr.png', fmt='png')
print("Saved: fdtd_patch_vswr.png")
plt.close('all')

# -- 3-D radiation pattern ----------------------------------------------------
rp = sim.get_radiation_pattern(freq=FREQ)
if rp is not None:
    fig_3d = plot_3d_radiation(rp)
    fig_3d.write_html('fdtd_patch_3d.html')
    print("Saved: fdtd_patch_3d.html")

sim.save('fdtd_patch_result.pkl')
print("Saved: fdtd_patch_result.pkl")
print("\nDone.")
