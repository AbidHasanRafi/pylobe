"""
Yagi-Uda inspired linear array with Dolph-Chebyshev amplitude weights.

Demonstrates:
  - LinearArray with Chebyshev weights (SLL control)
  - Beam steering animation
  - Grating lobe boundary check
  - Gain vs scan angle
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylobe.geometry.dipole import HalfWaveDipole
from pylobe.geometry.array import LinearArray
from pylobe.solver.analytical.array_factor import grating_lobe_condition
from pylobe.visualization.polar import plot_polar_compare
from pylobe.visualization.lobe3d import animate_beam_steering
from pylobe.constants import C0, PI

FREQ = 900e6   # 900 MHz
LAM  = C0 / FREQ
N    = 10
D    = LAM / 2

print("=" * 60)
print("  Yagi-Uda Inspired Linear Array (N=10, SLL = -30 dB)")
print("=" * 60)

# -- Geometry -----------------------------------------------------------------
elem = HalfWaveDipole(freq=FREQ)
arr  = LinearArray(elem, N=N, d=D)

# Chebyshev weights for SLL = -30 dB
arr.chebyshev_weights(sll_db=30.0)
print(f"\nAmplitude weights (Chebyshev, SLL = -30 dB):")
print("  " + "  ".join(f"{w:.4f}" for w in arr.amplitudes))

# -- Grating lobe check -------------------------------------------------------
theta_scan = 0.0
has_gl = grating_lobe_condition(D, FREQ, theta_scan)
print(f"\nGrating lobe at broadside (d/lam={D/LAM:.2f}): {has_gl}")

# -- Pattern comparison: uniform vs Chebyshev ---------------------------------
theta    = np.linspace(0, PI, 721)
arr_unif = LinearArray(elem, N=N, d=D)
AF_unif  = arr_unif.array_factor(theta, FREQ)

arr_cheb = LinearArray(elem, N=N, d=D)
arr_cheb.chebyshev_weights(sll_db=30.0)
AF_cheb  = arr_cheb.array_factor(theta, FREQ)

fig = plot_polar_compare(
    [theta, theta],
    [AF_unif, AF_cheb],
    labels=['Uniform', 'Chebyshev -30 dB'],
    title='10-Element ULA Array Factor (broadside)',
)
fig.savefig('array_chebyshev_compare.png', dpi=600, bbox_inches='tight')
print("\nSaved: array_chebyshev_compare.png")
plt.close('all')

# -- Scan gain vs scan angle --------------------------------------------------
scan_angles_deg = np.linspace(-60, 60, 61)
peak_af         = []
for sa in scan_angles_deg:
    arr_scan = LinearArray(elem, N=N, d=D)
    arr_scan.scan_to(sa, FREQ)        # scan_to takes degrees
    arr_scan.chebyshev_weights(sll_db=30.0)
    AF_sc = arr_scan.array_factor(theta, FREQ)
    peak_af.append(10 * np.log10(np.max(AF_sc)**2 * N + 1e-30))

fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(scan_angles_deg, peak_af, 'C0', lw=2)
ax.set_xlabel('Scan Angle (degrees)')
ax.set_ylabel('Peak Array Factor (dB)')
ax.set_title('Scan Loss vs Beam Steering Angle')
ax.set_xlim(-60, 60)
ax.axhline(peak_af[len(peak_af)//2], ls='--', c='gray', lw=1, label='Broadside')
ax.grid(True, alpha=0.3)
ax.legend()
fig2.tight_layout()
fig2.savefig('array_scan_loss.png', dpi=600, bbox_inches='tight')
print("Saved: array_scan_loss.png")
plt.close('all')

# -- Beam steering animation --------------------------------------------------
scan_step_deg = np.array([-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0])
fig_anim = animate_beam_steering(arr_cheb, scan_step_deg, FREQ)
fig_anim.write_html('array_beam_steering.html')
print("Saved: array_beam_steering.html")
print("\nDone.")
