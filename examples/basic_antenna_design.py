"""

Basic Antenna Design Examples

==============================

A beginner-friendly walkthrough of designing common antennas in PyLobe.



For each antenna you will see:

  - How to create the geometry with the right material

  - How to plot the PHYSICAL STRUCTURE (what it looks like)

  - How to compute and plot the RADIATION PATTERN



Output files

------------

All figures are saved as .png (patterns) or .html (3-D interactive) so you

can open them without a running GUI.



Antennas covered

----------------

1. Half-wave dipole at 300 MHz

2. Quarter-wave monopole at 900 MHz

3. Rectangular microstrip patch at 2.4 GHz (FR4)

4. Rectangular microstrip patch at 5.8 GHz (RT/duroid5880)

5. Circular patch at 2.45 GHz

6. Bow-tie UWB dipole

7. 8-element ULA of dipoles

"""

import sys

import io

# Ensure UTF-8 output on Windows terminals that default to cp1252.

if hasattr(sys.stdout, 'reconfigure'):

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')



import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt



# -- Geometry -------------------------------------------------------------------

from pylobe.geometry.patch   import RectangularPatch, CircularPatch

from pylobe.geometry.dipole  import HalfWaveDipole, BowTieDipole

from pylobe.geometry.monopole import QuarterWaveMonopole

from pylobe.geometry.array   import LinearArray



# -- Materials ------------------------------------------------------------------

from pylobe.geometry.base import (

    Material,

    FR4, RT5880,

    COPPER, GOLD, SILVER, ALUMINUM, PEC,

    register_material, get_material, print_material_table,

)



# -- Solvers --------------------------------------------------------------------

from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver

from pylobe.solver.analytical.dipole_solver import DipoleSolver

from pylobe.solver.analytical.array_factor import array_factor_ula



# -- Visualization --------------------------------------------------------------

from pylobe.visualization.structure import (

    plot_antenna_structure, plot_dipole_structure, plot_patch_structure,

    plot_circular_patch_structure, plot_bowtie_structure,

    plot_monopole_structure, plot_array_structure,

)

from pylobe.visualization.polar import plot_e_h_plane, plot_polar_compare, plot_polar

from pylobe.visualization.heatmap import plot_s11_vs_freq

from pylobe.visualization.style import export_fig



print("=" * 65)

print("  PyLobe -- Basic Antenna Design Examples")

print("=" * 65)

print()



# ============================================================

# EXAMPLE 1 -- Half-Wave Dipole  @ 300 MHz

# ============================================================

print("-" * 55)

print("Example 1: Half-Wave Dipole @ 300 MHz  (copper)")

print("-" * 55)



dipole = HalfWaveDipole(

    freq=300e6,

    length_factor=0.47,

    conductor_material=COPPER,

)

lambda0_d = 3e8 / 300e6

print(f"  Wavelength    : {lambda0_d*1e2:.1f} cm")

print(f"  Total length  : {dipole.L_total*1e2:.2f} cm  ({dipole.length_factor}λ)")

print(f"  Arm length    : {dipole.arm_length*1e2:.2f} cm")

print(f"  Wire radius   : {dipole.wire_radius*1e3:.2f} mm")

print(f"  Z_in (approx) : {dipole.impedance_approx} Ω")

print(f"  Material      : {dipole.conductor_material.name}")



# Physical structure

fig_d_struct = plot_dipole_structure(dipole, show_dimensions=True)

fig_d_struct.write_html("basic_dipole_structure.html")

print("  Saved: basic_dipole_structure.html")



# Radiation pattern

solver_d = DipoleSolver(dipole, freq=300e6)

theta    = np.linspace(0, np.pi, 361)

pat_lin  = solver_d.element_factor(theta)

pat_db   = 20 * np.log10(np.clip(pat_lin / np.max(pat_lin), 1e-10, None))



fig_d_pat = plot_polar(pat_db, np.rad2deg(theta),

                       label='Half-wave dipole  300 MHz',

                       dyn_range=40, color='#3498DB')

fig_d_pat.savefig("basic_dipole_pattern.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_dipole_pattern.png")

print(f"  Peak directivity: {solver_d.directivity:.2f} ({solver_d.directivity_dbi:.2f} dBi)")



# S11 sweep

freqs_d, s11_d = solver_d.s11(Z0=73.1, n_freq=300)

s11_db_d = 20 * np.log10(np.abs(s11_d) + 1e-20)

fig_s11_d = plot_s11_vs_freq(freqs_d, s11_db_d)

fig_s11_d.savefig("basic_dipole_s11.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_dipole_s11.png")

print()





# +==========================================================+

# |  EXAMPLE 2 -- Quarter-Wave Monopole  @ 900 MHz          |

# +==========================================================+

print("-" * 55)

print("Example 2: Quarter-Wave Monopole @ 900 MHz")

print("-" * 55)



mono = QuarterWaveMonopole(

    freq=900e6,

    length_factor=1.0,

    conductor_material=COPPER,

    ground_material=ALUMINUM,

)

print(f"  Length L      : {mono.L*1e2:.2f} cm")

print(f"  Ground radius : {mono.ground_radius*1e2:.2f} cm")

print(f"  Z_in (approx) : {mono.impedance_approx} Ω")

print(f"  Gain          : {mono.gain_dbi:.2f} dBi")



fig_m_struct = plot_monopole_structure(mono, show_dimensions=True)

fig_m_struct.write_html("basic_monopole_structure.html")

print("  Saved: basic_monopole_structure.html")



# Radiation pattern via half-dipole equivalence (image theory)

equiv_dipole = HalfWaveDipole(freq=900e6)

solver_m = DipoleSolver(equiv_dipole, freq=900e6)

# Monopole pattern = dipole pattern for θ ∈ [0, π/2] only (upper half-space)

theta_m = np.linspace(0, np.pi, 361)

pat_m   = solver_m.element_factor(theta_m)

# Zero out lower hemisphere (ground plane blocks it)

pat_m[theta_m > np.pi / 2] = 0.0

pat_db_m = 20 * np.log10(np.clip(pat_m / max(pat_m.max(), 1e-10), 1e-10, None))



fig_m_pat = plot_polar(pat_db_m, np.rad2deg(theta_m),

                       label='Quarter-wave monopole  900 MHz',

                       dyn_range=40, color='#E67E22')

fig_m_pat.savefig("basic_monopole_pattern.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_monopole_pattern.png")

print()





# +==========================================================+

# |  EXAMPLE 3 -- Rectangular Patch @ 2.4 GHz on FR4       |

# +==========================================================+

print("-" * 55)

print("Example 3: Rectangular Patch @ 2.4 GHz on FR4")

print("-" * 55)



patch_24 = RectangularPatch(

    freq=2.4e9,

    substrate_material=FR4,       # eps_r=4.4, tan_d=0.02

    h=1.6e-3,                     # 1.6 mm substrate thickness

    patch_material=COPPER,

    ground_material=PEC,

    inset_feed=True,

)

s = patch_24.summary()

print(f"  Substrate     : {s['substrate_material']}")

print(f"  Patch layer   : {s['patch_material']}")

print(f"  Ground plane  : {s['ground_material']}")

print(f"  Width  W      : {s['W_mm']:.3f} mm")

print(f"  Length L      : {s['L_mm']:.3f} mm")

print(f"  εeff          : {s['eps_eff']:.4f}")

print(f"  Feed inset y0 : {s['y0_mm']:.3f} mm")

print(f"  fr (design)   : {s['f_r_GHz']:.4f} GHz")

print(f"  BW (approx)   : {s['BW_percent']:.2f} %")



# Physical structure

fig_p24_struct = plot_patch_structure(patch_24, show_dimensions=True)

fig_p24_struct.write_html("basic_patch_24ghz_structure.html")

print("  Saved: basic_patch_24ghz_structure.html")



# Radiation pattern

solver_p24 = PatchAnalyticalSolver(patch_24, freq=2.4e9)

rp24 = solver_p24.radiation_pattern(n_theta=181, n_phi=73)

summary24 = rp24.summary()



fig_p24_eh = plot_e_h_plane(rp24)

fig_p24_eh.savefig("basic_patch_24ghz_pattern.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_patch_24ghz_pattern.png")

print(f"  Peak gain     : {summary24.peak_gain_dbi:.2f} dBi")

print(f"  HPBW (E)      : {summary24.hpbw_e_deg:.1f} deg")

print(f"  HPBW (H)      : {summary24.hpbw_h_deg:.1f} deg")



# S11 sweep

freqs_p24, s11_p24 = solver_p24.s11(Z0=50.0, n_freq=300)

s11_db_p24 = 20 * np.log10(np.abs(s11_p24) + 1e-20)

fig_s11_p24 = plot_s11_vs_freq(freqs_p24, s11_db_p24)

fig_s11_p24.savefig("basic_patch_24ghz_s11.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_patch_24ghz_s11.png")

print()





# +==========================================================+

# |  EXAMPLE 4 -- Rectangular Patch @ 5.8 GHz on RT5880    |

# +==========================================================+

print("-" * 55)

print("Example 4: Rectangular Patch @ 5.8 GHz on RT/duroid5880")

print("-" * 55)



patch_58 = RectangularPatch(

    freq=5.8e9,

    substrate_material=RT5880,    # eps_r=2.2, tan_d=0.0009 (low-loss)

    h=0.787e-3,

    patch_material=GOLD,          # gold-plated patch

    ground_material=COPPER,

    inset_feed=True,

)

s58 = patch_58.summary()

print(f"  Substrate     : {s58['substrate_material']}")

print(f"  Patch layer   : {s58['patch_material']}")

print(f"  W={s58['W_mm']:.3f} mm  L={s58['L_mm']:.3f} mm  "

      f"fr={s58['f_r_GHz']:.4f} GHz")



fig_p58_struct = plot_patch_structure(

    patch_58,

    title='5.8 GHz Patch -- RT/duroid5880 | Gold patch on Copper GND',

    show_dimensions=True,

)

fig_p58_struct.write_html("basic_patch_58ghz_structure.html")

print("  Saved: basic_patch_58ghz_structure.html")



solver_p58 = PatchAnalyticalSolver(patch_58, freq=5.8e9)

rp58 = solver_p58.radiation_pattern(n_theta=181, n_phi=73)

fig_p58_eh = plot_e_h_plane(rp58)

fig_p58_eh.savefig("basic_patch_58ghz_pattern.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_patch_58ghz_pattern.png")

print()





# +==========================================================+

# |  EXAMPLE 5 -- Circular Patch @ 2.45 GHz                 |

# +==========================================================+

print("-" * 55)

print("Example 5: Circular Patch @ 2.45 GHz on FR4")

print("-" * 55)



circ_patch = CircularPatch(

    freq=2.45e9,

    substrate_material=FR4,

    h=1.6e-3,

    patch_material=COPPER,

    ground_material=PEC,

)

print(f"  Physical radius a    : {circ_patch.a*1e3:.3f} mm")

print(f"  Effective radius a_eff: {circ_patch.a_eff*1e3:.3f} mm")



fig_c_struct = plot_circular_patch_structure(circ_patch)

fig_c_struct.write_html("basic_circular_patch_structure.html")

print("  Saved: basic_circular_patch_structure.html")

print()





# +==========================================================+

# |  EXAMPLE 6 -- Bow-Tie UWB Dipole @ 2 GHz               |

# +==========================================================+

print("-" * 55)

print("Example 6: Bow-Tie UWB Dipole @ 2 GHz  (60° flare, gold)")

print("-" * 55)



bowtie = BowTieDipole(

    freq=2e9,

    flare_angle_deg=60.0,

    conductor_material=GOLD,

)

print(f"  Arm length    : {bowtie.arm_length*1e3:.1f} mm")

print(f"  Flare angle   : {bowtie.flare_angle_deg}°")

print(f"  Material      : {bowtie.conductor_material.name}")



fig_bt_struct = plot_bowtie_structure(

    bowtie,

    title='Bow-Tie UWB Dipole -- 2 GHz | 60° | Gold',

)

fig_bt_struct.write_html("basic_bowtie_structure.html")

print("  Saved: basic_bowtie_structure.html")

print()





# +==========================================================+

# |  EXAMPLE 7 -- 8-Element ULA of Half-Wave Dipoles        |

# +==========================================================+

print("-" * 55)

print("Example 7: 8-element ULA of dipoles @ 2.4 GHz")

print("-" * 55)



lam = 3e8 / 2.4e9

elem = HalfWaveDipole(freq=2.4e9, conductor_material=COPPER)



# Broadside array (β=0)

ula_broad = LinearArray(element=elem, N=8, d=lam/2, beta=0.0)

# Scan to 30 degrees

ula_scan  = LinearArray(element=elem, N=8, d=lam/2)

ula_scan.scan_to(theta0_deg=30.0, freq=2.4e9)



theta_arr = np.linspace(0, np.pi, 361)

af_broad = ula_broad.array_factor(theta_arr, freq=2.4e9)

af_scan  = ula_scan.array_factor(theta_arr, freq=2.4e9)



fig_ula_struct = plot_array_structure(ula_broad, show_element_shape=True)

fig_ula_struct.write_html("basic_ula_structure.html")

print("  Saved: basic_ula_structure.html")



fig_ula_pat = plot_polar_compare(

    np.deg2rad(np.concatenate([np.rad2deg(theta_arr),

                               np.rad2deg(theta_arr) + 180.0])),

    patterns=[

        np.concatenate([af_broad, af_broad[::-1]]),

        np.concatenate([af_scan,  af_scan[::-1]]),

    ],

    labels=['Broadside (β=0)', 'Scan to 30°'],

    title='8-Element ULA Array Factor -- 2.4 GHz',

    dyn_range=40,

)

fig_ula_pat.savefig("basic_ula_pattern.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_ula_pattern.png")

print()





# +==========================================================+

# |  EXAMPLE 8 -- Pattern Overlay: Patch vs Dipole          |

# +==========================================================+

print("-" * 55)

print("Example 8: Pattern overlay -- Patch vs Dipole (E-plane)")

print("-" * 55)



rp_patch = solver_p24.radiation_pattern(n_theta=181, n_phi=73)

e_patch = rp_patch.e_plane_cut(0.0)

e_patch_max = max(e_patch.max(), 1e-10)

e_patch_db = 10 * np.log10(np.clip(e_patch / e_patch_max, 1e-10, None))



theta_half = np.linspace(0, np.pi, 181)

e_dipole_lin = solver_d.element_factor(theta_half)

e_dipole_max = max(e_dipole_lin.max(), 1e-10)

e_dipole_db  = 20 * np.log10(np.clip(e_dipole_lin / e_dipole_max, 1e-10, None))



# Mirror both to 0-360° for polar display

theta_360 = np.concatenate([np.rad2deg(theta_half),

                             np.rad2deg(theta_half) + 180.0])



fig_overlay = plot_polar_compare(

    np.deg2rad(theta_360),

    patterns=[

        np.concatenate([10**(e_patch_db/10), 10**(e_patch_db[::-1]/10)]),

        np.concatenate([10**(e_dipole_db/10), 10**(e_dipole_db[::-1]/10)]),

    ],

    labels=['Patch 2.4 GHz (E-plane)', 'Half-wave Dipole 300 MHz'],

    title='Pattern Comparison -- Patch vs Dipole',

    dyn_range=40,

)

fig_overlay.savefig("basic_pattern_comparison.png", dpi=600, bbox_inches='tight')

plt.close('all')

print("  Saved: basic_pattern_comparison.png")



print()

print("=" * 65)

print("  Summary of output files")

print("=" * 65)

outputs = [

    ("basic_dipole_structure.html",       "Dipole 300 MHz -- 3D structure"),

    ("basic_dipole_pattern.png",          "Dipole 300 MHz -- radiation pattern"),

    ("basic_dipole_s11.png",              "Dipole 300 MHz -- S11 sweep"),

    ("basic_monopole_structure.html",     "Monopole 900 MHz -- 3D structure"),

    ("basic_monopole_pattern.png",        "Monopole 900 MHz -- radiation pattern"),

    ("basic_patch_24ghz_structure.html",  "Patch 2.4 GHz FR4 -- 3D structure"),

    ("basic_patch_24ghz_pattern.png",     "Patch 2.4 GHz FR4 -- E/H plane"),

    ("basic_patch_24ghz_s11.png",         "Patch 2.4 GHz FR4 -- S11 sweep"),

    ("basic_patch_58ghz_structure.html",  "Patch 5.8 GHz RT5880 -- 3D structure"),

    ("basic_patch_58ghz_pattern.png",     "Patch 5.8 GHz RT5880 -- E/H plane"),

    ("basic_circular_patch_structure.html", "Circular patch 2.45 GHz -- 3D structure"),

    ("basic_bowtie_structure.html",       "Bow-tie UWB -- 3D structure"),

    ("basic_ula_structure.html",          "8-element ULA -- 3D structure"),

    ("basic_ula_pattern.png",             "8-element ULA -- array factor"),

    ("basic_pattern_comparison.png",      "Patch vs Dipole pattern overlay"),

]

for fname, desc in outputs:

    print(f"  {fname:<42} {desc}")

print()

