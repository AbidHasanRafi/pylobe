"""
Antenna Structure Visualisation Examples
=========================================
Demonstrates how to plot the physical 3-D structure of different antenna
types with their assigned materials using PyLobe.

Each antenna's structure is rendered as an interactive Plotly HTML figure
that you can open in any web browser to rotate, zoom, and inspect.

Antennas covered
----------------
1. Rectangular patch (FR4 / copper / PEC)
2. Rectangular patch with custom gold patch on Rogers RT/duroid5880
3. Circular patch
4. Annular ring patch
5. E-slot dual-band patch
6. Half-wave dipole (copper)
7. Folded dipole
8. Bow-tie ultra-wideband dipole
9. Quarter-wave monopole with aluminium ground
10. Helical antenna (axial mode)
11. 8-element linear array of dipoles
12. 4×4 planar array
13. Circular array of 8 elements
"""
import numpy as np

# ── PyLobe imports ─────────────────────────────────────────────────────────────
from pylobe.geometry.patch import (
    RectangularPatch, CircularPatch, AnnularRingPatch, ESlotPatch
)
from pylobe.geometry.dipole import HalfWaveDipole, FoldedDipole, BowTieDipole
from pylobe.geometry.monopole import QuarterWaveMonopole, HelicalMonopole
from pylobe.geometry.array import LinearArray, PlanarArray, CircularArray

from pylobe.geometry.base import (
    Material,
    FR4, RT5880, ROGERS4003, ALUMINA,
    COPPER, GOLD, SILVER, ALUMINUM, PEC,
    register_material, get_material, list_materials, print_material_table,
)

from pylobe.visualization.structure import (
    plot_antenna_structure,
    plot_patch_structure,
    plot_circular_patch_structure,
    plot_annular_patch_structure,
    plot_eslot_patch_structure,
    plot_dipole_structure,
    plot_bowtie_structure,
    plot_monopole_structure,
    plot_helical_structure,
    plot_array_structure,
)

print("=" * 65)
print("  PyLobe — Antenna Structure Visualisation Examples")
print("=" * 65)

# ── Print the material library ─────────────────────────────────────────────────
print("\nAvailable materials in the PyLobe library:")
print_material_table()

# ── 1. Rectangular Patch — FR4 substrate, copper patch, PEC ground ─────────────
print("\n[1] Rectangular patch  2.4 GHz on FR4")
patch_fr4 = RectangularPatch(
    freq=2.4e9,
    substrate_material=FR4,
    h=1.6e-3,
    patch_material=COPPER,
    ground_material=PEC,
    inset_feed=True,
)
print(f"    W={patch_fr4.W*1e3:.2f} mm  L={patch_fr4.L*1e3:.2f} mm  "
      f"y0={patch_fr4.y0*1e3:.2f} mm  fr={patch_fr4.resonant_frequency/1e9:.4f} GHz")
fig1 = plot_patch_structure(patch_fr4, show_dimensions=True)
fig1.write_html("struct_patch_fr4.html")
print("    Saved: struct_patch_fr4.html")

# ── 2. Rectangular Patch — RT5880 with gold patch ─────────────────────────────
print("\n[2] Rectangular patch  5.8 GHz on RT/duroid5880 with gold")
patch_rt = RectangularPatch(
    freq=5.8e9,
    substrate_material=RT5880,
    h=0.787e-3,
    patch_material=GOLD,
    ground_material=COPPER,
    inset_feed=True,
)
print(f"    W={patch_rt.W*1e3:.2f} mm  L={patch_rt.L*1e3:.2f} mm  "
      f"fr={patch_rt.resonant_frequency/1e9:.4f} GHz")
fig2 = plot_patch_structure(patch_rt,
                            title='5.8 GHz Patch — RT/duroid5880 | Gold Patch')
fig2.write_html("struct_patch_rt5880_gold.html")
print("    Saved: struct_patch_rt5880_gold.html")

# ── 3. Rectangular Patch — Rogers4003 with silver patch (custom colours) ───────
print("\n[3] Rectangular patch  10 GHz on Rogers4003 with silver")
patch_r4003 = RectangularPatch(
    freq=10e9,
    substrate_material=ROGERS4003,
    h=0.508e-3,
    patch_material=SILVER,
    ground_material=COPPER,
    inset_feed=True,
)
fig3 = plot_patch_structure(patch_r4003,
                            title='10 GHz Patch — Rogers4003C | Silver Patch')
fig3.write_html("struct_patch_rogers4003.html")
print("    Saved: struct_patch_rogers4003.html")

# ── 4. Circular Patch ──────────────────────────────────────────────────────────
print("\n[4] Circular patch  2.45 GHz on FR4")
circ = CircularPatch(
    freq=2.45e9,
    substrate_material=FR4,
    h=1.6e-3,
    patch_material=COPPER,
    ground_material=PEC,
)
print(f"    radius a={circ.a*1e3:.2f} mm  a_eff={circ.a_eff*1e3:.2f} mm")
fig4 = plot_circular_patch_structure(circ)
fig4.write_html("struct_circular_patch.html")
print("    Saved: struct_circular_patch.html")

# ── 5. Annular Ring Patch ─────────────────────────────────────────────────────
print("\n[5] Annular ring patch  3.5 GHz on ALUMINA")
ring = AnnularRingPatch(
    freq=3.5e9,
    substrate_material=ALUMINA,
    h=0.635e-3,
    patch_material=GOLD,
    ground_material=COPPER,
)
print(f"    r_in={ring.inner_radius*1e3:.2f} mm  r_out={ring.outer_radius*1e3:.2f} mm")
fig5 = plot_annular_patch_structure(ring)
fig5.write_html("struct_annular_ring.html")
print("    Saved: struct_annular_ring.html")

# ── 6. E-Slot Dual-Band Patch ─────────────────────────────────────────────────
print("\n[6] E-slot dual-band patch  2.4/5.2 GHz on FR4")
eslot = ESlotPatch(
    freq1=2.4e9, freq2=5.2e9,
    substrate_material=FR4,
    h=1.6e-3,
    patch_material=COPPER,
    ground_material=PEC,
)
fig6 = plot_eslot_patch_structure(eslot)
fig6.write_html("struct_eslot_dualband.html")
print("    Saved: struct_eslot_dualband.html")

# ── 7. Custom Material — Defined by the User ──────────────────────────────────
print("\n[7] Custom substrate: MyPCB  eps_r=3.2  tan_d=0.01")
my_sub = Material(
    name="MyPCB",
    eps_r=3.2,
    loss_tangent=0.01,
    color='#6495ED',   # cornflower blue
)
register_material(my_sub)
print(f"    Registered materials now: {len(list_materials())}")

patch_custom = RectangularPatch(
    freq=2.4e9,
    substrate_material=my_sub,
    h=1.2e-3,
    patch_material=COPPER,
    ground_material=ALUMINUM,
    inset_feed=True,
)
fig7 = plot_patch_structure(
    patch_custom,
    title='2.4 GHz Patch — Custom MyPCB substrate | Copper on Aluminum GND',
)
fig7.write_html("struct_patch_custom_material.html")
print("    Saved: struct_patch_custom_material.html")

# ── 8. Half-Wave Dipole ───────────────────────────────────────────────────────
print("\n[8] Half-wave dipole  300 MHz copper")
dipole = HalfWaveDipole(
    freq=300e6,
    length_factor=0.47,
    conductor_material=COPPER,
)
print(f"    L_total={dipole.L_total*1e2:.1f} cm  arm={dipole.arm_length*1e2:.1f} cm")
fig8 = plot_dipole_structure(dipole, show_dimensions=True)
fig8.write_html("struct_dipole_half_wave.html")
print("    Saved: struct_dipole_half_wave.html")

# ── 9. Folded Dipole ──────────────────────────────────────────────────────────
print("\n[9] Folded dipole  300 MHz silver")
folded = FoldedDipole(freq=300e6, conductor_material=SILVER)
fig9 = plot_dipole_structure(folded,
                             title='Folded Dipole — 300 MHz | Silver')
fig9.write_html("struct_folded_dipole.html")
print("    Saved: struct_folded_dipole.html")

# ── 10. Bow-Tie UWB Dipole ───────────────────────────────────────────────────
print("\n[10] Bow-tie UWB dipole  2 GHz  60° flare  gold")
bowtie = BowTieDipole(
    freq=2e9,
    flare_angle_deg=60.0,
    conductor_material=GOLD,
)
fig10 = plot_bowtie_structure(bowtie,
                              title='Bow-Tie UWB — 2 GHz | 60° flare | Gold')
fig10.write_html("struct_bowtie_dipole.html")
print("    Saved: struct_bowtie_dipole.html")

# ── 11. Quarter-Wave Monopole ─────────────────────────────────────────────────
print("\n[11] Quarter-wave monopole  900 MHz  copper + aluminium GND")
mono = QuarterWaveMonopole(
    freq=900e6,
    conductor_material=COPPER,
    ground_material=ALUMINUM,
)
print(f"    L={mono.L*1e2:.1f} cm  GND_r={mono.ground_radius*1e2:.1f} cm")
fig11 = plot_monopole_structure(mono, show_dimensions=True)
fig11.write_html("struct_monopole_quarter_wave.html")
print("    Saved: struct_monopole_quarter_wave.html")

# ── 12. Helical Axial-Mode Antenna ────────────────────────────────────────────
print("\n[12] Helical antenna (axial mode)  2.4 GHz  6 turns")
helix = HelicalMonopole(
    freq=2.4e9,
    N_turns=6,
    diameter=39e-3,      # ≈ λ/π for axial mode at 2.4 GHz
    pitch_angle=14.0,    # degrees
    mode='axial',
    conductor_material=COPPER,
    ground_material=ALUMINUM,
)
print(f"    C={helix.circumference*1e3:.1f} mm  S={helix.turn_spacing*1e3:.1f} mm  "
      f"gain≈{helix.gain_approx_dbi:.1f} dBi")
fig12 = plot_helical_structure(helix)
fig12.write_html("struct_helix_axial.html")
print("    Saved: struct_helix_axial.html")

# ── 13. Normal-Mode Helix ─────────────────────────────────────────────────────
print("\n[13] Helical antenna (normal mode)  433 MHz  3 turns")
helix_normal = HelicalMonopole(
    freq=433e6,
    N_turns=3,
    diameter=20e-3,
    pitch_angle=8.0,
    mode='normal',
    conductor_material=COPPER,
)
fig13 = plot_helical_structure(helix_normal,
                               title='Normal-Mode Helical — 433 MHz  3 turns')
fig13.write_html("struct_helix_normal.html")
print("    Saved: struct_helix_normal.html")

# ── 14. Linear Array of Dipoles ──────────────────────────────────────────────
print("\n[14] 8-element ULA of half-wave dipoles  2.4 GHz")
element_d = HalfWaveDipole(freq=2.4e9, conductor_material=COPPER)
lambda_m  = 3e8 / 2.4e9
ula = LinearArray(
    element=element_d,
    N=8,
    d=lambda_m / 2.0,   # half-wavelength spacing
)
ula.chebyshev_weights(sll_db=25)    # -25 dB Chebyshev taper
print(f"    N=8  d={lambda_m/2*1e3:.1f} mm  Chebyshev -25 dB taper")
fig14 = plot_array_structure(ula, show_element_shape=True)
fig14.write_html("struct_ula_dipole.html")
print("    Saved: struct_ula_dipole.html")

# ── 15. Planar Patch Array ────────────────────────────────────────────────────
print("\n[15] 4×4 planar patch array  5.8 GHz on RT5880")
elem_p = RectangularPatch(
    freq=5.8e9,
    substrate_material=RT5880,
    h=0.787e-3,
    patch_material=COPPER,
)
planar = PlanarArray(
    element=elem_p,
    M=4, N=4,
    dx=lambda_m / 2.0,
    dy=lambda_m / 2.0,
)
fig15 = plot_array_structure(planar,
                             title='4×4 Planar Patch Array — 5.8 GHz')
fig15.write_html("struct_planar_patch_array.html")
print("    Saved: struct_planar_patch_array.html")

# ── 16. Circular Array ────────────────────────────────────────────────────────
print("\n[16] Circular array  8 elements  2.4 GHz")
circ_arr = CircularArray(
    element=element_d,
    N=8,
    R=lambda_m / (2 * np.sin(np.pi / 8)),  # inter-element spacing ≈ λ/2
)
fig16 = plot_array_structure(circ_arr,
                             title='Circular Array — 8 elements | 2.4 GHz')
fig16.write_html("struct_circular_array.html")
print("    Saved: struct_circular_array.html")

# ── 17. Using the generic dispatcher ─────────────────────────────────────────
print("\n[17] Generic dispatcher: plot_antenna_structure() auto-dispatch")
antennas = [
    patch_fr4, circ, dipole, mono, helix, ula
]
names    = [
    "patch_fr4", "circular", "dipole", "monopole", "helix", "ula"
]
for ant, name in zip(antennas, names):
    fig = plot_antenna_structure(ant)
    fname = f"struct_auto_{name}.html"
    fig.write_html(fname)
    print(f"    Saved: {fname}")

print("\n" + "=" * 65)
print("  All structure plots saved as .html — open in your browser!")
print("=" * 65)
