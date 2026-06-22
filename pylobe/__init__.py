"""
PyLobe: Antenna Design, EM Simulation, and AI Optimization Platform.

Supports:
  - Parametric antenna geometry (patch, dipole, monopole, slot, fractal, arrays)
  - Analytical, MoM, and FDTD solvers
  - Radiation pattern analysis and lobe decomposition
  - Smith chart, impedance, S-parameter analysis, and matching networks
  - Interactive 3-D and polar visualization (Matplotlib + Plotly)
  - Publication-quality figure export with IEEE / presentation style presets
  - GA, PSO, DE, and Bayesian optimization
  - Neural surrogate and inverse design
  - Monte Carlo manufacturing tolerance analysis
  - Unified AntennaDesign workflow object

All physical constants are from scipy.constants.
All dimensions are in SI units (metres, Hz, Ohms).

References:
  Balanis, C.A. — Antenna Theory: Analysis and Design, 4th Ed.
  Taflove & Hagness — Computational Electrodynamics, 3rd Ed.
  Harrington — Field Computation by Moment Methods.
  Pozar, D.M. — Microwave Engineering, 4th Ed.

Quick start
-----------
>>> from pylobe import RectangularPatch, ROGERS4003
>>> from pylobe.design import AntennaDesign
>>> patch = RectangularPatch(freq=2.4e9, substrate_material=ROGERS4003)
>>> design = AntennaDesign(patch).solve()
>>> print(design.summary())
>>> design.plot_pattern()
"""

__version__ = "0.1.0"
__author__  = "PyLobe Contributors"

# ── Geometry ──────────────────────────────────────────────────────────────
from pylobe.geometry.base import (
    AntennaGeometry, Material,
    AIR, FR4, RT5880, ROGERS4003, PEC,
    COPPER, GOLD, SILVER, ALUMINUM, BRASS,
    TEFLON, ALUMINA, SILICON, GaAs, ROGERS3010, ARLON250, FOAM,
    MATERIAL_LIBRARY, register_material, get_material, list_materials,
    print_material_table,
)
from pylobe.geometry.patch import (
    RectangularPatch, CircularPatch, AnnularRingPatch, ESlotPatch
)
from pylobe.geometry.dipole import HalfWaveDipole, FoldedDipole, BowTieDipole
from pylobe.geometry.monopole import QuarterWaveMonopole, HelicalMonopole
from pylobe.geometry.slot import SlotAntenna, VivaldiAntenna
from pylobe.geometry.fractal import KochDipole, SierpinskiGasket, MinkowskiPatch
from pylobe.geometry.array import LinearArray, PlanarArray, CircularArray
from pylobe.geometry.yagi import YagiUda
from pylobe.geometry.horn import PyramidalHorn
from pylobe.geometry.loop import SmallLoopAntenna, LargeLoopAntenna
from pylobe.geometry.lpda import LogPeriodicArray
from pylobe.geometry.pifa import PIFA

# ── Solvers ───────────────────────────────────────────────────────────────
from pylobe.solver.fdtd import FDTDSimulation
from pylobe.solver.analytical.patch_solver import (
    PatchAnalyticalSolver, CircularPatchAnalyticalSolver,
)
from pylobe.solver.analytical.dipole_solver import DipoleSolver
from pylobe.solver.analytical.array_factor import (
    array_factor_ula, array_factor_2d, grating_lobe_condition
)
from pylobe.solver.mom.wire import WireMoMSolver
from pylobe.solver.analytical.monopole_solver import MonopoleSolver
from pylobe.solver.analytical.yagi_solver import YagiAnalyticalSolver
from pylobe.solver.analytical.horn_solver import HornSolver
from pylobe.solver.analytical.loop_solver import LoopSolver
from pylobe.solver.analytical.slot_solver import SlotSolver
from pylobe.solver.analytical.folded_dipole_solver import FoldedDipoleSolver

# ── Analysis ──────────────────────────────────────────────────────────────
from pylobe.analysis.radiation import RadiationPattern, PatternSummary, PlanecutResult
from pylobe.analysis.lobe import LobeAnalyzer, Lobe
from pylobe.analysis.smith import SmithChart
from pylobe.analysis.metrics import (
    directivity, gain, radiation_efficiency,
    beamwidth_hpbw, side_lobe_level, front_to_back_ratio, axial_ratio,
)
from pylobe.analysis.sensitivity import SensitivityAnalyzer
from pylobe.analysis.tolerance import ToleranceAnalyzer, ToleranceResult, ToleranceResults

# ── Visualization ─────────────────────────────────────────────────────────
from pylobe.visualization.polar import (
    plot_polar, plot_polar_compare, plot_e_h_plane, plot_phase_pattern,
)
from pylobe.visualization.lobe3d import (
    plot_3d_radiation, plot_lobe_decomposition, animate_beam_steering
)
from pylobe.visualization.smith_chart import plot_smith_chart
from pylobe.visualization.heatmap import (
    plot_gain_heatmap, plot_s11_vs_freq, plot_vswr_vs_freq, plot_impedance_vs_freq,
)
from pylobe.visualization.nearfield_plot import (
    plot_nearfield_2d, plot_current_distribution, plot_geometry_3d
)
from pylobe.visualization.cartesian import plot_pattern_cartesian, plot_array_factor
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
from pylobe.visualization.style import set_style, get_style, export_fig

# ── Optimization ──────────────────────────────────────────────────────────
from pylobe.optimization.genetic import GeneticAlgorithm
from pylobe.optimization.pso import ParticleSwarmOptimizer
from pylobe.optimization.differential_evolution import DifferentialEvolution
from pylobe.optimization.bayesian import BayesianOptimizer
from pylobe.optimization.objectives import (
    maximize_gain, minimize_sll, minimize_s11, multi_objective
)

# ── AI ────────────────────────────────────────────────────────────────────
from pylobe.ai.surrogate import NeuralSurrogate
from pylobe.ai.inverse_design import InverseDesigner
from pylobe.ai.dataset import (
    generate_patch_dataset, generate_dipole_dataset,
    PATCH_FEATURE_NAMES, PATCH_TARGET_NAMES,
)

# ── Export ────────────────────────────────────────────────────────────────
from pylobe.geometry.export import to_dxf, to_stl, to_gds, to_json, from_json
from pylobe.export.report import generate_report

# ── High-level design workflow ────────────────────────────────────────────
from pylobe.design import AntennaDesign

# ── Constants ─────────────────────────────────────────────────────────────
from pylobe import constants

# ── Sources (FDTD) ────────────────────────────────────────────────────────
from pylobe.solver.fdtd.sources import (
    gaussian_pulse, modulated_gaussian, sinusoidal_cw
)

__all__ = [
    # Geometry
    "AntennaGeometry", "Material",
    # Pre-defined materials
    "AIR", "FR4", "RT5880", "ROGERS4003", "PEC",
    "COPPER", "GOLD", "SILVER", "ALUMINUM", "BRASS",
    "TEFLON", "ALUMINA", "SILICON", "GaAs", "ROGERS3010", "ARLON250", "FOAM",
    # Material registry
    "MATERIAL_LIBRARY", "register_material", "get_material",
    "list_materials", "print_material_table",
    "RectangularPatch", "CircularPatch", "AnnularRingPatch", "ESlotPatch",
    "HalfWaveDipole", "FoldedDipole", "BowTieDipole",
    "QuarterWaveMonopole", "HelicalMonopole",
    "SlotAntenna", "VivaldiAntenna",
    "KochDipole", "SierpinskiGasket", "MinkowskiPatch",
    "LinearArray", "PlanarArray", "CircularArray",
    "YagiUda", "PyramidalHorn",
    "SmallLoopAntenna", "LargeLoopAntenna",
    "LogPeriodicArray", "PIFA",
    # Solvers
    "FDTDSimulation", "PatchAnalyticalSolver", "CircularPatchAnalyticalSolver", "DipoleSolver",
    "array_factor_ula", "array_factor_2d", "grating_lobe_condition",
    "WireMoMSolver",
    "MonopoleSolver", "YagiAnalyticalSolver", "HornSolver",
    "LoopSolver", "SlotSolver", "FoldedDipoleSolver",
    # Analysis
    "RadiationPattern", "PatternSummary", "PlanecutResult",
    "LobeAnalyzer", "Lobe", "SmithChart",
    "directivity", "gain", "radiation_efficiency",
    "beamwidth_hpbw", "side_lobe_level", "front_to_back_ratio", "axial_ratio",
    "SensitivityAnalyzer",
    "ToleranceAnalyzer", "ToleranceResult", "ToleranceResults",
    # Visualization
    "plot_polar", "plot_polar_compare", "plot_e_h_plane", "plot_phase_pattern",
    "plot_3d_radiation", "plot_lobe_decomposition", "animate_beam_steering",
    "plot_smith_chart",
    "plot_gain_heatmap", "plot_s11_vs_freq", "plot_vswr_vs_freq",
    "plot_impedance_vs_freq",
    "plot_nearfield_2d", "plot_current_distribution", "plot_geometry_3d",
    "plot_pattern_cartesian", "plot_array_factor",
    # Physical structure visualization
    "plot_antenna_structure",
    "plot_patch_structure",
    "plot_circular_patch_structure",
    "plot_annular_patch_structure",
    "plot_eslot_patch_structure",
    "plot_dipole_structure",
    "plot_bowtie_structure",
    "plot_monopole_structure",
    "plot_helical_structure",
    "plot_array_structure",
    # Visualization style system
    "set_style", "get_style", "export_fig",
    # High-level design workflow
    "AntennaDesign",
    # Optimization
    "GeneticAlgorithm", "ParticleSwarmOptimizer", "DifferentialEvolution",
    "BayesianOptimizer",
    "maximize_gain", "minimize_sll", "minimize_s11", "multi_objective",
    # AI
    "NeuralSurrogate", "InverseDesigner",
    "generate_patch_dataset", "generate_dipole_dataset",
    "PATCH_FEATURE_NAMES", "PATCH_TARGET_NAMES",
    # Export
    "to_dxf", "to_stl", "to_gds", "to_json", "from_json", "generate_report",
    # Constants & Sources
    "constants",
    "gaussian_pulse", "modulated_gaussian", "sinusoidal_cw",
]
