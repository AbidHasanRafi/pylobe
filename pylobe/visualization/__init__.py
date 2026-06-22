"""PyLobe visualization subpackage — research-grade antenna pattern plots.

Exports
-------
Radiation pattern plots:
  plot_polar, plot_polar_compare, plot_e_h_plane, plot_phase_pattern
  plot_3d_radiation, plot_lobe_decomposition, animate_beam_steering
  plot_smith_chart
  plot_gain_heatmap, plot_gain_heatmap_mpl
  plot_s11_vs_freq, plot_vswr_vs_freq, plot_impedance_vs_freq
  plot_return_loss_vs_freq
  plot_waterfall
  plot_nearfield_2d, plot_current_distribution, plot_geometry_3d
  plot_current_1d, plot_surface_current
  plot_pattern_cartesian, plot_array_factor

Comprehensive dashboards:
  plot_pattern_dashboard
  plot_frequency_dashboard
  plot_design_dashboard
  plot_compare_patterns

Physical structure visualization:
  plot_antenna_structure, plot_patch_structure, plot_circular_patch_structure,
  plot_annular_patch_structure, plot_eslot_patch_structure,
  plot_dipole_structure, plot_bowtie_structure, plot_monopole_structure,
  plot_helical_structure, plot_array_structure

Style and export (600 DPI):
  set_style, get_style, export_fig
"""
from pylobe.visualization.polar import (
    plot_polar, plot_polar_compare, plot_e_h_plane, plot_phase_pattern,
)
from pylobe.visualization.lobe3d import (
    plot_3d_radiation, plot_lobe_decomposition, animate_beam_steering,
)
from pylobe.visualization.smith_chart import plot_smith_chart
from pylobe.visualization.heatmap import (
    plot_gain_heatmap,
    plot_gain_heatmap_mpl,
    plot_s11_vs_freq,
    plot_vswr_vs_freq,
    plot_impedance_vs_freq,
    plot_return_loss_vs_freq,
    plot_waterfall,
)
from pylobe.visualization.nearfield_plot import (
    plot_nearfield_2d, plot_current_distribution, plot_geometry_3d,
)
from pylobe.visualization.current_plot import plot_current_1d, plot_surface_current
from pylobe.visualization.cartesian import plot_pattern_cartesian, plot_array_factor
from pylobe.visualization.dashboard import (
    plot_pattern_dashboard,
    plot_frequency_dashboard,
    plot_design_dashboard,
    plot_compare_patterns,
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
from pylobe.visualization.style import set_style, get_style, export_fig

__all__ = [
    # Radiation pattern plots
    'plot_polar', 'plot_polar_compare', 'plot_e_h_plane', 'plot_phase_pattern',
    'plot_3d_radiation', 'plot_lobe_decomposition', 'animate_beam_steering',
    'plot_smith_chart',
    'plot_gain_heatmap', 'plot_gain_heatmap_mpl',
    'plot_s11_vs_freq', 'plot_vswr_vs_freq', 'plot_impedance_vs_freq',
    'plot_return_loss_vs_freq', 'plot_waterfall',
    'plot_nearfield_2d', 'plot_current_distribution', 'plot_geometry_3d',
    'plot_current_1d', 'plot_surface_current',
    'plot_pattern_cartesian', 'plot_array_factor',
    # Comprehensive dashboards
    'plot_pattern_dashboard', 'plot_frequency_dashboard',
    'plot_design_dashboard', 'plot_compare_patterns',
    # Physical structure visualization
    'plot_antenna_structure',
    'plot_patch_structure',
    'plot_circular_patch_structure',
    'plot_annular_patch_structure',
    'plot_eslot_patch_structure',
    'plot_dipole_structure',
    'plot_bowtie_structure',
    'plot_monopole_structure',
    'plot_helical_structure',
    'plot_array_structure',
    # Style and export utilities
    'set_style', 'get_style', 'export_fig',
]
