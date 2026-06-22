"""Tests for visualization modules (headless / non-rendering)."""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')   # headless backend


@pytest.fixture(scope='module')
def dipole_pattern():
    from pylobe.geometry.dipole import HalfWaveDipole
    from pylobe.solver.analytical.dipole_solver import DipoleSolver
    dipole = HalfWaveDipole(freq=1e9)
    solver = DipoleSolver(dipole, freq=1e9)
    return solver.radiation_pattern(n_theta=91, n_phi=37)


@pytest.fixture(scope='module')
def patch_solver():
    from pylobe.geometry.patch import RectangularPatch
    from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
    patch  = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
    return PatchAnalyticalSolver(patch, freq=2.4e9)


def _to_db(pat_linear):
    """Convert linear pattern to normalised dB."""
    p_max = np.max(pat_linear) or 1.0
    return 10.0 * np.log10(np.clip(pat_linear / p_max, 1e-10, None))


class TestPolarPlots:
    def test_plot_polar_returns_figure(self, dipole_pattern):
        from pylobe.visualization.polar import plot_polar
        import matplotlib.pyplot as plt
        theta_deg = np.rad2deg(dipole_pattern.theta)
        pat_db    = _to_db(dipole_pattern.e_plane_cut())
        fig       = plot_polar(pat_db, theta_deg, label='E-plane')
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_plot_e_h_plane_returns_figure(self, dipole_pattern):
        from pylobe.visualization.polar import plot_e_h_plane
        import matplotlib.pyplot as plt
        fig = plot_e_h_plane(dipole_pattern)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_plot_polar_compare_returns_figure(self, dipole_pattern):
        from pylobe.visualization.polar import plot_polar_compare
        import matplotlib.pyplot as plt
        theta = dipole_pattern.theta        # radians
        pat   = dipole_pattern.e_plane_cut()  # linear
        fig   = plot_polar_compare([theta, theta], [pat, pat], labels=['A', 'B'])
        assert isinstance(fig, plt.Figure)
        plt.close('all')


class TestHeatmapPlots:
    def test_s11_plot_returns_figure(self, patch_solver):
        from pylobe.visualization.heatmap import plot_s11_vs_freq
        import matplotlib.pyplot as plt
        freqs = np.linspace(2.0e9, 2.8e9, 50)
        # s11() returns (freq_array, S11_complex); compute at each freq via impedance_sweep
        Z    = patch_solver.impedance_sweep(n_freq=50, span_fraction=0.4)
        s11  = (Z - 50.0) / (Z + 50.0)
        s11_db = 20 * np.log10(np.abs(s11) + 1e-20)
        fig = plot_s11_vs_freq(freqs, s11_db)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_vswr_plot_returns_figure(self, patch_solver):
        from pylobe.visualization.heatmap import plot_vswr_vs_freq
        import matplotlib.pyplot as plt
        freqs = np.linspace(2.0e9, 2.8e9, 50)
        Z    = patch_solver.impedance_sweep(n_freq=50, span_fraction=0.4)
        s11  = np.abs((Z - 50.0) / (Z + 50.0))
        vswr = (1 + s11) / (1 - s11 + 1e-12)
        fig = plot_vswr_vs_freq(freqs, vswr)
        assert isinstance(fig, plt.Figure)
        plt.close('all')

    def test_gain_heatmap_returns_figure(self, dipole_pattern):
        from pylobe.visualization.heatmap import plot_gain_heatmap
        fig = plot_gain_heatmap(dipole_pattern)
        assert fig is not None


class TestSmithChartPlot:
    def test_smith_chart_returns_figure(self):
        from pylobe.visualization.smith_chart import plot_smith_chart
        from pylobe.analysis.smith import SmithChart
        sc     = SmithChart(Z0=50.0)
        freqs  = np.linspace(2.0e9, 2.8e9, 30)
        gammas = np.array([0.2 * np.exp(1j * 2 * np.pi * f / 2.4e9) for f in freqs])
        fig    = plot_smith_chart(gammas, freqs)
        assert fig is not None


class TestGeometry3D:
    def test_plot_geometry_3d_runs(self):
        from pylobe.geometry.patch import RectangularPatch
        from pylobe.visualization.nearfield_plot import plot_geometry_3d
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        fig   = plot_geometry_3d(patch)
        assert fig is not None

    def test_plot_current_1d_runs(self):
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.mom.wire import WireMoMSolver
        from pylobe.visualization.current_plot import plot_current_1d
        import matplotlib.pyplot as plt
        dipole = HalfWaveDipole(freq=300e6, N_segments=11)
        solver = WireMoMSolver(dipole, freq=300e6, N_segments=11)
        solver.solve()
        z = np.linspace(-dipole.L_total/2, dipole.L_total/2, solver.N)
        fig = plot_current_1d(z, solver._currents, freq=300e6)
        assert isinstance(fig, plt.Figure)
        plt.close('all')


class TestLobe3D:
    def test_plot_3d_radiation_runs(self, dipole_pattern):
        from pylobe.visualization.lobe3d import plot_3d_radiation
        fig = plot_3d_radiation(dipole_pattern)
        assert fig is not None

    def test_plot_lobe_decomposition_runs(self, dipole_pattern):
        from pylobe.visualization.lobe3d import plot_lobe_decomposition
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        lobes    = analyzer.find_lobes()
        fig      = plot_lobe_decomposition(dipole_pattern, lobes)
        assert fig is not None
