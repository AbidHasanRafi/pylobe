"""Tests for lobe analysis engine."""
import numpy as np
import pytest
from pylobe.constants import C0, PI


@pytest.fixture
def dipole_pattern():
    from pylobe.geometry.dipole import HalfWaveDipole
    from pylobe.solver.analytical.dipole_solver import DipoleSolver
    dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
    solver = DipoleSolver(dipole, freq=1e9)
    return solver.radiation_pattern(n_theta=91, n_phi=37)


class TestLobeAnalyzer:
    def test_find_lobes_returns_list(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        lobes = analyzer.find_lobes()
        assert isinstance(lobes, list)
        assert len(lobes) >= 1

    def test_main_lobe_type(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        main = analyzer.main_lobe()
        assert main is not None
        assert main.lobe_type == 'main'

    def test_main_lobe_peak_is_global_max(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        main = analyzer.main_lobe()
        D_dbi = dipole_pattern.to_dbi()
        assert abs(main.peak_gain_dbi - np.max(D_dbi)) < 0.5

    def test_beam_solid_angle_positive(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        omega = analyzer.beam_solid_angle()
        assert omega > 0

    def test_beam_solid_angle_relation_to_directivity(self, dipole_pattern):
        """D = 4π / Ω_A: within 20% for coarse grid."""
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        omega = analyzer.beam_solid_angle()
        D_from_omega = 4.0 * PI / omega
        D_actual = dipole_pattern.peak_directivity_linear
        assert abs(D_from_omega - D_actual) / D_actual < 0.20

    def test_null_map_shape(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        nm = analyzer.null_map()
        assert nm.shape == (len(dipole_pattern.theta), len(dipole_pattern.phi))

    def test_lobes_sorted_descending(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        lobes = analyzer.find_lobes()
        gains = [l.peak_gain_dbi for l in lobes]
        assert gains == sorted(gains, reverse=True)

    def test_encircled_power_fraction_monotonic(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        f30  = analyzer.encircled_power_fraction(30.0)
        f60  = analyzer.encircled_power_fraction(60.0)
        f120 = analyzer.encircled_power_fraction(120.0)
        assert f30 <= f60 <= f120

    def test_lobe_asymmetry_index_range(self, dipole_pattern):
        from pylobe.analysis.lobe import LobeAnalyzer
        analyzer = LobeAnalyzer(dipole_pattern)
        lai = analyzer.lobe_asymmetry_index()
        assert 0.0 <= lai <= 1.0


class TestRadiationPattern:
    def test_summary_keys(self, dipole_pattern):
        s = dipole_pattern.summary()
        for key in ('peak_gain_dbi', 'hpbw_e', 'hpbw_h', 'sll_db', 'fbr_db'):
            assert key in s

    def test_directivity_positive(self, dipole_pattern):
        assert dipole_pattern.peak_directivity_linear > 0

    def test_e_plane_nonnegative(self, dipole_pattern):
        e_cut = dipole_pattern.e_plane_cut()
        assert np.all(e_cut >= 0.0)
