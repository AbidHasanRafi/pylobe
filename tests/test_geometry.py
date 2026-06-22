"""Tests for antenna geometry classes."""
import numpy as np
import pytest
from pylobe.geometry.patch import RectangularPatch, CircularPatch
from pylobe.geometry.dipole import HalfWaveDipole, FoldedDipole
from pylobe.geometry.monopole import QuarterWaveMonopole
from pylobe.geometry.array import LinearArray
from pylobe.constants import C0


class TestRectangularPatch:
    def test_width_formula(self):
        """W = c/(2f)*sqrt(2/(εr+1))"""
        freq, eps_r = 2.4e9, 4.4
        patch = RectangularPatch(freq=freq, eps_r=eps_r, h=1.6e-3)
        expected_W = (C0 / (2 * freq)) * np.sqrt(2.0 / (eps_r + 1))
        assert abs(patch.W - expected_W) / expected_W < 1e-10

    def test_resonant_frequency_close_to_design(self):
        """Resonant frequency within 5% of design frequency."""
        freq = 2.4e9
        patch = RectangularPatch(freq=freq, eps_r=4.4, h=1.6e-3)
        assert abs(patch.resonant_frequency - freq) / freq < 0.05

    def test_vertices_shape(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        assert patch.vertices.shape[1] == 3
        assert len(patch.vertices) == 8

    def test_feed_point_exists(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        assert patch.feed_point is not None
        assert len(patch.feed_point) == 3

    def test_eps_eff_between_1_and_eps_r(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        assert 1.0 < patch.eps_eff < 4.4

    def test_g1_positive(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        assert patch.G1 > 0

    def test_inset_feed_y0_in_range(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3, inset_feed=True)
        assert 0 <= patch.y0 <= patch.L

    def test_bounding_box(self):
        patch = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        lo, hi = patch.bounding_box()
        assert hi[2] == pytest.approx(1.6e-3, rel=1e-6)
        assert lo[2] == pytest.approx(0.0, abs=1e-15)

    def test_rt5880_substrate(self):
        """Rogers RT/duroid 5880: εr=2.2, h=1.57mm → f_r ≈ 2.4 GHz within 2%."""
        freq = 2.4e9
        patch = RectangularPatch(freq=freq, eps_r=2.2, h=1.57e-3)
        assert abs(patch.resonant_frequency - freq) / freq < 0.02


class TestHalfWaveDipole:
    def test_total_length(self):
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        lam = C0 / 1e9
        assert abs(dipole.L_total - 0.5 * lam) / (0.5 * lam) < 1e-10

    def test_vertices_along_z(self):
        dipole = HalfWaveDipole(freq=1e9)
        assert dipole.vertices.shape == (dipole.N_segments + 1, 3)
        assert np.allclose(dipole.vertices[:, 0], 0.0)
        assert np.allclose(dipole.vertices[:, 1], 0.0)

    def test_feed_at_origin(self):
        dipole = HalfWaveDipole(freq=1e9)
        assert np.allclose(dipole.feed_point, [0, 0, 0])

    def test_wire_radius_default(self):
        dipole = HalfWaveDipole(freq=1e9)
        assert dipole.wire_radius > 0
        assert dipole.wire_radius < dipole.L_total


class TestLinearArray:
    def test_broadside_main_beam(self):
        """4-element ULA, d=λ/2, β=0 → main beam at θ=90°."""
        from pylobe.geometry.dipole import HalfWaveDipole
        freq = 1e9
        lam  = C0 / freq
        elem = HalfWaveDipole(freq=freq)
        arr  = LinearArray(elem, N=4, d=lam/2, beta=0.0)
        theta = np.linspace(0, np.pi, 361)
        af = arr.array_factor(theta, freq)
        peak_theta = np.rad2deg(theta[np.argmax(af)])
        assert abs(peak_theta - 90.0) < 2.0

    def test_endfire_main_beam(self):
        """4-element ULA, d=λ/4, β=-kd → main beam at θ≈0°."""
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.constants import PI
        freq = 1e9
        lam  = C0 / freq
        k    = 2 * PI * freq / C0
        d    = lam / 4
        elem = HalfWaveDipole(freq=freq)
        arr  = LinearArray(elem, N=4, d=d, beta=-k*d)
        theta = np.linspace(0, np.pi, 361)
        af = arr.array_factor(theta, freq)
        peak_theta = np.rad2deg(theta[np.argmax(af)])
        assert peak_theta < 25.0   # endfire region

    def test_af_normalised(self):
        from pylobe.geometry.dipole import HalfWaveDipole
        freq = 2e9
        lam  = C0 / freq
        elem = HalfWaveDipole(freq=freq)
        arr  = LinearArray(elem, N=8, d=lam/2)
        theta = np.linspace(0, np.pi, 181)
        af = arr.array_factor(theta, freq)
        assert np.max(af) <= 1.0 + 1e-10

    def test_grating_lobe_condition(self):
        from pylobe.solver.analytical.array_factor import grating_lobe_condition
        freq = 1e9
        lam  = C0 / freq
        # d = lam → grating lobe exists at broadside
        assert grating_lobe_condition(lam, freq, 0.0) is True
        # d = lam/2 → no grating lobe
        assert grating_lobe_condition(lam/2, freq, 0.0) is False
