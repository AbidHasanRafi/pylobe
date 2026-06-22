"""Physics validation tests for analytical solvers."""
import numpy as np
import pytest
from pylobe.constants import C0, PI


class TestDipoleSolver:
    def test_half_wave_dipole_directivity(self):
        """Directivity of half-wave dipole must be 1.64 (2.15 dBi) within 1%."""
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.analytical.dipole_solver import DipoleSolver
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        solver = DipoleSolver(dipole, freq=1e9)
        D = solver.directivity
        assert abs(D - 1.6409) / 1.6409 < 0.01, f"D={D:.4f}, expected 1.6409"

    def test_dipole_directivity_dbi(self):
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.analytical.dipole_solver import DipoleSolver
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        solver = DipoleSolver(dipole, freq=1e9)
        D_dbi = solver.directivity_dbi
        assert abs(D_dbi - 2.15) < 0.05

    def test_radiation_resistance_half_wave(self):
        """Radiation resistance of half-wave dipole ≈ 73.1 Ω."""
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.analytical.dipole_solver import DipoleSolver
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        solver = DipoleSolver(dipole, freq=1e9)
        Rr = solver.radiation_resistance
        assert abs(Rr - 73.1) / 73.1 < 0.02, f"Rr={Rr:.2f}, expected ~73.1"

    def test_pattern_null_at_theta_0_and_180(self):
        """Half-wave dipole pattern should be zero at θ=0 and θ=π."""
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.analytical.dipole_solver import DipoleSolver
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        solver = DipoleSolver(dipole, freq=1e9)
        F_0   = float(solver.element_factor(np.array([1e-6]))[0])
        F_180 = float(solver.element_factor(np.array([PI - 1e-6]))[0])
        assert F_0   < 1e-4
        assert F_180 < 1e-4

    def test_pattern_maximum_at_90(self):
        """Pattern maximum at θ=90°."""
        from pylobe.geometry.dipole import HalfWaveDipole
        from pylobe.solver.analytical.dipole_solver import DipoleSolver
        dipole = HalfWaveDipole(freq=1e9, length_factor=0.5)
        solver = DipoleSolver(dipole, freq=1e9)
        theta = np.linspace(1e-3, PI - 1e-3, 1801)
        F     = solver.element_factor(theta)
        peak_theta = np.rad2deg(theta[np.argmax(F)])
        assert abs(peak_theta - 90.0) < 1.0


class TestPatchSolver:
    def test_patch_resonant_frequency_fr4(self):
        """Patch at 2.4 GHz on FR4 (εr=4.4, h=1.6mm): fr within 1%."""
        from pylobe.geometry.patch import RectangularPatch
        from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
        freq = 2.4e9
        patch = RectangularPatch(freq=freq, eps_r=4.4, h=1.6e-3)
        solver = PatchAnalyticalSolver(patch, freq=freq)
        fr = solver.resonant_frequency
        assert abs(fr - freq) / freq < 0.01, f"fr={fr/1e9:.4f} GHz, expected {freq/1e9}"

    def test_directivity_positive(self):
        from pylobe.geometry.patch import RectangularPatch
        from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
        patch  = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        solver = PatchAnalyticalSolver(patch, freq=2.4e9)
        assert solver.directivity > 1.0

    def test_e_plane_pattern_normalised(self):
        from pylobe.geometry.patch import RectangularPatch
        from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
        patch  = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        solver = PatchAnalyticalSolver(patch, freq=2.4e9)
        theta  = np.linspace(0, PI, 181)
        pat    = solver.e_plane_pattern(theta)
        assert np.max(pat) <= 1.0 + 1e-10
        assert np.min(pat) >= 0.0

    def test_input_impedance_returns_complex(self):
        from pylobe.geometry.patch import RectangularPatch
        from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
        patch  = RectangularPatch(freq=2.4e9, eps_r=4.4, h=1.6e-3)
        solver = PatchAnalyticalSolver(patch, freq=2.4e9)
        Zin = solver.input_impedance()
        assert isinstance(Zin, complex)
        assert Zin.real > 0


class TestSmithChart:
    def test_matched_load(self):
        """Z=Z0 → Γ=0."""
        from pylobe.analysis.smith import SmithChart
        sc = SmithChart(Z0=50.0)
        gamma = sc.impedance_to_gamma(50.0 + 0j)
        assert abs(gamma) < 1e-10

    def test_short_circuit(self):
        """Z=0 → Γ=-1."""
        from pylobe.analysis.smith import SmithChart
        sc = SmithChart(Z0=50.0)
        gamma = sc.impedance_to_gamma(0.0 + 0j)
        assert abs(gamma - (-1.0)) < 1e-10

    def test_open_circuit(self):
        """Z→∞ → Γ→+1."""
        from pylobe.analysis.smith import SmithChart
        sc = SmithChart(Z0=50.0)
        gamma = sc.impedance_to_gamma(1e8 + 0j)
        assert abs(gamma - 1.0) < 1e-4

    def test_vswr_matched(self):
        """VSWR at Γ=0 should be 1."""
        from pylobe.analysis.smith import SmithChart
        sc = SmithChart(Z0=50.0)
        vswr = sc.vswr(np.array([0.0 + 0j]))
        assert abs(vswr[0] - 1.0) < 1e-10

    def test_vswr_s11_relation(self):
        """VSWR = (1+|S11|)/(1-|S11|)."""
        from pylobe.analysis.smith import SmithChart
        sc = SmithChart(Z0=50.0)
        gamma = np.array([0.3 + 0.2j])
        mag   = np.abs(gamma)
        vswr_expected = (1 + mag) / (1 - mag)
        vswr_got = sc.vswr(gamma)
        assert np.allclose(vswr_got, vswr_expected, rtol=1e-10)

    def test_return_loss(self):
        """RL = -20·log10(|Γ|)."""
        from pylobe.analysis.smith import SmithChart
        sc  = SmithChart(Z0=50.0)
        mag = 0.316   # ≈ -10 dB
        rl  = sc.return_loss_db(np.array([mag + 0j]))
        assert abs(rl[0] - (-20.0 * np.log10(mag))) < 1e-8


class TestArrayFactor:
    def test_broadside_ula(self):
        """4-element ULA, d=λ/2, β=0 → peak at θ=90°."""
        from pylobe.solver.analytical.array_factor import array_factor_ula
        freq = 1e9
        lam  = C0 / freq
        theta = np.linspace(0, PI, 361)
        af    = array_factor_ula(theta, freq, N=4, d=lam/2, beta=0.0)
        assert np.argmax(af) == len(theta) // 2   # ≈ θ=90°

    def test_af_positive(self):
        from pylobe.solver.analytical.array_factor import array_factor_ula
        freq  = 1e9
        lam   = C0 / freq
        theta = np.linspace(0, PI, 181)
        af    = array_factor_ula(theta, freq, N=8, d=lam/2)
        assert np.all(af >= 0.0)
