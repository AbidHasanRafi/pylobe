"""Tests for FDTD solver components."""
import numpy as np
import pytest
from pylobe.solver.fdtd.grid import FDTDGrid
from pylobe.solver.fdtd.pml import PML
from pylobe.constants import C0


class TestFDTDGrid:
    def test_courant_stability(self):
        """dt_courant must satisfy S = c0*dt*sqrt(1/dx²+...) ≤ 1."""
        dx = 1e-3
        grid = FDTDGrid(0.1, 0.1, 0.1, dx, dx, dx)
        dt = grid.courant_number()
        S = C0 * dt * np.sqrt(3.0 / dx**2)
        assert S <= 1.0 + 1e-10

    def test_field_array_shapes(self):
        grid = FDTDGrid(0.06, 0.06, 0.06, 0.01, 0.01, 0.01)
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        assert grid.Ex.shape == (Nx,   Ny+1, Nz+1)
        assert grid.Ey.shape == (Nx+1, Ny,   Nz+1)
        assert grid.Ez.shape == (Nx+1, Ny+1, Nz  )
        assert grid.Hx.shape == (Nx+1, Ny,   Nz  )
        assert grid.Hy.shape == (Nx,   Ny+1, Nz  )
        assert grid.Hz.shape == (Nx,   Ny,   Nz+1)

    def test_material_arrays_default(self):
        grid = FDTDGrid(0.06, 0.06, 0.06, 0.01, 0.01, 0.01)
        assert np.all(grid.eps_r == 1.0)
        assert np.all(grid.mu_r  == 1.0)
        assert np.all(grid.sigma == 0.0)

    def test_add_dielectric(self):
        grid = FDTDGrid(0.06, 0.06, 0.06, 0.01, 0.01, 0.01)
        grid.add_dielectric((0, 0.03), (0, 0.06), (0, 0.03), eps_r=4.4)
        assert np.any(grid.eps_r > 1.0)

    def test_add_pec(self):
        grid = FDTDGrid(0.06, 0.06, 0.06, 0.01, 0.01, 0.01)
        grid.add_pec((0.02, 0.04), (0.02, 0.04), (0.04, 0.05))
        assert np.any(grid.sigma > 1e5)

    def test_update_coefficients_shape(self):
        grid = FDTDGrid(0.06, 0.06, 0.06, 0.01, 0.01, 0.01)
        dt = grid.courant_number() * 0.99
        CA, CB = grid.update_coefficients(dt)
        assert CA.shape == (grid.Nx, grid.Ny, grid.Nz)
        assert CB.shape == (grid.Nx, grid.Ny, grid.Nz)
        assert np.all(CA <= 1.0)
        assert np.all(CB > 0.0)


class TestPML:
    def test_pml_raises_on_thin(self):
        grid = FDTDGrid(0.1, 0.1, 0.1, 0.005, 0.005, 0.005)
        with pytest.raises(ValueError):
            PML(grid, thickness=3)

    def test_pml_sets_sigma(self):
        grid = FDTDGrid(0.1, 0.1, 0.1, 0.005, 0.005, 0.005)
        pml  = PML(grid, thickness=8, R0=1e-8)
        assert np.any(grid.sigma > 0)

    def test_sigma_max_formula(self):
        """sigma_max = -(m+1)*ln(R0)*eps0*c0/(2*d_pml)."""
        from pylobe.constants import EPS0
        grid  = FDTDGrid(0.1, 0.1, 0.1, 0.005, 0.005, 0.005)
        pml   = PML(grid, thickness=10, m=3, R0=1e-8)
        d_pml = 10 * 0.005
        expected = -(3 + 1) * np.log(1e-8) * EPS0 * C0 / (2 * d_pml)
        assert expected > 0

    def test_sigma_profile_monotonic(self):
        grid = FDTDGrid(0.1, 0.1, 0.1, 0.005, 0.005, 0.005)
        pml  = PML(grid, thickness=10)
        d    = np.linspace(0, 0.05, 50)
        sig  = pml.sigma_profile(d, 0.05)
        assert np.all(np.diff(sig) >= 0)


class TestFDTDSources:
    def test_gaussian_pulse_peak(self):
        from pylobe.solver.fdtd.sources import gaussian_pulse
        t    = np.linspace(0, 10e-9, 1000)
        t0   = 5e-9
        sig  = 1e-9
        pulse = gaussian_pulse(t, t0, sig)
        assert abs(t[np.argmax(pulse)] - t0) < (t[1] - t[0])

    def test_gaussian_normalised(self):
        from pylobe.solver.fdtd.sources import gaussian_pulse
        t  = np.linspace(0, 10e-9, 1000)
        gp = gaussian_pulse(t, 5e-9, 1e-9)
        assert np.max(gp) <= 1.0 + 1e-12

    def test_modulated_gaussian_carrier(self):
        from pylobe.solver.fdtd.sources import modulated_gaussian
        f0 = 2.4e9
        t  = np.linspace(0, 5e-9, 5000)
        sig = modulated_gaussian(t, f0, 2.5e-9, 1e9)
        # Should be close to zero at t=0 (before pulse centre)
        assert abs(sig[0]) < 0.01

    def test_sinusoidal_cw_steady_state(self):
        from pylobe.solver.fdtd.sources import sinusoidal_cw
        f0  = 1e9
        T   = 1.0 / f0
        t   = np.linspace(0, 20 * T, 2000)
        sig = sinusoidal_cw(t, f0, ramp_cycles=5)
        # Steady state: amplitude ≈ 1
        steady = sig[1500:]
        assert np.max(np.abs(steady)) > 0.95
