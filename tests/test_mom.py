"""Tests for Method of Moments solver."""
import numpy as np
import pytest
from pylobe.geometry.dipole import HalfWaveDipole
from pylobe.solver.mom.wire import WireMoMSolver
from pylobe.constants import C0


class TestWireMoMSolver:
    @pytest.fixture
    def solved_dipole(self):
        freq   = 300e6
        dipole = HalfWaveDipole(freq=freq, length_factor=0.47, N_segments=11)
        solver = WireMoMSolver(dipole, freq=freq, N_segments=11)
        solver.solve()
        return solver

    def test_impedance_matrix_shape(self):
        freq   = 300e6
        dipole = HalfWaveDipole(freq=freq, N_segments=7)
        solver = WireMoMSolver(dipole, freq=freq, N_segments=7)
        Z = solver.build_impedance_matrix()
        assert Z.shape == (7, 7)

    def test_impedance_matrix_dtype(self):
        freq   = 300e6
        dipole = HalfWaveDipole(freq=freq, N_segments=7)
        solver = WireMoMSolver(dipole, freq=freq, N_segments=7)
        Z = solver.build_impedance_matrix()
        assert np.iscomplexobj(Z)

    def test_current_shape(self, solved_dipole):
        I = solved_dipole._currents
        assert len(I) == solved_dipole.N

    def test_current_nonzero(self, solved_dipole):
        I = solved_dipole._currents
        assert np.max(np.abs(I)) > 1e-10

    def test_input_impedance_real_positive(self, solved_dipole):
        Zin = solved_dipole.input_impedance()
        assert Zin.real > 0

    def test_current_symmetric_about_feed(self, solved_dipole):
        """Current distribution should be (approximately) symmetric."""
        I = np.abs(solved_dipole._currents)
        N = len(I)
        # Compare first half to reversed second half
        half = N // 2
        symmetry_error = np.mean(np.abs(I[:half] - I[N-half:][::-1])) / I.max()
        assert symmetry_error < 0.15
