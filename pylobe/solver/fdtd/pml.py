"""Uniaxial Perfectly Matched Layer (UPML) absorbing boundary.

Implements polynomial-graded conductivity profile.
Reference: Taflove & Hagness, *Computational Electrodynamics*, 3rd Ed., Ch. 7.
"""
import numpy as np
from pylobe.constants import C0, EPS0, MU0


class PML:
    """Uniaxial PML absorbing boundary condition.

    Conductivity profile (polynomial grading):
        σ(d) = σ_max · (d / d_pml)^m

    where:
        d      = distance from inner PML interface [m]
        d_pml  = PML thickness [m]
        m      = grading order (typically 3–4)
        σ_max  = -(m+1)·ln(R0)·ε0·c0 / (2·d_pml)
        R0     = target reflection coefficient (default 1e-8)

    Parameters
    ----------
    grid : FDTDGrid
    thickness : int
        PML thickness in cells.
    m : int
        Polynomial grading order.
    R0 : float
        Target reflection coefficient.
    """

    def __init__(self, grid, thickness: int = 10,
                 m: int = 3, R0: float = 1e-8):
        if thickness < 8:
            raise ValueError(f"PML thickness must be >= 8 cells, got {thickness}")
        self.grid = grid
        self.thickness = thickness
        self.m = m
        self.R0 = R0
        self._build_sigma_arrays()

    # ------------------------------------------------------------------
    # Conductivity profile
    # ------------------------------------------------------------------
    def sigma_profile(self, d: np.ndarray, d_pml: float) -> np.ndarray:
        """Return σ at distance d into the PML [S/m].

        Parameters
        ----------
        d : array_like
            Distance from inner PML boundary [m]. 0 at interface, d_pml at outer.
        d_pml : float
            PML thickness [m].

        Returns
        -------
        ndarray
        """
        sigma_max = (-(self.m + 1) * np.log(self.R0) * EPS0 * C0
                     / (2.0 * d_pml))
        return sigma_max * (np.asarray(d) / d_pml) ** self.m

    # ------------------------------------------------------------------
    # Build sigma arrays on the grid
    # ------------------------------------------------------------------
    def _build_sigma_arrays(self):
        """Set σx, σy, σz (and magnetic counterparts) on all 6 faces + edges + corners."""
        g = self.grid
        t = self.thickness
        Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

        d_pml_x = t * g.dx
        d_pml_y = t * g.dy
        d_pml_z = t * g.dz

        # Allocate per-axis conductivity (cell-centred)
        self.sigma_x = np.zeros((Nx, Ny, Nz))
        self.sigma_y = np.zeros((Nx, Ny, Nz))
        self.sigma_z = np.zeros((Nx, Ny, Nz))

        # X-faces
        for i in range(t):
            d = (t - i - 0.5) * g.dx
            s = self.sigma_profile(d, d_pml_x)
            self.sigma_x[i, :, :] = s
            self.sigma_x[Nx - 1 - i, :, :] = s

        # Y-faces
        for j in range(t):
            d = (t - j - 0.5) * g.dy
            s = self.sigma_profile(d, d_pml_y)
            self.sigma_y[:, j, :] = s
            self.sigma_y[:, Ny - 1 - j, :] = s

        # Z-faces
        for k in range(t):
            d = (t - k - 0.5) * g.dz
            s = self.sigma_profile(d, d_pml_z)
            self.sigma_z[:, :, k] = s
            self.sigma_z[:, :, Nz - 1 - k] = s

        # Total effective conductivity (for update coefficients)
        g.sigma = self.sigma_x + self.sigma_y + self.sigma_z

    # ------------------------------------------------------------------
    # PML update coefficients
    # ------------------------------------------------------------------
    def pml_e_coefficients(self, dt: float, freq: float = 1e9):
        """Pre-compute E-field PML update coefficients for each axis.

        Returns (CA_x, CB_x, CA_y, CB_y, CA_z, CB_z).
        """
        def coeff(sig, eps_r):
            eps = EPS0 * eps_r
            denom = 1.0 + sig * dt / (2.0 * eps)
            CA = (1.0 - sig * dt / (2.0 * eps)) / denom
            CB = (dt / eps) / denom
            return CA, CB

        g = self.grid
        eps_r = g.eps_r
        CA_x, CB_x = coeff(self.sigma_x, eps_r)
        CA_y, CB_y = coeff(self.sigma_y, eps_r)
        CA_z, CB_z = coeff(self.sigma_z, eps_r)
        return CA_x, CB_x, CA_y, CB_y, CA_z, CB_z

    def apply_to_grid(self):
        """Write effective conductivity to grid (already done in __init__)."""
        pass  # sigma already applied in _build_sigma_arrays

    def update_H_pml(self, grid, dt: float):
        """H-field update in PML regions (same as interior — UPML formulation)."""
        from pylobe.solver.fdtd.update_equations import update_H
        update_H(grid, dt)

    def update_E_pml(self, grid, CA: np.ndarray, CB: np.ndarray, dt: float):
        """E-field update in PML regions."""
        from pylobe.solver.fdtd.update_equations import update_E
        update_E(grid, CA, CB, dt)
