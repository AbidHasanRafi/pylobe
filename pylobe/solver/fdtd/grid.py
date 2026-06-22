"""FDTD 3-D Yee grid definition.

Yee cell layout:
  - E-field components at edge centres
  - H-field components at face centres
  - Staggered in both space (Δ/2) and time (Δt/2)

Reference: Taflove & Hagness, *Computational Electrodynamics*, 3rd Ed., Ch. 3.
"""
import numpy as np
from pylobe.constants import C0, MU0, EPS0


class FDTDGrid:
    """3-D Yee grid for FDTD electromagnetic simulation.

    Parameters
    ----------
    size_x, size_y, size_z : float
        Physical domain size [m].
    dx, dy, dz : float
        Yee cell sizes [m].
    """

    def __init__(self, size_x: float, size_y: float, size_z: float,
                 dx: float, dy: float, dz: float):
        self.Nx = int(np.ceil(size_x / dx))
        self.Ny = int(np.ceil(size_y / dy))
        self.Nz = int(np.ceil(size_z / dz))
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.size_x = self.Nx * dx
        self.size_y = self.Ny * dy
        self.size_z = self.Nz * dz

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        # E-field components (at edge centres)
        self.Ex = np.zeros((Nx,   Ny+1, Nz+1))
        self.Ey = np.zeros((Nx+1, Ny,   Nz+1))
        self.Ez = np.zeros((Nx+1, Ny+1, Nz  ))

        # H-field components (at face centres, +1 in all dims for stencil boundary access)
        self.Hx = np.zeros((Nx+1, Ny+1, Nz+1))
        self.Hy = np.zeros((Nx+1, Ny+1, Nz+1))
        self.Hz = np.zeros((Nx+1, Ny+1, Nz+1))

        # Material arrays (per-cell, cell-centred)
        self.eps_r = np.ones((Nx, Ny, Nz))
        self.mu_r  = np.ones((Nx, Ny, Nz))
        self.sigma = np.zeros((Nx, Ny, Nz))   # electric conductivity [S/m]
        self.sigma_m = np.zeros((Nx, Ny, Nz)) # magnetic conductivity (for PML)

    # ------------------------------------------------------------------
    # Material assignment helpers
    # ------------------------------------------------------------------
    def add_dielectric(self, x_range: tuple, y_range: tuple,
                       z_range: tuple, eps_r: float,
                       loss_tan: float = 0.0):
        """Assign dielectric filling to a box region.

        Parameters
        ----------
        x_range, y_range, z_range : tuple of (lo, hi) in metres
        eps_r : float
            Relative permittivity.
        loss_tan : float
            Loss tangent; sets effective conductivity σ = ω·ε·tan(δ).
            Stored as tangent (frequency-dependent conversion done at run-time).
        """
        xi, xf = self._cell_range_x(x_range)
        yi, yf = self._cell_range_y(y_range)
        zi, zf = self._cell_range_z(z_range)
        self.eps_r[xi:xf, yi:yf, zi:zf] = eps_r
        self.sigma[xi:xf, yi:yf, zi:zf] = loss_tan  # stored as tan(δ)

    def add_pec(self, x_range: tuple, y_range: tuple, z_range: tuple):
        """Set a box region to perfect electric conductor.

        PEC is modelled as σ → ∞ (represented by large conductivity).
        """
        xi, xf = self._cell_range_x(x_range)
        yi, yf = self._cell_range_y(y_range)
        zi, zf = self._cell_range_z(z_range)
        self.sigma[xi:xf, yi:yf, zi:zf] = 1e7   # large conductivity → PEC

    def add_geometry(self, geometry, freq: float):
        """Map an AntennaGeometry object onto the FDTD grid.

        The geometry is centred in the XY plane and placed one PML-thickness
        above the bottom PML in Z so that the source sits well inside the
        radiation region, not inside the absorbing boundary.

        Parameters
        ----------
        geometry : AntennaGeometry
        freq : float
            Design frequency (needed for loss conversion).
        """
        from pylobe.geometry.patch import RectangularPatch
        if isinstance(geometry, RectangularPatch):
            lo, hi = geometry.bounding_box()
            geom_W = hi[0] - lo[0]   # physical width  (x)
            geom_L = hi[1] - lo[1]   # physical length (y)

            # Centre in XY; place substrate bottom 1 PML-thickness above z=0.
            # Store the applied offset so FDTDSimulation.add_source can use it.
            x_off = (self.size_x - geom_W) / 2.0
            y_off = (self.size_y - geom_L) / 2.0
            # Snap to nearest cell boundary to avoid fractional-cell offsets
            x_off = int(x_off / self.dx) * self.dx
            y_off = int(y_off / self.dy) * self.dy
            # Z: substrate bottom sits just above the PML (estimate pml from
            # the grid margin — use 10% of domain height as a safe clearance)
            z_off = max(self.dz * 2, self.size_z * 0.12)
            z_off = int(z_off / self.dz) * self.dz

            self._geom_offset = (x_off, y_off, z_off)

            x0, x1 = lo[0] + x_off, hi[0] + x_off
            y0, y1 = lo[1] + y_off, hi[1] + y_off
            z_bot  = z_off
            z_top  = z_off + geometry.h

            # Substrate layer
            self.add_dielectric(
                (x0, x1), (y0, y1), (z_bot, z_top),
                eps_r=geometry.eps_r,
                loss_tan=geometry.material.loss_tangent,
            )
            # Patch conductor (top surface)
            self.add_pec(
                (x0, x1), (y0, y1),
                (z_top - self.dz, z_top),
            )
            # Ground plane (bottom of substrate)
            self.add_pec(
                (x0, x1), (y0, y1), (z_bot, z_bot + self.dz),
            )

    # ------------------------------------------------------------------
    # Courant stability
    # ------------------------------------------------------------------
    def courant_number(self) -> float:
        """Maximum time step satisfying the Courant stability condition.

        S = c0·dt·sqrt(1/dx² + 1/dy² + 1/dz²) ≤ 1
        Returns dt_max [s].
        """
        return 1.0 / (C0 * np.sqrt(
            1.0/self.dx**2 + 1.0/self.dy**2 + 1.0/self.dz**2
        ))

    def update_coefficients(self, dt: float, freq: float = 1e9):
        """Pre-compute Yee update coefficients CA and CB.

        CA = (1 - σ·dt/(2·ε)) / (1 + σ·dt/(2·ε))
        CB = (dt/ε)             / (1 + σ·dt/(2·ε))

        where σ is converted from loss tangent if needed.

        Parameters
        ----------
        dt : float
            Time step [s].
        freq : float
            Centre frequency for loss-tangent conversion [Hz].

        Returns
        -------
        CA, CB : ndarray, shape (Nx, Ny, Nz)
        """
        import math
        omega = 2.0 * math.pi * freq
        # Convert stored loss_tangent to conductivity where sigma < 1 (not PEC)
        sigma_eff = np.where(
            self.sigma > 1.0,
            self.sigma,
            omega * EPS0 * self.eps_r * self.sigma,
        )
        eps = EPS0 * self.eps_r
        denom = 1.0 + sigma_eff * dt / (2.0 * eps)
        CA = (1.0 - sigma_eff * dt / (2.0 * eps)) / denom
        CB = (dt / eps) / denom
        return CA, CB

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _cell_range_x(self, x_range):
        lo = max(0, int(x_range[0] / self.dx))
        hi = min(self.Nx, int(np.ceil(x_range[1] / self.dx)))
        return lo, hi

    def _cell_range_y(self, y_range):
        lo = max(0, int(y_range[0] / self.dy))
        hi = min(self.Ny, int(np.ceil(y_range[1] / self.dy)))
        return lo, hi

    def _cell_range_z(self, z_range):
        lo = max(0, int(z_range[0] / self.dz))
        hi = min(self.Nz, int(np.ceil(z_range[1] / self.dz)))
        return lo, hi
