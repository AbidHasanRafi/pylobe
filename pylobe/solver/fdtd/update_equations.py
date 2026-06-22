"""Vectorised Yee FDTD curl update equations.

Implements the exact discretisation of Maxwell's curl equations.
All array operations are vectorised NumPy slicing — no Python loops.

Reference: Taflove & Hagness, *Computational Electrodynamics*, 3rd Ed., Ch. 3.
"""
import numpy as np
from pylobe.constants import MU0


def update_H(grid, dt: float):
    """Vectorised H-field update for all interior Yee cells (n-1/2 → n+1/2).

    Hx^{n+1/2} = Hx^{n-1/2} - dt/(μ0·μr) * (∂Ez/∂y - ∂Ey/∂z)
    Hy^{n+1/2} = Hy^{n-1/2} - dt/(μ0·μr) * (∂Ex/∂z - ∂Ez/∂x)
    Hz^{n+1/2} = Hz^{n-1/2} - dt/(μ0·μr) * (∂Ey/∂x - ∂Ex/∂y)

    Parameters
    ----------
    grid : FDTDGrid
    dt : float
        Time step [s].
    """
    coeff = dt / (MU0 * grid.mu_r[:, :-1, :-1])   # shape (Nx, Ny, Nz) → (Nx, Ny-1, Nz-1) subset

    # Hx update: shape (Nx+1, Ny, Nz) → interior slice
    coeff_x = dt / (MU0 * grid.mu_r[:, :, :])     # (Nx, Ny, Nz)
    # Broadcast to (Nx, Ny, Nz) for Hx[:, :-1 (or :), :-1 (or :)]
    # Use grid dimensions directly; mu_r is (Nx, Ny, Nz)

    # Hx: index [i, j, k] updates from Ez[i,j+1,k]-Ez[i,j,k] and Ey[i,j,k+1]-Ey[i,j,k]
    grid.Hx[:grid.Nx, :grid.Ny, :grid.Nz] -= (
        dt / (MU0 * grid.mu_r) * (
            (grid.Ez[:grid.Nx, 1:grid.Ny+1, :grid.Nz] - grid.Ez[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dy
          - (grid.Ey[:grid.Nx, :grid.Ny, 1:grid.Nz+1] - grid.Ey[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dz
        )
    )

    # Hy: index [i, j, k] updates from Ex[i,j,k+1]-Ex[i,j,k] and Ez[i+1,j,k]-Ez[i,j,k]
    grid.Hy[:grid.Nx, :grid.Ny, :grid.Nz] -= (
        dt / (MU0 * grid.mu_r) * (
            (grid.Ex[:grid.Nx, :grid.Ny, 1:grid.Nz+1] - grid.Ex[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dz
          - (grid.Ez[1:grid.Nx+1, :grid.Ny, :grid.Nz] - grid.Ez[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dx
        )
    )

    # Hz: index [i, j, k] updates from Ey[i+1,j,k]-Ey[i,j,k] and Ex[i,j+1,k]-Ex[i,j,k]
    grid.Hz[:grid.Nx, :grid.Ny, :grid.Nz] -= (
        dt / (MU0 * grid.mu_r) * (
            (grid.Ey[1:grid.Nx+1, :grid.Ny, :grid.Nz] - grid.Ey[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dx
          - (grid.Ex[:grid.Nx, 1:grid.Ny+1, :grid.Nz] - grid.Ex[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dy
        )
    )


def update_E(grid, CA: np.ndarray, CB: np.ndarray, dt: float):
    """Vectorised E-field update for all interior Yee cells (n → n+1).

    Ex^{n+1} = CA·Ex^n + CB·(∂Hz/∂y - ∂Hy/∂z)
    Ey^{n+1} = CA·Ey^n + CB·(∂Hx/∂z - ∂Hz/∂x)
    Ez^{n+1} = CA·Ez^n + CB·(∂Hy/∂x - ∂Hx/∂y)

    where CA, CB include conductivity loss (pre-computed by grid.update_coefficients).

    Parameters
    ----------
    grid : FDTDGrid
    CA : ndarray, shape (Nx, Ny, Nz)
        Update coefficient (dimensionless).
    CB : ndarray, shape (Nx, Ny, Nz)
        Update coefficient [m/A·s → m·s/(F)].
    dt : float
        Time step [s].
    """
    # Ex update: interior slice
    grid.Ex[:grid.Nx, :grid.Ny, :grid.Nz] = (
        CA * grid.Ex[:grid.Nx, :grid.Ny, :grid.Nz]
        + CB * (
            (grid.Hz[:grid.Nx, 1:grid.Ny+1, :grid.Nz] - grid.Hz[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dy
          - (grid.Hy[:grid.Nx, :grid.Ny, 1:grid.Nz+1] - grid.Hy[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dz
        )
    )

    # Ey update
    grid.Ey[:grid.Nx, :grid.Ny, :grid.Nz] = (
        CA * grid.Ey[:grid.Nx, :grid.Ny, :grid.Nz]
        + CB * (
            (grid.Hx[:grid.Nx, :grid.Ny, 1:grid.Nz+1] - grid.Hx[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dz
          - (grid.Hz[1:grid.Nx+1, :grid.Ny, :grid.Nz] - grid.Hz[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dx
        )
    )

    # Ez update
    grid.Ez[:grid.Nx, :grid.Ny, :grid.Nz] = (
        CA * grid.Ez[:grid.Nx, :grid.Ny, :grid.Nz]
        + CB * (
            (grid.Hy[1:grid.Nx+1, :grid.Ny, :grid.Nz] - grid.Hy[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dx
          - (grid.Hx[:grid.Nx, 1:grid.Ny+1, :grid.Nz] - grid.Hx[:grid.Nx, :grid.Ny, :grid.Nz]) / grid.dy
        )
    )
