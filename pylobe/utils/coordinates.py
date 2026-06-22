"""Coordinate conversion utilities for spherical/Cartesian transforms."""
import numpy as np


def cart_to_sph(x, y, z):
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates [m].

    Returns
    -------
    r : ndarray
        Radial distance [m].
    theta : ndarray
        Polar angle [rad], 0 to π.
    phi : ndarray
        Azimuthal angle [rad], 0 to 2π.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.where(r == 0, 1, r), -1, 1))
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, theta, phi


def sph_to_cart(r, theta, phi):
    """Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : array_like
        Radial distance [m].
    theta : array_like
        Polar angle [rad].
    phi : array_like
        Azimuthal angle [rad].

    Returns
    -------
    x, y, z : ndarray
    """
    r, theta, phi = np.asarray(r), np.asarray(theta), np.asarray(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def theta_hat(theta, phi):
    """Unit vector θ̂ in Cartesian: [cosθ cosφ, cosθ sinφ, -sinθ].

    Parameters
    ----------
    theta, phi : array_like
        Angles [rad].

    Returns
    -------
    ndarray, shape (..., 3)
    """
    theta, phi = np.asarray(theta), np.asarray(phi)
    return np.stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ], axis=-1)


def phi_hat(theta, phi):
    """Unit vector φ̂ in Cartesian: [-sinφ, cosφ, 0].

    Parameters
    ----------
    theta, phi : array_like
        Angles [rad].

    Returns
    -------
    ndarray, shape (..., 3)
    """
    phi = np.asarray(phi)
    zeros = np.zeros_like(phi)
    return np.stack([-np.sin(phi), np.cos(phi), zeros], axis=-1)


def r_hat(theta, phi):
    """Unit vector r̂ in Cartesian: [sinθ cosφ, sinθ sinφ, cosθ].

    Parameters
    ----------
    theta, phi : array_like
        Angles [rad].

    Returns
    -------
    ndarray, shape (..., 3)
    """
    theta, phi = np.asarray(theta), np.asarray(phi)
    return np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], axis=-1)


def db(x, power: bool = False):
    """Convert linear quantity to decibels.

    Parameters
    ----------
    x : array_like
        Input values (must be positive for power=True, any for field).
    power : bool
        If True, use 10·log10; if False (field/amplitude), use 20·log10(|x|).

    Returns
    -------
    ndarray
        Value in dB.
    """
    x = np.asarray(x, dtype=complex if not power else float)
    if power:
        return 10.0 * np.log10(np.abs(x))
    return 20.0 * np.log10(np.abs(x) + 1e-300)


def db_to_linear(x_db):
    """Convert power dB to linear scale: 10^(x_dB/10).

    Parameters
    ----------
    x_db : array_like
        Power in dB.

    Returns
    -------
    ndarray
    """
    return 10.0 ** (np.asarray(x_db) / 10.0)
