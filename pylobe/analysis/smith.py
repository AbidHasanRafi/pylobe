"""Smith chart impedance calculations and matching network analysis."""
import numpy as np
from pylobe.constants import PI


class SmithChart:
    """Smith chart calculations in normalised impedance z = Z/Z0.

    Parameters
    ----------
    Z0 : float
        Reference (normalised) impedance [Ω]. Default 50 Ω.
    """

    def __init__(self, Z0: float = 50.0):
        self.Z0 = Z0

    # ------------------------------------------------------------------
    # Conversion functions
    # ------------------------------------------------------------------
    def impedance_to_gamma(self, Z: np.ndarray) -> np.ndarray:
        """Complex reflection coefficient.

        Γ = (Z - Z0) / (Z + Z0)

        Parameters
        ----------
        Z : array_like of complex
            Load impedance [Ω].

        Returns
        -------
        ndarray of complex
        """
        Z = np.asarray(Z, dtype=complex)
        return (Z - self.Z0) / (Z + self.Z0)

    def gamma_to_impedance(self, gamma: np.ndarray) -> np.ndarray:
        """Convert reflection coefficient to impedance.

        Z = Z0 · (1 + Γ) / (1 - Γ)

        Parameters
        ----------
        gamma : array_like of complex

        Returns
        -------
        ndarray of complex [Ω]
        """
        gamma = np.asarray(gamma, dtype=complex)
        denom = 1.0 - gamma
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        return self.Z0 * (1.0 + gamma) / denom

    def vswr(self, gamma: np.ndarray) -> np.ndarray:
        """Voltage Standing Wave Ratio.

        VSWR = (1 + |Γ|) / (1 - |Γ|)

        Parameters
        ----------
        gamma : array_like of complex

        Returns
        -------
        ndarray of float
        """
        mag = np.clip(np.abs(np.asarray(gamma, dtype=complex)), 0.0, 0.9999)
        return (1.0 + mag) / (1.0 - mag)

    def return_loss_db(self, gamma: np.ndarray) -> np.ndarray:
        """Return loss [dB].

        RL = -20·log10(|Γ|)

        Parameters
        ----------
        gamma : array_like of complex

        Returns
        -------
        ndarray of float [dB]
        """
        mag = np.abs(np.asarray(gamma, dtype=complex))
        return -20.0 * np.log10(np.clip(mag, 1e-15, None))

    def mismatch_loss_db(self, gamma: np.ndarray) -> np.ndarray:
        """Mismatch loss [dB].

        ML = -10·log10(1 - |Γ|²)

        Parameters
        ----------
        gamma : array_like of complex

        Returns
        -------
        ndarray of float [dB]
        """
        mag2 = np.abs(np.asarray(gamma, dtype=complex)) ** 2
        return -10.0 * np.log10(np.clip(1.0 - mag2, 1e-15, None))

    # ------------------------------------------------------------------
    # Matching network
    # ------------------------------------------------------------------
    def matching_network(self, Z_source: complex, Z_load: complex) -> dict:
        """L-network matching for conjugate match Z_in = Z_source*.

        Computes the two possible L-network topologies:
        - Series element + shunt element (source side)
        - Shunt element + series element (load side)

        Parameters
        ----------
        Z_source : complex
            Source impedance [Ω].
        Z_load : complex
            Load impedance [Ω].

        Returns
        -------
        dict with keys:
            'topology' : str
            'Q'        : float (network Q factor)
            'X_series' : float [Ω]
            'X_shunt'  : float [Ω]
            'description' : str
        """
        Rs = Z_source.real
        Rl = Z_load.real

        if Rs <= 0 or Rl <= 0:
            return {'topology': 'none', 'description': 'Non-resistive source/load'}

        if Rs > Rl:
            # Step-down: shunt element at source, series at load
            Q = np.sqrt(Rs / Rl - 1.0)
            Xs_shunt  = Rs / Q
            Xs_series = Q * Rl
            topology = 'shunt-source_series-load'
        else:
            # Step-up: series element at source, shunt at load
            Q = np.sqrt(Rl / Rs - 1.0)
            Xs_shunt  = Rl / Q
            Xs_series = Q * Rs
            topology = 'series-source_shunt-load'

        return {
            'topology':    topology,
            'Q':           float(Q),
            'X_series_Ω': float(Xs_series),
            'X_shunt_Ω':  float(Xs_shunt),
            'description': (
                f"L-network: Q={Q:.2f}, "
                f"X_series={Xs_series:.2f} Ω, X_shunt={Xs_shunt:.2f} Ω"
            ),
        }

    # ------------------------------------------------------------------
    # Constant-Q circles
    # ------------------------------------------------------------------
    def q_circles(self, Q: float, n_pts: int = 200) -> np.ndarray:
        """Generate constant-Q circles in the Γ-plane.

        Q = Im(z) / Re(z)  (normalised impedance z = Z/Z0)
        Constant-Q loci are circles in the Γ-plane.

        Parameters
        ----------
        Q : float
            Quality factor.
        n_pts : int
            Number of points on the circle.

        Returns
        -------
        ndarray of complex, shape (n_pts,)
            Points on the constant-Q circle in the Γ-plane.
        """
        # Normalised resistance range
        r_arr = np.logspace(-2, 2, n_pts)
        z_arr = r_arr * (1.0 + 1j * Q)
        gamma_arr = (z_arr - 1.0) / (z_arr + 1.0)
        return gamma_arr

    # ------------------------------------------------------------------
    # Grid lines for visualisation
    # ------------------------------------------------------------------
    def constant_r_circle(self, r: float, n_pts: int = 200) -> np.ndarray:
        """Normalised resistance circle in Γ-plane.

        For z = r + jx (r fixed, x varies):
            Γ = (z - 1) / (z + 1)
        Circle centre: r/(1+r), radius: 1/(1+r).

        Parameters
        ----------
        r : float
            Normalised resistance.
        n_pts : int

        Returns
        -------
        ndarray of complex
        """
        x = np.linspace(-50.0, 50.0, n_pts)
        z = r + 1j * x
        return (z - 1.0) / (z + 1.0)

    def constant_x_arc(self, x: float, n_pts: int = 200) -> np.ndarray:
        """Normalised reactance arc in Γ-plane.

        For z = r + jx (x fixed, r varies 0 to ∞):
        Circle centre: (1, 1/x), radius: 1/|x|.

        Parameters
        ----------
        x : float
            Normalised reactance (positive or negative).
        n_pts : int

        Returns
        -------
        ndarray of complex
        """
        r = np.logspace(-3, 3, n_pts)
        z = r + 1j * x
        return (z - 1.0) / (z + 1.0)
