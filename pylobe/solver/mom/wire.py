"""Method of Moments solver for wire antennas.

Implements Pocklington's integro-differential equation with pulse basis
functions and point matching.
Reference: Harrington, *Field Computation by Moment Methods*, Ch. 5.
"""
import numpy as np
from scipy.integrate import quad
from pylobe.constants import C0, PI, ETA0, EPS0, MU0
from pylobe.geometry.dipole import HalfWaveDipole
from pylobe.geometry.base import AntennaGeometry


class WireMoMSolver:
    """Method of Moments solver for thin wire antennas.

    Uses pulse basis functions and point matching (collocation).
    Pocklington's kernel with reduced kernel for self-impedance.

    Parameters
    ----------
    geometry : AntennaGeometry
        Wire geometry (must have ``vertices`` along z-axis).
    freq : float
        Operating frequency [Hz].
    N_segments : int
        Number of wire segments (odd preferred for symmetric feed).
    """

    def __init__(self, geometry: AntennaGeometry, freq: float,
                 N_segments: int = 21):
        self.geometry = geometry
        self.freq = freq
        self.N = N_segments
        self.k = 2.0 * PI * freq / C0
        self.omega = 2.0 * PI * freq

        # Extract wire end-points
        verts = geometry.vertices
        z_min = verts[:, 2].min()
        z_max = verts[:, 2].max()
        self.z_total = z_max - z_min
        self.delta = self.z_total / N_segments

        # Segment centres and endpoints
        self.z_seg = np.linspace(z_min + self.delta / 2.0,
                                 z_max - self.delta / 2.0, N_segments)

        # Wire radius from geometry attribute or default
        if hasattr(geometry, 'wire_radius'):
            self.a = geometry.wire_radius
        else:
            self.a = self.z_total / 200.0

        # Determine feed segment index (closest to z=0 or feed_point z)
        if geometry.feed_point is not None:
            z_feed = geometry.feed_point[2]
        else:
            z_feed = 0.0
        self.feed_seg = int(np.argmin(np.abs(self.z_seg - z_feed)))

        # Cache
        self._Z_matrix = None
        self._currents = None

    # ------------------------------------------------------------------
    # Impedance matrix
    # ------------------------------------------------------------------
    def build_impedance_matrix(self) -> np.ndarray:
        """Build N×N complex impedance matrix Z_mn.

        Z_mn = ∫_{segment n} kernel K(z_m, z') dz'

        Diagonal (self-terms) use reduced kernel to avoid singularity.

        Returns
        -------
        ndarray of complex128, shape (N, N)
        """
        N = self.N
        k = self.k
        a = self.a
        delta = self.delta
        z = self.z_seg

        Z = np.zeros((N, N), dtype=complex)
        prefactor = -1j * k * ETA0 / (4.0 * PI)

        for m in range(N):
            for n in range(N):
                if m == n:
                    # Self-impedance: reduced kernel R = sqrt(z²+a²)
                    half = delta / 2.0

                    def ker_re(dz):
                        R = np.sqrt(dz**2 + a**2)
                        return np.cos(k * R) / R

                    def ker_im(dz):
                        R = np.sqrt(dz**2 + a**2)
                        return -np.sin(k * R) / R

                    re, _ = quad(ker_re, -half, half, limit=40)
                    im, _ = quad(ker_im, -half, half, limit=40)
                    Z[m, n] = prefactor * (re + 1j * im) * delta
                else:
                    # Mutual impedance: full kernel
                    zm, zn = z[m], z[n]
                    half = delta / 2.0

                    def ker_re_mn(z_prime, _zm=zm):
                        dz = _zm - z_prime
                        R = np.sqrt(dz**2 + a**2)
                        return np.cos(k * R) / R

                    def ker_im_mn(z_prime, _zm=zm):
                        dz = _zm - z_prime
                        R = np.sqrt(dz**2 + a**2)
                        return -np.sin(k * R) / R

                    lo, hi = zn - half, zn + half
                    re, _ = quad(ker_re_mn, lo, hi, limit=40)
                    im, _ = quad(ker_im_mn, lo, hi, limit=40)
                    Z[m, n] = prefactor * (re + 1j * im) * delta

        self._Z_matrix = Z
        return Z

    # ------------------------------------------------------------------
    # Excitation vector
    # ------------------------------------------------------------------
    def excitation_vector(self, feed_segment: int = None,
                          V_gap: float = 1.0) -> np.ndarray:
        """Delta-gap voltage source excitation.

        V_m = V_gap if m == feed_segment else 0

        Parameters
        ----------
        feed_segment : int or None
            Index of the feed segment. Defaults to self.feed_seg.
        V_gap : float
            Gap voltage [V].

        Returns
        -------
        ndarray of complex128, shape (N,)
        """
        fs = feed_segment if feed_segment is not None else self.feed_seg
        V = np.zeros(self.N, dtype=complex)
        V[fs] = -V_gap
        return V

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """Solve Z·I = V for current distribution.

        Returns
        -------
        ndarray of complex128, shape (N,)
            Current at each segment centre [A].
        """
        if self._Z_matrix is None:
            self.build_impedance_matrix()
        V = self.excitation_vector()
        self._currents = np.linalg.solve(self._Z_matrix, V)
        return self._currents

    # ------------------------------------------------------------------
    # Input impedance
    # ------------------------------------------------------------------
    def input_impedance(self) -> complex:
        """Input impedance at feed point.

        Z_in = V_gap / I[feed_segment]

        Returns
        -------
        complex [Ω]
        """
        if self._currents is None:
            self.solve()
        I_feed = self._currents[self.feed_seg]
        if abs(I_feed) < 1e-30:
            raise RuntimeError("Feed current is near-zero; check geometry.")
        return complex(1.0 / I_feed)

    # ------------------------------------------------------------------
    # Far-field
    # ------------------------------------------------------------------
    def far_field(self, theta: np.ndarray,
                  phi: np.ndarray) -> tuple:
        """Compute far-field (E_theta, E_phi) from current distribution.

        E_θ = -j·η0·k/(4π) · e^{-jkr}/r · F_θ(θ,φ)
        F_θ = ∫ I(z')·sinθ·exp(j·k·z'·cosθ) dz'

        (Wire along z-axis: only E_theta component; E_phi = 0.)

        Parameters
        ----------
        theta : array_like [rad]
        phi   : array_like [rad]

        Returns
        -------
        tuple (E_theta, E_phi) as ndarray of complex, shape (Ntheta, Nphi)
        """
        if self._currents is None:
            self.solve()

        theta = np.asarray(theta, dtype=float)
        phi   = np.asarray(phi,   dtype=float)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')

        k = self.k
        prefactor = -1j * ETA0 * k / (4.0 * PI)

        # Accumulate far-field integral (r=1 m, so e^{-jkr}/r → e^{-jk}/1)
        F_theta = np.zeros_like(TH, dtype=complex)
        for i, (zn, In) in enumerate(zip(self.z_seg, self._currents)):
            phase = np.exp(1j * k * zn * np.cos(TH))
            F_theta += In * np.sin(TH) * phase * self.delta

        E_theta = prefactor * F_theta
        E_phi   = np.zeros_like(E_theta)
        return E_theta, E_phi

    def radiation_pattern(self, n_theta: int = 181,
                          n_phi: int = 361) -> "RadiationPattern":
        """Full 3-D radiation pattern from MoM currents.

        Returns
        -------
        RadiationPattern
        """
        from pylobe.analysis.radiation import RadiationPattern
        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        E_theta, E_phi = self.far_field(theta, phi)
        return RadiationPattern(E_theta, E_phi, theta, phi, self.freq)
