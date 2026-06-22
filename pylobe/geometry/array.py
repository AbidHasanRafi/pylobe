"""Antenna array geometry and array factor calculations."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry
from pylobe.constants import C0, PI
from pylobe.utils.validation import check_frequency, check_positive, check_array_element_count


class LinearArray:
    """Uniform Linear Array (ULA) along the z-axis.

    Array Factor:
        AF(θ) = Σ_{n=0}^{N-1} a_n · exp(j·n·ψ)
        ψ = k·d·cos(θ) + β

    Parameters
    ----------
    element : AntennaGeometry
        Single antenna element.
    N : int
        Number of elements.
    d : float
        Element spacing [m].
    beta : float
        Progressive phase shift [rad].
        Broadside: β = 0
        Endfire:   β = ±k·d
        Scan to θ0: β = -k·d·cos(θ0)
    amplitudes : array_like of shape (N,) or None
        Element excitation amplitudes. Default: uniform (all 1.0).
    """

    def __init__(self, element: AntennaGeometry, N: int, d: float,
                 beta: float = 0.0, amplitudes=None):
        check_array_element_count(N, "N")
        check_positive(d, "d (element spacing)")
        self.element = element
        self.N = N
        self.d = d
        self.beta = beta
        self.amplitudes = (
            np.ones(N) if amplitudes is None else np.asarray(amplitudes, dtype=float)
        )
        if len(self.amplitudes) != N:
            raise ValueError(f"amplitudes length {len(self.amplitudes)} != N={N}")

    def array_factor(self, theta: np.ndarray, freq: float) -> np.ndarray:
        """Compute normalised |AF(θ)| at given polar angles.

        Uses exact complex summation over all elements.

        Parameters
        ----------
        theta : array_like
            Polar angles [rad].
        freq : float
            Frequency [Hz].

        Returns
        -------
        ndarray
            Normalised |AF| in linear scale, same shape as theta.
        """
        theta = np.asarray(theta)
        k = 2.0 * PI * freq / C0
        psi = k * self.d * np.cos(theta)[..., np.newaxis] + self.beta  # (..., 1)
        n_arr = np.arange(self.N)
        AF = np.sum(self.amplitudes * np.exp(1j * n_arr * psi), axis=-1)
        max_val = np.max(np.abs(AF))
        return np.abs(AF) / (max_val if max_val > 0 else 1.0)

    def array_factor_db(self, theta: np.ndarray, freq: float) -> np.ndarray:
        """Array factor in dB (20·log10)."""
        af = self.array_factor(theta, freq)
        return 20.0 * np.log10(np.clip(af, 1e-10, None))

    def scan_to(self, theta0_deg: float, freq: float):
        """Set progressive phase shift to steer main beam to theta0.

        β = -k·d·cos(θ0)

        Parameters
        ----------
        theta0_deg : float
            Target scan angle [degrees].
        freq : float
            Operating frequency [Hz].
        """
        k = 2.0 * PI * freq / C0
        self.beta = -k * self.d * np.cos(np.deg2rad(theta0_deg))

    def chebyshev_weights(self, sll_db: float):
        """Compute Dolph-Chebyshev amplitude weights.

        Parameters
        ----------
        sll_db : float
            Desired side-lobe level below main lobe [dB], e.g. 25 for -25 dB.
        """
        N = self.N
        R0 = 10.0 ** (sll_db / 20.0)   # linear side-lobe ratio

        # Chebyshev order = N-1
        m = N - 1
        x0 = np.cosh(np.arccosh(R0) / m)

        # Compute weights via IDFT of Chebyshev polynomial
        weights = np.zeros(N)
        for i in range(N):
            for k in range(m + 1):
                z = x0 * np.cos(PI * k / m)
                T = self._cheby_eval(m, z)
                weights[i] += T * np.cos(2.0 * PI * k * (i - m / 2.0) / m)
        weights = np.abs(weights)
        weights /= weights.max()
        self.amplitudes = weights
        return weights

    @staticmethod
    def _cheby_eval(n: int, x: float) -> float:
        """Evaluate Chebyshev polynomial T_n(x)."""
        if abs(x) <= 1.0:
            return np.cos(n * np.arccos(np.clip(x, -1.0, 1.0)))
        return np.cosh(n * np.arccosh(abs(x))) * np.sign(x) ** n

    def taylor_weights(self, sll_db: float, nbar: int = 5):
        """Compute Taylor line-source amplitude taper.

        Parameters
        ----------
        sll_db : float
            Side-lobe level [dB] (positive number, e.g. 25 for -25 dB).
        nbar : int
            Number of equal-level side lobes adjacent to main lobe.
        """
        N = self.N
        A = np.arccosh(10.0 ** (sll_db / 20.0)) / PI
        sigma = nbar / np.sqrt(A**2 + (nbar - 0.5)**2)

        n_idx = np.arange(1, nbar)
        Fn = np.ones(nbar - 1)
        for n in n_idx:
            num = 1.0
            den = 1.0
            for m in n_idx:
                if m != n:
                    num *= 1.0 - (n / sigma)**2 / (A**2 + (m - 0.5)**2)
                    den *= 1.0 - (n**2) / (m**2) if m != n else 1.0
            Fn[n - 1] = num

        i_arr = np.arange(N)
        weights = np.ones(N)
        for n in n_idx:
            weights += 2.0 * Fn[n - 1] * np.cos(2.0 * PI * n * (i_arr - (N-1)/2.0) / N)
        weights = np.abs(weights)
        weights /= weights.max()
        self.amplitudes = weights
        return weights

    def positions(self) -> np.ndarray:
        """Cartesian positions of array elements along z-axis [m].

        Returns ndarray, shape (N, 3).
        """
        return np.column_stack([
            np.zeros(self.N),
            np.zeros(self.N),
            np.arange(self.N) * self.d,
        ])


class PlanarArray:
    """M×N Planar Array.

    Array Factor:
        AF(θ,φ) = AFx(θ,φ) × AFy(θ,φ)
    where AFx and AFy are the x- and y-direction linear array factors.

    Parameters
    ----------
    element : AntennaGeometry
        Single antenna element.
    M : int
        Number of elements along x.
    N : int
        Number of elements along y.
    dx : float
        Element spacing along x [m].
    dy : float
        Element spacing along y [m].
    beta_x : float
        Progressive phase shift along x [rad].
    beta_y : float
        Progressive phase shift along y [rad].
    """

    def __init__(self, element: AntennaGeometry, M: int, N: int,
                 dx: float, dy: float,
                 beta_x: float = 0.0, beta_y: float = 0.0):
        self.element = element
        self.M = M
        self.N = N
        self.dx = dx
        self.dy = dy
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.amplitudes = np.ones((M, N))

    def array_factor(self, theta: np.ndarray, phi: np.ndarray,
                     freq: float) -> np.ndarray:
        """Compute normalised |AF(θ,φ)| on a meshgrid.

        Parameters
        ----------
        theta : ndarray, shape (Nt,)
            Polar angles [rad].
        phi : ndarray, shape (Np,)
            Azimuthal angles [rad].
        freq : float
            Frequency [Hz].

        Returns
        -------
        ndarray, shape (Nt, Np)
        """
        from pylobe.solver.analytical.array_factor import array_factor_2d
        return array_factor_2d(theta, phi, freq, self.M, self.N,
                               self.dx, self.dy, self.beta_x, self.beta_y)

    def scan_to(self, theta0_deg: float, phi0_deg: float, freq: float):
        """Steer main beam to (θ0, φ0).

        β_x = -k·dx·sin(θ0)·cos(φ0)
        β_y = -k·dy·sin(θ0)·sin(φ0)
        """
        k = 2.0 * PI * freq / C0
        t0, p0 = np.deg2rad(theta0_deg), np.deg2rad(phi0_deg)
        self.beta_x = -k * self.dx * np.sin(t0) * np.cos(p0)
        self.beta_y = -k * self.dy * np.sin(t0) * np.sin(p0)


class CircularArray:
    """Uniformly-spaced circular array of N elements on radius R.

    Array Factor:
        AF(θ,φ) = Σ_n a_n · exp(j·k·R·sin(θ)·cos(φ - 2πn/N) + j·α_n)

    Parameters
    ----------
    element : AntennaGeometry
        Single antenna element.
    N : int
        Number of elements.
    R : float
        Array radius [m].
    alpha : array_like of shape (N,) or None
        Element phase excitations [rad]. Default: uniform 0.
    amplitudes : array_like of shape (N,) or None
        Element amplitudes. Default: uniform 1.
    """

    def __init__(self, element: AntennaGeometry, N: int, R: float,
                 alpha=None, amplitudes=None):
        self.element = element
        self.N = N
        self.R = R
        self.alpha = np.zeros(N) if alpha is None else np.asarray(alpha)
        self.amplitudes = np.ones(N) if amplitudes is None else np.asarray(amplitudes)

    def array_factor(self, theta: np.ndarray, phi: np.ndarray,
                     freq: float) -> np.ndarray:
        """Compute |AF(θ,φ)| on a meshgrid.

        Parameters
        ----------
        theta : ndarray, shape (Nt,)
        phi   : ndarray, shape (Np,)
        freq  : float

        Returns
        -------
        ndarray, shape (Nt, Np)
        """
        k = 2.0 * PI * freq / C0
        th, ph = np.meshgrid(theta, phi, indexing='ij')  # (Nt, Np)
        AF = np.zeros_like(th, dtype=complex)
        for n in range(self.N):
            phi_n = 2.0 * PI * n / self.N
            AF += self.amplitudes[n] * np.exp(
                1j * (k * self.R * np.sin(th) * np.cos(ph - phi_n) + self.alpha[n])
            )
        max_v = np.max(np.abs(AF))
        return np.abs(AF) / (max_v if max_v > 0 else 1.0)

    def positions(self) -> np.ndarray:
        """Cartesian element positions [m], shape (N, 3)."""
        phi_n = 2.0 * PI * np.arange(self.N) / self.N
        return np.column_stack([
            self.R * np.cos(phi_n),
            self.R * np.sin(phi_n),
            np.zeros(self.N),
        ])
