"""RadiationPattern container with analysis and cut methods."""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from pylobe.constants import ETA0, PI
from pylobe.analysis.metrics import (
    directivity, beamwidth_hpbw, side_lobe_level, front_to_back_ratio,
    axial_ratio as compute_axial_ratio,
)


class PlanecutResult(NamedTuple):
    """Result of a 1-D radiation pattern cut.

    Supports both attribute access and two-value unpacking::

        ep_dbi, angles_deg = pattern.e_plane_cut()
        # or
        result = pattern.e_plane_cut()
        result.gain_dbi    # absolute directivity in dBi
        result.theta_deg   # polar angle in degrees

    Attributes
    ----------
    gain_dbi : ndarray, shape (Ntheta,)
        Absolute directivity along the cut [dBi].
    theta_deg : ndarray, shape (Ntheta,)
        Polar angle [degrees], 0 to 180.
    """
    gain_dbi:  np.ndarray
    theta_deg: np.ndarray


@dataclass
class PatternSummary:
    """Typed summary of key radiation pattern metrics.

    Attributes
    ----------
    freq_ghz : float
        Analysis frequency [GHz].
    peak_gain_dbi : float
        Peak directivity / gain [dBi].
    theta_max_deg : float
        Elevation angle of peak [degrees].
    phi_max_deg : float
        Azimuthal angle of peak [degrees].
    hpbw_e_deg : float
        E-plane half-power beamwidth [degrees].
    hpbw_h_deg : float
        H-plane half-power beamwidth [degrees].
    sll_db : float
        Side-lobe level relative to main lobe [dB, negative].
    fbr_db : float
        Front-to-back ratio [dB].
    """
    freq_ghz:     float
    peak_gain_dbi: float
    theta_max_deg: float
    phi_max_deg:   float
    hpbw_e_deg:   float
    hpbw_h_deg:   float
    sll_db:       float
    fbr_db:       float

    # ------------------------------------------------------------------
    # Backward-compat aliases (old dict-key names)
    # ------------------------------------------------------------------
    @property
    def hpbw_e_plane_deg(self) -> float:
        """Alias for ``hpbw_e_deg`` (backward compat)."""
        return self.hpbw_e_deg

    @property
    def hpbw_h_plane_deg(self) -> float:
        """Alias for ``hpbw_h_deg`` (backward compat)."""
        return self.hpbw_h_deg

    @property
    def sll_e_plane_db(self) -> float:
        """Alias for ``sll_db`` (backward compat)."""
        return self.sll_db

    @property
    def polarisation(self) -> str:
        """Polarisation descriptor (always 'linear' for far-field cuts)."""
        return 'linear'

    def __getitem__(self, key: str):
        """Dict-style access: ``summary()['peak_gain_dbi']``."""
        _ALIASES = {
            'hpbw_e_plane_deg': 'hpbw_e_deg',
            'hpbw_h_plane_deg': 'hpbw_h_deg',
            'sll_e_plane_db':   'sll_db',
        }
        attr = _ALIASES.get(key, key)
        try:
            return getattr(self, attr)
        except AttributeError:
            raise KeyError(
                f"{key!r} is not a PatternSummary field. "
                f"Available: {list(self.__dataclass_fields__.keys())}"
            )

    def __str__(self) -> str:
        return (
            f"PatternSummary @ {self.freq_ghz:.3f} GHz\n"
            f"  Peak gain   : {self.peak_gain_dbi:.2f} dBi  "
            f"(th={self.theta_max_deg:.1f} deg, ph={self.phi_max_deg:.1f} deg)\n"
            f"  HPBW E/H    : {self.hpbw_e_deg:.1f} deg / {self.hpbw_h_deg:.1f} deg\n"
            f"  SLL         : {self.sll_db:.1f} dB\n"
            f"  F/B ratio   : {self.fbr_db:.1f} dB"
        )


class RadiationPattern:
    """Container for 2-D/3-D radiation pattern data.

    Stores complex E-field components on a (theta, phi) grid and
    provides analysis, cutting, and conversion methods.

    Parameters
    ----------
    E_theta : ndarray, shape (Ntheta, Nphi)
        Complex θ-component of far-field [V/m at r=1 m].
    E_phi : ndarray, shape (Ntheta, Nphi)
        Complex φ-component of far-field.
    theta : ndarray, shape (Ntheta,)
        Polar angles [rad], 0 to π.
    phi : ndarray, shape (Nphi,)
        Azimuthal angles [rad], 0 to 2π.
    freq : float
        Frequency [Hz].

    Raises
    ------
    ValueError
        If E_theta and E_phi shapes do not match (Ntheta, Nphi).
    """

    def __init__(self, E_theta: np.ndarray, E_phi: np.ndarray,
                 theta: np.ndarray, phi: np.ndarray, freq: float):
        E_theta = np.asarray(E_theta, dtype=complex)
        E_phi   = np.asarray(E_phi,   dtype=complex)
        theta   = np.asarray(theta,   dtype=float)
        phi     = np.asarray(phi,     dtype=float)

        # Shape validation — catch mismatched arrays before they produce
        # silent wrong directivity/HPBW values downstream.
        expected = (len(theta), len(phi))
        if E_theta.shape != expected:
            raise ValueError(
                f"E_theta shape {E_theta.shape} does not match "
                f"(len(theta), len(phi)) = {expected}."
            )
        if E_phi.shape != expected:
            raise ValueError(
                f"E_phi shape {E_phi.shape} does not match "
                f"(len(theta), len(phi)) = {expected}."
            )
        if np.max(np.abs(E_theta)) == 0 and np.max(np.abs(E_phi)) == 0:
            raise ValueError(
                "E_theta and E_phi are both zero everywhere. "
                "The radiation pattern has no radiated power."
            )
        if freq <= 0:
            raise ValueError(f"freq must be positive [Hz], got {freq!r}")

        self.E_theta = E_theta
        self.E_phi   = E_phi
        self.theta   = theta
        self.phi     = phi
        self.freq    = float(freq)

        # Lazy-evaluated cache
        self._D_2d  = None
        self._D_max = None
        self._D_dBi = None
        self._P_rad = None

    # ------------------------------------------------------------------
    # Power pattern
    # ------------------------------------------------------------------
    @property
    def total_power_pattern(self) -> np.ndarray:
        """Radiation intensity U(θ,φ) [W/sr at unit distance].

        U = (|E_θ|² + |E_φ|²) / (2·η0)
        """
        return (np.abs(self.E_theta)**2 + np.abs(self.E_phi)**2) / (2.0 * ETA0)

    @property
    def P_rad(self) -> float:
        """Total radiated power [W at unit distance, r=1 m]."""
        if self._P_rad is None:
            U = self.total_power_pattern
            sinT = np.sin(self.theta)
            integrand = U * sinT[:, np.newaxis]
            try:
                _trapz = np.trapezoid
            except AttributeError:
                _trapz = np.trapz
            inner = _trapz(integrand, self.phi, axis=1)
            self._P_rad = float(_trapz(inner, self.theta))
        return self._P_rad

    # ------------------------------------------------------------------
    # Directivity
    # ------------------------------------------------------------------
    def _compute_directivity(self):
        if self._D_2d is None:
            self._D_2d, self._D_max, self._D_dBi = directivity(
                self.E_theta, self.E_phi, self.theta, self.phi
            )

    @property
    def directivity_2d(self) -> np.ndarray:
        """2-D directivity map (linear), shape (Ntheta, Nphi)."""
        self._compute_directivity()
        return self._D_2d

    @property
    def peak_directivity_linear(self) -> float:
        """Peak directivity (linear)."""
        self._compute_directivity()
        return self._D_max

    @property
    def peak_directivity_dbi(self) -> float:
        """Peak directivity [dBi]."""
        self._compute_directivity()
        return self._D_dBi

    # ------------------------------------------------------------------
    # dBi conversion
    # ------------------------------------------------------------------
    def to_dbi(self) -> np.ndarray:
        """Return 2-D directivity in dBi."""
        D = self.directivity_2d
        return 10.0 * np.log10(np.clip(D, 1e-20, None))

    # ------------------------------------------------------------------
    # Phase pattern
    # ------------------------------------------------------------------
    def phase_pattern(self, component: str = 'theta') -> np.ndarray:
        """Return phase [degrees] of the specified E-field component.

        Parameters
        ----------
        component : str
            'theta' or 'phi'.

        Returns
        -------
        ndarray, shape (Ntheta, Nphi)
            Phase in degrees, range [-180, 180].
        """
        if component == 'theta':
            return np.rad2deg(np.angle(self.E_theta))
        elif component == 'phi':
            return np.rad2deg(np.angle(self.E_phi))
        else:
            raise ValueError(f"component must be 'theta' or 'phi', got {component!r}")

    # ------------------------------------------------------------------
    # Pattern cuts
    # ------------------------------------------------------------------
    def phi_cut(self, phi_deg: float) -> np.ndarray:
        """Extract 1-D pattern cut at given φ angle (linear power).

        Parameters
        ----------
        phi_deg : float
            Azimuthal angle [degrees].

        Returns
        -------
        ndarray, shape (Ntheta,)
        """
        phi_rad = np.deg2rad(phi_deg) % (2.0 * PI)
        idx = int(np.argmin(np.abs(self.phi - phi_rad)))
        U = (np.abs(self.E_theta[:, idx])**2
             + np.abs(self.E_phi[:, idx])**2) / (2.0 * ETA0)
        return U

    def e_plane_cut(self, phi_cut_deg: float = 0.0) -> PlanecutResult:
        """E-plane cut: returns ``(gain_dbi, theta_deg)`` NamedTuple.

        Parameters
        ----------
        phi_cut_deg : float
            Azimuthal angle of the cut [degrees]. Default 0° (E-plane).

        Returns
        -------
        PlanecutResult
            Named tuple with fields:

            * ``gain_dbi``  — absolute directivity [dBi], shape (Ntheta,)
            * ``theta_deg`` — polar angle [degrees], shape (Ntheta,)

        Examples
        --------
        >>> ep_dbi, angles = pattern.e_plane_cut()
        >>> # or attribute access:
        >>> result = pattern.e_plane_cut()
        >>> result.gain_dbi
        """
        phi_rad = np.deg2rad(phi_cut_deg) % (2.0 * PI)
        idx     = int(np.argmin(np.abs(self.phi - phi_rad)))
        gain_dbi = 10.0 * np.log10(np.clip(self.directivity_2d[:, idx], 1e-20, None))
        return PlanecutResult(gain_dbi=gain_dbi, theta_deg=np.rad2deg(self.theta))

    def h_plane_cut(self, phi_cut_deg: float = 90.0) -> PlanecutResult:
        """H-plane cut: returns ``(gain_dbi, theta_deg)`` NamedTuple.

        Parameters
        ----------
        phi_cut_deg : float
            Azimuthal angle of the cut [degrees]. Default 90° (H-plane).

        Returns
        -------
        PlanecutResult
            Same structure as :meth:`e_plane_cut`.
        """
        return self.e_plane_cut(phi_cut_deg)

    def theta_cut_db(self, phi_deg: float) -> np.ndarray:
        """1-D pattern cut in dB at given φ, normalised to 0 dB peak."""
        U = self.phi_cut(phi_deg)
        U_max = np.max(U) if np.max(U) > 0 else 1.0
        return 10.0 * np.log10(np.clip(U / U_max, 1e-10, None))

    # ------------------------------------------------------------------
    # Peak direction
    # ------------------------------------------------------------------
    def peak_direction(self) -> tuple:
        """Angular location of main beam peak.

        Returns
        -------
        tuple (theta_max_deg, phi_max_deg)
        """
        D = self.directivity_2d
        idx = np.unravel_index(np.argmax(D), D.shape)
        return (
            float(np.rad2deg(self.theta[idx[0]])),
            float(np.rad2deg(self.phi[idx[1]])),
        )

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    def summary(self) -> PatternSummary:
        """Compute and return all key performance metrics.

        Returns
        -------
        PatternSummary
            Typed dataclass with all metrics as named attributes.

        Examples
        --------
        >>> from pylobe import HalfWaveDipole, DipoleSolver
        >>> d = HalfWaveDipole(freq=300e6)
        >>> s = DipoleSolver(d, 300e6)
        >>> rp = s.radiation_pattern()
        >>> sm = rp.summary()
        >>> sm.peak_gain_dbi > 2.0
        True
        """
        theta_deg = np.rad2deg(self.theta)
        e_cut = self.phi_cut(0.0)    # linear power, used for HPBW
        h_cut = self.phi_cut(90.0)   # linear power, used for HPBW
        e_db  = self.theta_cut_db(0.0)
        h_db  = self.theta_cut_db(90.0)

        hpbw_e = beamwidth_hpbw(e_cut, theta_deg)
        hpbw_h = beamwidth_hpbw(h_cut, theta_deg)
        sll    = side_lobe_level(e_db, theta_deg)
        fbr    = front_to_back_ratio(e_db, theta_deg)
        th_max, ph_max = self.peak_direction()

        return PatternSummary(
            freq_ghz=self.freq / 1e9,
            peak_gain_dbi=self.peak_directivity_dbi,
            theta_max_deg=th_max,
            phi_max_deg=ph_max,
            hpbw_e_deg=hpbw_e,
            hpbw_h_deg=hpbw_h,
            sll_db=sll,
            fbr_db=fbr,
        )

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"RadiationPattern("
            f"f={self.freq/1e9:.3f} GHz, "
            f"D={s.peak_gain_dbi:.2f} dBi, "
            f"HPBW_E={s.hpbw_e_deg:.1f} deg)"
        )
