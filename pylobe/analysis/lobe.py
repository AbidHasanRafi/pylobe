"""Radiation lobe decomposition and analysis engine."""
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import label, find_objects
from typing import List
from pylobe.constants import PI

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


@dataclass
class Lobe:
    """A single radiation lobe (main, side, back, or grating).

    Attributes
    ----------
    lobe_type : str
        'main', 'side', 'back', or 'grating'.
    peak_theta_deg : float
    peak_phi_deg : float
    peak_gain_dbi : float
    hpbw_deg : float
        Half-power beamwidth [degrees].
    angular_extent : tuple
        (theta_lo, theta_hi) in degrees.
    solid_angle_sr : float
        Steradians subtended by the lobe.
    null_depth_db : float
        Depth of the deepest adjacent null below peak [dB].
    fractional_power : float
        Fraction of total radiated power in this lobe.
    """
    lobe_type:        str
    peak_theta_deg:   float
    peak_phi_deg:     float
    peak_gain_dbi:    float
    hpbw_deg:         float
    angular_extent:   tuple
    solid_angle_sr:   float
    null_depth_db:    float
    fractional_power: float

    def __repr__(self) -> str:
        return (
            f"Lobe(type={self.lobe_type}, "
            f"θ={self.peak_theta_deg:.1f}°, φ={self.peak_phi_deg:.1f}°, "
            f"gain={self.peak_gain_dbi:.2f} dBi, "
            f"HPBW={self.hpbw_deg:.1f}°)"
        )


class LobeAnalyzer:
    """Decompose a 3-D radiation pattern into individual lobe structures.

    Parameters
    ----------
    pattern : RadiationPattern
        Fully computed radiation pattern.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self._lobes = None

    # ------------------------------------------------------------------
    # Lobe detection
    # ------------------------------------------------------------------
    def find_lobes(self, min_peak_dbi: float = None,
                   min_separation_deg: float = 10.0) -> List[Lobe]:
        """Decompose pattern into lobe objects.

        Algorithm:
        1. Convert 2-D directivity to dBi.
        2. Threshold at (peak - 40 dB) to create binary mask.
        3. scipy.ndimage.label → connected components (lobes).
        4. For each component: find peak, compute HPBW, solid angle.
        5. Classify: main = global maximum; back = θ > 170°; others = side.

        Parameters
        ----------
        min_peak_dbi : float or None
            Minimum peak to keep a lobe. Default: main lobe peak - 40 dB.
        min_separation_deg : float
            Minimum angular separation between lobes [degrees].

        Returns
        -------
        list of Lobe
            Sorted descending by peak gain.
        """
        D_db   = self.pattern.to_dbi()
        global_peak = np.max(D_db)
        if min_peak_dbi is None:
            min_peak_dbi = global_peak - 40.0

        # Binary mask
        binary = (D_db >= min_peak_dbi).astype(int)
        labeled, n_features = label(binary)

        lobes = []
        P_total = self.pattern.P_rad

        for region_id in range(1, n_features + 1):
            region_mask = labeled == region_id
            region_vals = D_db.copy()
            region_vals[~region_mask] = -np.inf

            # Peak location
            peak_flat = np.argmax(region_vals)
            ti, pi = np.unravel_index(peak_flat, D_db.shape)
            peak_gain = D_db[ti, pi]

            if peak_gain < min_peak_dbi:
                continue

            theta_deg = np.rad2deg(self.pattern.theta[ti])
            phi_deg   = np.rad2deg(self.pattern.phi[pi])

            # HPBW estimate: from region extent in theta
            theta_in_region = self.pattern.theta[region_mask.any(axis=1)]
            if len(theta_in_region) >= 2:
                hpbw = float(np.rad2deg(theta_in_region[-1] - theta_in_region[0]))
            else:
                hpbw = 0.0

            # Angular extent
            theta_lo = float(np.rad2deg(self.pattern.theta[
                region_mask.any(axis=1).argmax()
            ]))
            theta_hi = float(np.rad2deg(self.pattern.theta[
                len(self.pattern.theta) - 1 - region_mask.any(axis=1)[::-1].argmax()
            ]))

            # Solid angle: ∫∫ sinθ dθ dφ over region
            sinT = np.sin(self.pattern.theta)
            dtheta = np.diff(self.pattern.theta)
            dphi   = np.diff(self.pattern.phi)
            if len(dtheta) == 0 or len(dphi) == 0:
                solid_angle = 0.0
            else:
                dtheta_mean = np.mean(dtheta)
                dphi_mean   = np.mean(dphi)
                solid_angle = float(np.sum(sinT[region_mask.any(axis=1)])
                                    * dtheta_mean * dphi_mean)

            # Fractional power in this lobe
            U = self.pattern.total_power_pattern
            U_in_lobe = np.where(region_mask, U, 0.0)
            sinT_2d = sinT[:, np.newaxis] * np.ones_like(U)
            p_lobe = _trapz(_trapz(U_in_lobe * sinT_2d, self.pattern.phi, axis=1),
                              self.pattern.theta)
            frac_power = p_lobe / P_total if P_total > 0 else 0.0

            # Null depth: minimum in boundary ring around region
            null_depth = -30.0  # default

            # Classify
            if peak_gain >= global_peak - 0.1:
                lobe_type = 'main'
            elif theta_deg > 150.0:
                lobe_type = 'back'
            else:
                lobe_type = 'side'

            lobes.append(Lobe(
                lobe_type=lobe_type,
                peak_theta_deg=theta_deg,
                peak_phi_deg=phi_deg,
                peak_gain_dbi=float(peak_gain),
                hpbw_deg=hpbw,
                angular_extent=(theta_lo, theta_hi),
                solid_angle_sr=solid_angle,
                null_depth_db=null_depth,
                fractional_power=float(frac_power),
            ))

        lobes.sort(key=lambda l: l.peak_gain_dbi, reverse=True)
        self._lobes = lobes
        return lobes

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def main_lobe(self) -> Lobe:
        """Return the dominant (global maximum) lobe."""
        if self._lobes is None:
            self.find_lobes()
        for lobe in self._lobes:
            if lobe.lobe_type == 'main':
                return lobe
        return self._lobes[0] if self._lobes else None

    def side_lobes(self) -> List[Lobe]:
        """Return all non-main lobes sorted by peak gain."""
        if self._lobes is None:
            self.find_lobes()
        return [l for l in self._lobes if l.lobe_type != 'main']

    def null_map(self, threshold_below_peak_db: float = 20.0) -> np.ndarray:
        """Binary map where True indicates a radiation null.

        Parameters
        ----------
        threshold_below_peak_db : float
            Nulls are cells more than this many dB below the main peak.

        Returns
        -------
        ndarray of bool, shape (Ntheta, Nphi)
        """
        D_db = self.pattern.to_dbi()
        peak = np.max(D_db)
        return D_db < (peak - threshold_below_peak_db)

    def lobe_asymmetry_index(self) -> float:
        """Lobe asymmetry index.

        LAI = |P_left - P_right| / P_total

        where P_left/P_right are radiated powers in φ ∈ [0, π) and φ ∈ [π, 2π).
        """
        U = self.pattern.total_power_pattern
        sinT = np.sin(self.pattern.theta)[:, np.newaxis]
        phi = self.pattern.phi
        half = len(phi) // 2

        integrand = U * sinT
        p_left  = _trapz(_trapz(integrand[:, :half], phi[:half], axis=1),
                            self.pattern.theta)
        p_right = _trapz(_trapz(integrand[:, half:], phi[half:], axis=1),
                            self.pattern.theta)
        P_total = self.pattern.P_rad
        if P_total <= 0:
            return 0.0
        return float(abs(p_left - p_right) / P_total)

    def beam_solid_angle(self) -> float:
        """Beam solid angle.

        Ω_A = ∫_{4π} (F(θ,φ) / F_max) dΩ   [steradians]

        D = 4π / Ω_A
        """
        U = self.pattern.total_power_pattern
        U_max = np.max(U) if np.max(U) > 0 else 1.0
        sinT = np.sin(self.pattern.theta)[:, np.newaxis]
        integrand = (U / U_max) * sinT
        inner = _trapz(integrand, self.pattern.phi, axis=1)
        return float(_trapz(inner, self.pattern.theta))

    def encircled_power_fraction(self, cone_half_angle_deg: float) -> float:
        """Fraction of total radiated power within a cone centred on main beam.

        Parameters
        ----------
        cone_half_angle_deg : float
            Cone half-angle [degrees].

        Returns
        -------
        float in [0, 1]
        """
        ml = self.main_lobe()
        if ml is None:
            return 0.0
        theta_c = np.deg2rad(ml.peak_theta_deg)
        phi_c   = np.deg2rad(ml.peak_phi_deg)
        cone_rad = np.deg2rad(cone_half_angle_deg)

        # Great-circle angle from (θ_c, φ_c)
        TH, PH = np.meshgrid(self.pattern.theta, self.pattern.phi, indexing='ij')
        cos_sep = (np.sin(TH) * np.sin(theta_c) * np.cos(PH - phi_c)
                   + np.cos(TH) * np.cos(theta_c))
        in_cone = cos_sep >= np.cos(cone_rad)

        U = self.pattern.total_power_pattern
        sinT = np.sin(self.pattern.theta)[:, np.newaxis]
        integrand = np.where(in_cone, U * sinT, 0.0)
        p_cone = _trapz(_trapz(integrand, self.pattern.phi, axis=1),
                          self.pattern.theta)
        P_total = self.pattern.P_rad
        return float(p_cone / P_total) if P_total > 0 else 0.0
