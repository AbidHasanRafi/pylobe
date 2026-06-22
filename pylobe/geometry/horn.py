"""Horn antenna geometry classes."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER
from pylobe.constants import C0, PI
from pylobe.utils.validation import check_frequency, check_positive


class PyramidalHorn(AntennaGeometry):
    """Standard-gain pyramidal horn antenna.

    A pyramidal horn flares in both the E-plane (b → b_ap) and H-plane
    (a → a_ap) from a rectangular waveguide feed.

    Gain formula (Balanis, Antenna Theory, 4th Ed., Eq. 13-53):
        G = η_ap · (4π / λ²) · (a_ap × b_ap)
    with aperture efficiency η_ap ≈ 0.51 for the optimum horn.

    Design rules (optimum horn: maximise G for fixed length R):
        a_ap = sqrt(3λ R_H)  [H-plane]
        b_ap = sqrt(2λ R_E)  [E-plane]

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    a_wg : float
        Waveguide broad-wall dimension [m]  (e.g. WR-90: 22.86 mm).
    b_wg : float
        Waveguide narrow-wall dimension [m] (e.g. WR-90: 10.16 mm).
    a_ap : float or None
        Aperture width [m] (H-plane).  If None, computed for optimum gain.
    b_ap : float or None
        Aperture height [m] (E-plane).  If None, computed for optimum gain.
    length : float or None
        Axial length of the horn [m].  If None, defaults to 3λ.
    conductor_material : Material or None
        Horn wall conductor. Defaults to COPPER.

    Attributes
    ----------
    gain_approx_dbi : float
        Approximate gain [dBi] from aperture theory.
    hpbw_e_deg, hpbw_h_deg : float
        Approximate E- and H-plane half-power beamwidths [°].
    """

    # Standard WR waveguide inner dimensions [m]  (broad × narrow)
    WG_STANDARDS = {
        'WR-650': (165.1e-3, 82.55e-3),
        'WR-430': (109.2e-3, 54.61e-3),
        'WR-284': ( 72.14e-3, 34.04e-3),
        'WR-187': ( 47.55e-3, 22.15e-3),
        'WR-137': ( 34.85e-3, 15.80e-3),
        'WR-90' : ( 22.86e-3, 10.16e-3),
        'WR-62' : ( 15.80e-3,  7.90e-3),
        'WR-42' : ( 10.67e-3,  4.32e-3),
        'WR-28' : (  7.11e-3,  3.56e-3),
    }

    def __init__(self, freq: float,
                 a_wg: float = None, b_wg: float = None,
                 a_ap: float = None, b_ap: float = None,
                 length: float = None,
                 conductor_material: Material = None):
        check_frequency(freq)
        lambda0 = C0 / freq

        # Auto-select waveguide if not specified (pick the one with cut-off just below freq)
        if a_wg is None or b_wg is None:
            a_wg, b_wg = self._auto_waveguide(freq)

        check_positive(a_wg, "a_wg")
        check_positive(b_wg, "b_wg")
        if a_ap is not None:
            check_positive(a_ap, "a_ap")
        if b_ap is not None:
            check_positive(b_ap, "b_ap")

        super().__init__(
            name="PyramidalHorn",
            material=AIR,
            freq_design=freq,
        )
        self.freq   = freq
        self.lambda0 = lambda0
        self.a_wg   = a_wg
        self.b_wg   = b_wg

        self.length = length if length is not None else 3.0 * lambda0

        # Optimum aperture from square-law phase error criterion
        if a_ap is None:
            a_ap = np.sqrt(3.0 * lambda0 * self.length)
        if b_ap is None:
            b_ap = np.sqrt(2.0 * lambda0 * self.length)

        # Enforce aperture ≥ waveguide
        self.a_ap = max(a_ap, a_wg)
        self.b_ap = max(b_ap, b_wg)

        self.conductor_material = (conductor_material if conductor_material is not None
                                   else COPPER)

        # Approximate gain with aperture efficiency η_ap ≈ 0.51
        A_ap = self.a_ap * self.b_ap
        self.gain_approx_dbi = float(
            10 * np.log10(0.51 * 4 * PI * A_ap / lambda0 ** 2)
        )

        # Approximate HPBW (half-power beamwidths)
        self.hpbw_e_deg = float(np.degrees(0.886 * lambda0 / self.b_ap))
        self.hpbw_h_deg = float(np.degrees(1.189 * lambda0 / self.a_ap))

        # Build wireframe vertices (rectangular frustum outline)
        L   = self.length
        aw2, bw2 = a_wg / 2, b_wg / 2
        aa2, ba2 = self.a_ap / 2, self.b_ap / 2

        # Input waveguide aperture at z=0, output at z=L
        feed_corners = np.array([
            [-aw2, -bw2, 0.0], [ aw2, -bw2, 0.0],
            [ aw2,  bw2, 0.0], [-aw2,  bw2, 0.0],
        ])
        ap_corners = np.array([
            [-aa2, -ba2, L], [ aa2, -ba2, L],
            [ aa2,  ba2, L], [-aa2,  ba2, L],
        ])
        self.vertices = np.vstack([feed_corners, ap_corners])
        # Edges: feed rect, aperture rect, 4 flare edges
        self.edges = (
            [(0,1),(1,2),(2,3),(3,0),           # feed
             (4,5),(5,6),(6,7),(7,4),            # aperture
             (0,4),(1,5),(2,6),(3,7)]            # flare
        )
        self.feed_point = np.array([0.0, 0.0, 0.0])

    @staticmethod
    def _auto_waveguide(freq: float):
        """Select waveguide whose TE10 cut-off is below freq with headroom."""
        WG_STANDARDS = PyramidalHorn.WG_STANDARDS
        C0_local = 3e8
        best = None
        best_margin = float('inf')
        for name, (a, b) in WG_STANDARDS.items():
            f_co = C0_local / (2 * a)  # TE10 cut-off
            if f_co < freq:
                margin = freq - f_co
                if margin < best_margin:
                    best_margin = margin
                    best = (a, b)
        if best is None:
            # Fall back to WR-90 (2.4-18 GHz capable)
            best = WG_STANDARDS['WR-90']
        return best

    def __repr__(self) -> str:
        return (
            f"PyramidalHorn\n"
            f"  Frequency       : {self.freq / 1e9:.3f} GHz\n"
            f"  Waveguide (a×b) : {self.a_wg*1e3:.2f} × {self.b_wg*1e3:.2f} mm\n"
            f"  Aperture (a×b)  : {self.a_ap*1e3:.2f} × {self.b_ap*1e3:.2f} mm\n"
            f"  Length          : {self.length*1e3:.1f} mm\n"
            f"  Approx gain     : {self.gain_approx_dbi:.1f} dBi\n"
            f"  HPBW (E/H)      : {self.hpbw_e_deg:.1f}° / {self.hpbw_h_deg:.1f}°\n"
        )
