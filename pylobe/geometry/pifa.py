"""Planar Inverted-F Antenna (PIFA) geometry."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, FR4, COPPER, PEC
from pylobe.constants import C0, PI
from pylobe.utils.validation import check_frequency, check_positive


class PIFA(AntennaGeometry):
    """Planar Inverted-F Antenna (PIFA).

    The PIFA is a compact resonant antenna widely used in mobile handsets,
    IoT modules, and wearables.  It is a quarter-wavelength patch with one
    end shorted to the ground plane via a shorting strip/pin.

    Resonance condition:
        L_eff = L + W_short/4 ≈ λ_g / 4
    where λ_g = λ₀ / √ε_eff and ε_eff ≈ (ε_r + 1) / 2 for a thin patch
    over a ground plane.

    Key feature: the shorting pin moves the feed impedance from 0 Ω
    (at the short) to ~50 Ω at a distance x_f from the shorting wall.

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    substrate_material : Material or None
        Substrate material. Defaults to FR4.
    h : float
        Height above ground plane [m].  Defaults to λ/50.
    L : float or None
        PIFA plate length [m] (x-direction). If None, auto-computed.
    W : float or None
        PIFA plate width [m]  (y-direction). Defaults to L × 0.8.
    W_short : float or None
        Shorting strip width [m]. Defaults to W/3.
    x_feed : float or None
        Feed point x-offset from shorting wall [m].  Defaults to L × 0.2.
    conductor_material : Material or None
        Patch + shorting strip conductor. Defaults to COPPER.
    ground_material : Material or None
        Ground plane conductor. Defaults to PEC.

    Attributes
    ----------
    L, W : float
        PIFA plate dimensions [m].
    h : float
        Height above ground [m].
    eps_eff : float
        Effective permittivity.
    resonant_freq_approx : float
        Approximate resonant frequency from resonance condition [Hz].
    """

    def __init__(self, freq: float,
                 substrate_material: Material = None,
                 h: float = None,
                 L: float = None, W: float = None,
                 W_short: float = None,
                 x_feed: float = None,
                 conductor_material: Material = None,
                 ground_material: Material = None):
        check_frequency(freq)
        lambda0 = C0 / freq

        if substrate_material is None:
            substrate_material = FR4

        eps_r = substrate_material.eps_r

        if h is None:
            h = lambda0 / 50.0
        check_positive(h, "h")

        super().__init__(
            name="PIFA",
            material=substrate_material,
            freq_design=freq,
        )
        self.freq               = freq
        self.lambda0            = lambda0
        self.substrate_material = substrate_material
        self.h                  = h
        self.eps_r              = eps_r
        self.conductor_material = conductor_material if conductor_material is not None else COPPER
        self.ground_material    = ground_material if ground_material is not None else PEC

        # Effective permittivity (simplified)
        self.eps_eff = (eps_r + 1.0) / 2.0
        lambda_g     = lambda0 / np.sqrt(self.eps_eff)

        # PIFA resonance condition: L + W_short/4 = λ_g/4
        # Choose L and W_short satisfying this
        if L is None:
            # Default: use 80% of the quarter-wave length for L, 20% for W_short effect
            L = 0.75 * lambda_g / 4.0
        check_positive(L, "L")
        self.L = L

        if W is None:
            W = L * 0.8
        check_positive(W, "W")
        self.W = W

        if W_short is None:
            W_short = W / 3.0
        check_positive(W_short, "W_short")
        self.W_short = W_short

        if x_feed is None:
            x_feed = L * 0.20
        check_positive(x_feed, "x_feed")
        self.x_feed = x_feed

        # Resonant frequency estimate: L_eff = L + W_short/4 = λ_g/4
        L_eff = L + W_short / 4.0
        self.resonant_freq_approx = float(
            C0 / (4.0 * L_eff * np.sqrt(self.eps_eff))
        )

        # Approximate gain (PIFA over infinite ground): ~2–4 dBi
        self.gain_approx_dbi = 3.0

        # Build wireframe: PIFA plate + shorting strip + ground plane outline
        # Plate at height h, x from 0 to L, y from 0 to W
        plate = np.array([
            [0.0, 0.0, h],
            [L,   0.0, h],
            [L,   W,   h],
            [0.0, W,   h],
        ])
        # Shorting strip at x=0
        short = np.array([
            [0.0, 0.0, h],
            [0.0, 0.0, 0.0],
            [0.0, W_short, 0.0],
            [0.0, W_short, h],
        ])
        # Ground plane (partial, under patch)
        gnd = np.array([
            [0.0, 0.0,       0.0],
            [L * 1.5, 0.0,   0.0],
            [L * 1.5, W*1.5, 0.0],
            [0.0, W*1.5,     0.0],
        ])
        self.vertices = np.vstack([plate, short, gnd])
        self.edges = [
            (0,1),(1,2),(2,3),(3,0),    # plate outline
            (4,5),(5,6),(6,7),(7,4),    # shorting strip
            (8,9),(9,10),(10,11),(11,8) # ground outline
        ]
        self.feed_point = np.array([x_feed, W / 2.0, 0.0])

    def __repr__(self) -> str:
        return (
            f"PIFA\n"
            f"  Frequency       : {self.freq/1e9:.3f} GHz\n"
            f"  Substrate       : {self.substrate_material.name}"
            f"  (εᵣ = {self.eps_r})\n"
            f"  Height h        : {self.h*1e3:.2f} mm\n"
            f"  Plate L × W     : {self.L*1e3:.2f} × {self.W*1e3:.2f} mm\n"
            f"  Shorting strip  : {self.W_short*1e3:.2f} mm wide\n"
            f"  Feed x offset   : {self.x_feed*1e3:.2f} mm\n"
            f"  f_res (approx)  : {self.resonant_freq_approx/1e9:.3f} GHz\n"
            f"  Gain (approx)   : {self.gain_approx_dbi:.1f} dBi\n"
        )
