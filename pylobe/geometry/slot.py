"""Slot and tapered-slot (Vivaldi) antenna geometry classes."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR
from pylobe.constants import C0, PI, ETA0
from pylobe.utils.validation import check_frequency, check_positive, check_length_factor


class SlotAntenna(AntennaGeometry):
    """Resonant slot in an infinite conducting ground plane.

    Babinet's principle:  Z_slot × Z_dipole = (η0/2)² = 35476.17 Ω²
    Resonant slot length ≈ 0.47λ.
    Z_slot ≈ 363 Ω for thin slot.

    Parameters
    ----------
    freq : float
        Design frequency [Hz].
    length_factor : float
        Slot length as fraction of λ. Default 0.47 for near-resonance.
    width : float or None
        Slot width [m]. Defaults to λ/100.
    ground_size : float or None
        Square ground plane side length [m]. Defaults to λ.
    """

    # Babinet constant Z_slot * Z_dipole = (η0/2)²
    BABINET_CONST = (ETA0 / 2.0) ** 2   # ≈ 35476 Ω²

    def __init__(self, freq: float, length_factor: float = 0.47,
                 width: float = None, ground_size: float = None):
        check_frequency(freq)
        check_length_factor(length_factor)
        if width is not None:
            check_positive(width, "width")
        if ground_size is not None:
            check_positive(ground_size, "ground_size")
        material = Material(name="PEC_ground", conductivity=1e7)
        super().__init__(
            name="SlotAntenna",
            material=material,
            freq_design=freq,
        )
        lambda0 = C0 / freq
        self.slot_length = length_factor * lambda0
        self.slot_width  = width if width is not None else lambda0 / 100.0
        self.ground_size = ground_size if ground_size is not None else lambda0

        # Impedance via Babinet's principle
        # Z_dipole ≈ 73 + j42.5 for half-wave → Z_slot = BABINET_CONST / Z_dipole*
        Z_dipole_conj = complex(73.1, -42.5)
        self.Z_slot = self.BABINET_CONST / Z_dipole_conj

        G = self.ground_size
        half_L = self.slot_length / 2.0
        half_W = self.slot_width  / 2.0

        # Ground plane vertices (centred at origin, z=0)
        gnd_corners = np.array([
            [-G/2, -G/2, 0.0],
            [ G/2, -G/2, 0.0],
            [ G/2,  G/2, 0.0],
            [-G/2,  G/2, 0.0],
        ])
        # Slot outline (aperture along y-axis centred at origin)
        slot_corners = np.array([
            [-half_W, -half_L, 0.0],
            [ half_W, -half_L, 0.0],
            [ half_W,  half_L, 0.0],
            [-half_W,  half_L, 0.0],
        ])
        self.vertices = np.vstack([gnd_corners, slot_corners])
        self.faces = [
            [0,1,2,3],   # ground plane
            [4,5,6,7],   # slot aperture
        ]
        self.feed_point = np.array([0.0, 0.0, 0.0])


class VivaldiAntenna(AntennaGeometry):
    """Antipodal Vivaldi tapered-slot antenna.

    Exponential taper profile: y(x) = A·exp(R·x) + B
    where R is the taper rate and A, B are set from feed width (W_feed)
    to aperture width (W_aperture).

    Ultra-wideband: typically 3:1 bandwidth or wider.

    Parameters
    ----------
    freq_low : float
        Lower band edge frequency [Hz].
    freq_high : float
        Upper band edge frequency [Hz].
    taper_rate : float
        Exponential taper rate coefficient R [1/m].
    substrate_length : float or None
        Physical length of the substrate [m]. Defaults to λ_low / 2.
    feed_width : float or None
        Feed slot width at input [m]. Defaults to λ_low / 20.
    """

    def __init__(self, freq_low: float, freq_high: float,
                 taper_rate: float = 0.05, substrate_length: float = None,
                 feed_width: float = None):
        f_centre = (freq_low + freq_high) / 2.0
        material = Material(name="Substrate_FR4", eps_r=4.4, loss_tangent=0.02)
        super().__init__(
            name="VivaldiAntenna",
            material=material,
            freq_design=f_centre,
        )
        lambda_low = C0 / freq_low
        self.freq_low  = freq_low
        self.freq_high = freq_high
        self.taper_rate = taper_rate
        self.substrate_length = substrate_length if substrate_length else lambda_low / 2.0

        # Feed slot width at input
        self.feed_width = feed_width if feed_width else lambda_low / 20.0

        # Aperture width (at end of taper) ≈ λ_low / 2
        W_aperture = lambda_low / 2.0
        W_feed = self.feed_width
        L = self.substrate_length

        # Solve A and B from boundary conditions:
        # y(0) = W_feed/2  →  A + B = W_feed/2
        # y(L) = W_aperture/2  →  A*exp(R*L) + B = W_aperture/2
        R = taper_rate
        A = (W_aperture / 2.0 - W_feed / 2.0) / (np.exp(R * L) - 1.0)
        B = W_feed / 2.0 - A
        self.A, self.B, self.R = A, B, R

        # Generate taper profile for both flanges
        N_pts = 128
        x_arr = np.linspace(0, L, N_pts)
        y_arr = A * np.exp(R * x_arr) + B  # upper flange edge

        # Upper flange
        upper = np.column_stack([x_arr, y_arr, np.zeros(N_pts)])
        lower = np.column_stack([x_arr, -y_arr, np.zeros(N_pts)])
        self.vertices = np.vstack([upper, lower])
        self.feed_point = np.array([0.0, 0.0, 0.0])
