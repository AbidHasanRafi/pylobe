"""Log-Periodic Dipole Array (LPDA) geometry."""
import numpy as np
from pylobe.geometry.base import AntennaGeometry, Material, AIR, COPPER
from pylobe.constants import C0, PI
from pylobe.utils.validation import check_frequency, check_positive, check_range


class LogPeriodicArray(AntennaGeometry):
    """Log-Periodic Dipole Array (LPDA) for broadband coverage.

    The LPDA is self-similar: element lengths and positions follow a
    geometric progression controlled by the taper ratio τ (tau) and the
    relative spacing σ (sigma):

        L_n+1 / L_n = τ          (successive lengths)
        d_n+1 / d_n = τ          (successive spacings)
        σ = d_n / (2 · L_n)      (relative spacing)

    For τ and σ in the practical ranges below, the LPDA is well-matched
    over the full decade bandwidth:
        τ = 0.80–0.95  (closer to 1 → more elements, flatter response)
        σ = 0.05–0.20  (closer to 0.20 → narrower elements, higher gain)

    The LPDA is modelled as a series of half-wave dipoles connected to a
    balanced twin-boom feeder of alternating polarity (phase reversal
    between adjacent elements).

    Parameters
    ----------
    freq_low : float
        Lower edge of operating band [Hz].
    freq_high : float
        Upper edge of operating band [Hz].
    tau : float
        Taper ratio (0.80–0.95).
    sigma : float
        Relative spacing (0.05–0.20).
    conductor_material : Material or None
        Element/boom conductor. Defaults to COPPER.

    Attributes
    ----------
    element_lengths : ndarray
        Half-element lengths for each dipole [m] (full length = 2 × half).
    element_positions : ndarray
        Boom positions (z-axis) [m] for each element; front (shortest) at z=0.
    N_elements : int
        Number of dipole elements.
    gain_approx_dbi : float
        Empirical front gain [dBi].
    bandwidth_ratio : float
        Actual achieved bandwidth: f_high / f_low.
    """

    def __init__(self, freq_low: float, freq_high: float,
                 tau: float = 0.88, sigma: float = 0.12,
                 conductor_material: Material = None):
        check_frequency(freq_low)
        check_frequency(freq_high)
        if freq_high <= freq_low:
            raise ValueError("freq_high must be greater than freq_low")
        check_range(tau,   0.75, 0.98, "tau")
        check_range(sigma, 0.03, 0.25, "sigma")

        f_centre = np.sqrt(freq_low * freq_high)
        super().__init__(
            name="LogPeriodicArray",
            material=AIR,
            freq_design=f_centre,
        )
        self.freq_low  = freq_low
        self.freq_high = freq_high
        self.tau       = tau
        self.sigma     = sigma
        self.conductor_material = (conductor_material if conductor_material is not None
                                   else COPPER)

        # Design the LPDA: shortest element resonant at freq_high,
        # longest resonant at freq_low (add 10% guard).
        L_min = 0.5 * C0 / (freq_high * 1.1)   # shortest half-element length [m]
        L_max = 0.5 * C0 / (freq_low  * 0.9)   # longest  half-element length [m]

        # Number of elements
        N = int(np.ceil(np.log(L_max / L_min) / np.log(1.0 / tau))) + 1
        self.N_elements = N

        # Element half-lengths (front to back, shortest to longest)
        half_lengths = L_min * (1.0 / tau) ** np.arange(N)
        self.element_lengths = half_lengths  # half-lengths [m]

        # Element positions along the boom
        # d_n = 2 · sigma · L_n  (spacing from element n to n+1)
        positions = np.zeros(N)
        for i in range(1, N):
            positions[i] = positions[i - 1] + 2.0 * sigma * half_lengths[i - 1]
        self.element_positions = positions

        self.boom_length = positions[-1]
        self.bandwidth_ratio = freq_high / freq_low

        # Empirical gain (Carrel, 1961): G ≈ 10·log10(N · τ) + 4 dBi
        self.gain_approx_dbi = float(10 * np.log10(N * tau) + 4.0)

        # Build vertices
        all_verts = []
        all_edges = []
        v_offset  = 0
        N_seg     = 7

        for i, (pos_z, half_L) in enumerate(zip(positions, half_lengths)):
            y_pts = np.linspace(-half_L, half_L, N_seg + 1)
            verts = np.column_stack([
                np.zeros(N_seg + 1),
                y_pts,
                np.full(N_seg + 1, pos_z),
            ])
            edges = [(v_offset + j, v_offset + j + 1) for j in range(N_seg)]
            all_verts.append(verts)
            all_edges.extend(edges)
            v_offset += N_seg + 1

        self.vertices = np.vstack(all_verts)
        self.edges    = all_edges
        self.feed_point = np.array([0.0, 0.0, 0.0])

    def __repr__(self) -> str:
        return (
            f"LogPeriodicArray\n"
            f"  Band            : {self.freq_low/1e6:.0f}–{self.freq_high/1e6:.0f} MHz\n"
            f"  τ / σ           : {self.tau:.3f} / {self.sigma:.3f}\n"
            f"  N elements      : {self.N_elements}\n"
            f"  Boom length     : {self.boom_length*1e3:.1f} mm\n"
            f"  Shortest elem   : {self.element_lengths[0]*2*1e3:.1f} mm\n"
            f"  Longest  elem   : {self.element_lengths[-1]*2*1e3:.1f} mm\n"
            f"  Approx gain     : {self.gain_approx_dbi:.1f} dBi\n"
        )
