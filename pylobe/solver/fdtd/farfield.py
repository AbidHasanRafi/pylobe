"""Near-to-far-field (NTFF) transformation via surface equivalence theorem.

Computes equivalent surface currents on a closed Huygens surface, then
integrates to obtain the far-field radiation pattern.

Reference: Taflove & Hagness, *Computational Electrodynamics*, 3rd Ed., Ch. 8.
"""
import numpy as np
from pylobe.constants import PI, ETA0, C0


class NearToFarField:
    """Near-to-far-field transformer using the equivalence principle.

    Equivalent surface currents on closed surface S:
        J_s = n̂ × H    (electric surface current)
        M_s = -n̂ × E   (magnetic surface current)

    Far-field integrals (Stratton-Chu):
        N_θ = ∫_S [J_s · θ̂ · exp(j·k·r'·r̂)] dS
        L_θ = ∫_S [M_s · θ̂ · exp(j·k·r'·r̂)] dS
        (similarly for φ components)

    Far fields:
        E_θ = -j·k·exp(-jkr)/(4π·r) · (N_θ + L_φ/η0)
        E_φ =  j·k·exp(-jkr)/(4π·r) · (N_φ - L_θ/η0)

    Parameters
    ----------
    grid : FDTDGrid
    surface_offset : int
        Number of cells inward from domain boundary for equivalence surface.
    freq : float
        Single frequency for DFT accumulation [Hz].
    """

    def __init__(self, grid, surface_offset: int = 5, freq: float = None):
        self.grid = grid
        self.surf_off = surface_offset
        self.freq = freq
        g = grid
        t = surface_offset

        # Equivalence surface cell boundaries
        self.ix0, self.ix1 = t, g.Nx - t
        self.iy0, self.iy1 = t, g.Ny - t
        self.iz0, self.iz1 = t, g.Nz - t

        # DFT accumulators for tangential E and H on 6 faces.
        # Indexed as: E[0]=Ex, E[1]=Ey, E[2]=Ez; H[0]=Hx, H[1]=Hy, H[2]=Hz
        self._dft_accum = {}
        for face in ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'):
            self._dft_accum[face] = {'E': np.zeros(3, dtype=complex),
                                     'H': np.zeros(3, dtype=complex)}
        self._n_steps = 0
        self._dt = None

    def record_tangential_fields(self, grid, freq: float, t_now: float,
                                 dt: float):
        """Accumulate in-place DFT of tangential fields on equivalence surface.

        Called every FDTD time step:
            F(f) += f(t) · exp(j·2πf·t) · dt

        This implementation stores spatially-averaged tangential fields
        on each of the 6 faces for memory efficiency.

        Parameters
        ----------
        grid : FDTDGrid
        freq : float
            Target frequency for DFT [Hz].
        t_now : float
            Current simulation time [s].
        dt : float
            Time step [s].
        """
        if self._dt is None:
            self._dt = dt
        phase = np.exp(1j * 2.0 * PI * freq * t_now) * dt
        g = grid
        x0, x1 = self.ix0, self.ix1
        y0, y1 = self.iy0, self.iy1
        z0, z1 = self.iz0, self.iz1

        def avg(arr, slices):
            region = arr[tuple(slices)]
            return float(np.mean(region)) if region.size > 0 else 0.0

        # ── z-low face (n̂ = -ẑ): tangential = Ex, Ey, Hx, Hy ──
        acc = self._dft_accum['zlo']
        acc['E'][0] += avg(g.Ex, [slice(x0, x1), slice(y0, y1), z0]) * phase
        acc['E'][1] += avg(g.Ey, [slice(x0, x1), slice(y0, y1), z0]) * phase
        acc['H'][0] += avg(g.Hx, [slice(x0, x1), slice(y0, y1), z0]) * phase
        acc['H'][1] += avg(g.Hy, [slice(x0, x1), slice(y0, y1), z0]) * phase

        # ── z-high face (n̂ = +ẑ): tangential = Ex, Ey, Hx, Hy ──
        acc = self._dft_accum['zhi']
        acc['E'][0] += avg(g.Ex, [slice(x0, x1), slice(y0, y1), z1]) * phase
        acc['E'][1] += avg(g.Ey, [slice(x0, x1), slice(y0, y1), z1]) * phase
        acc['H'][0] += avg(g.Hx, [slice(x0, x1), slice(y0, y1), z1]) * phase
        acc['H'][1] += avg(g.Hy, [slice(x0, x1), slice(y0, y1), z1]) * phase

        # ── x-low face (n̂ = -x̂): tangential = Ey, Ez, Hy, Hz ──
        acc = self._dft_accum['xlo']
        acc['E'][1] += avg(g.Ey, [x0, slice(y0, y1), slice(z0, z1)]) * phase
        acc['E'][2] += avg(g.Ez, [x0, slice(y0, y1), slice(z0, z1)]) * phase
        acc['H'][1] += avg(g.Hy, [x0, slice(y0, y1), slice(z0, z1)]) * phase
        acc['H'][2] += avg(g.Hz, [x0, slice(y0, y1), slice(z0, z1)]) * phase

        # ── x-high face (n̂ = +x̂): tangential = Ey, Ez, Hy, Hz ──
        acc = self._dft_accum['xhi']
        acc['E'][1] += avg(g.Ey, [x1, slice(y0, y1), slice(z0, z1)]) * phase
        acc['E'][2] += avg(g.Ez, [x1, slice(y0, y1), slice(z0, z1)]) * phase
        acc['H'][1] += avg(g.Hy, [x1, slice(y0, y1), slice(z0, z1)]) * phase
        acc['H'][2] += avg(g.Hz, [x1, slice(y0, y1), slice(z0, z1)]) * phase

        # ── y-low face (n̂ = -ŷ): tangential = Ex, Ez, Hx, Hz ──
        acc = self._dft_accum['ylo']
        acc['E'][0] += avg(g.Ex, [slice(x0, x1), y0, slice(z0, z1)]) * phase
        acc['E'][2] += avg(g.Ez, [slice(x0, x1), y0, slice(z0, z1)]) * phase
        acc['H'][0] += avg(g.Hx, [slice(x0, x1), y0, slice(z0, z1)]) * phase
        acc['H'][2] += avg(g.Hz, [slice(x0, x1), y0, slice(z0, z1)]) * phase

        # ── y-high face (n̂ = +ŷ): tangential = Ex, Ez, Hx, Hz ──
        acc = self._dft_accum['yhi']
        acc['E'][0] += avg(g.Ex, [slice(x0, x1), y1, slice(z0, z1)]) * phase
        acc['E'][2] += avg(g.Ez, [slice(x0, x1), y1, slice(z0, z1)]) * phase
        acc['H'][0] += avg(g.Hx, [slice(x0, x1), y1, slice(z0, z1)]) * phase
        acc['H'][2] += avg(g.Hz, [slice(x0, x1), y1, slice(z0, z1)]) * phase

        self._n_steps += 1

    def compute_far_field(self, theta: np.ndarray,
                          phi: np.ndarray, freq: float = None) -> dict:
        """Compute far-field radiation pattern from accumulated DFT fields.

        Parameters
        ----------
        theta : ndarray, shape (Nt,)
            Polar angles [rad].
        phi : ndarray, shape (Np,)
            Azimuthal angles [rad].
        freq : float or None
            Frequency [Hz]. Uses self.freq if None.

        Returns
        -------
        dict with keys:
            'E_theta' : complex ndarray, shape (Nt, Np)
            'E_phi'   : complex ndarray, shape (Nt, Np)
            'gain_db' : float ndarray, shape (Nt, Np)
            'directivity_db' : float ndarray, shape (Nt, Np)
        """
        from pylobe.constants import C0
        from pylobe.analysis.metrics import directivity as compute_dir

        freq = freq if freq is not None else self.freq
        if freq is None:
            raise ValueError("freq must be specified")
        k = 2.0 * PI * freq / C0

        theta = np.asarray(theta, dtype=float)
        phi   = np.asarray(phi,   dtype=float)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')

        sinT, cosT = np.sin(TH), np.cos(TH)
        sinP, cosP = np.sin(PH), np.cos(PH)

        g = self.grid
        dx = g.dx * (self.ix1 - self.ix0)
        dy = g.dy * (self.iy1 - self.iy0)
        dz = g.dz * (self.iz1 - self.iz0)
        area_xy = dx * dy    # z-faces
        area_xz = dx * dz    # y-faces
        area_yz = dy * dz    # x-faces

        # Accumulate N (electric current) and L (magnetic current) vectors
        # from all six faces using the equivalence principle.
        # n̂×H → J_s (electric current); -n̂×E → M_s (magnetic current)
        # Far-field contributions (Stratton-Chu, scalar-average version):
        #   N_θ, N_φ, L_θ, L_φ — summed over all faces.
        Ntheta = np.zeros(TH.shape, dtype=complex)
        Nphi   = np.zeros(TH.shape, dtype=complex)
        Ltheta = np.zeros(TH.shape, dtype=complex)
        Lphi   = np.zeros(TH.shape, dtype=complex)

        # ── z-faces: n̂ = ±ẑ, tangential = Ex, Ey, Hx, Hy ──────────────
        for face, sign, area in (('zhi', +1.0, area_xy), ('zlo', -1.0, area_xy)):
            Ex = self._dft_accum[face]['E'][0]
            Ey = self._dft_accum[face]['E'][1]
            Hx = self._dft_accum[face]['H'][0]
            Hy = self._dft_accum[face]['H'][1]
            # J_s = sign·ẑ × H = sign·(Hx·ŷ - Hy·x̂)
            Jx = -sign * Hy;  Jy = sign * Hx
            # M_s = -sign·ẑ × E = -sign·(Ex·ŷ - Ey·x̂) → Mx=sign·Ey, My=-sign·Ex
            Mx = sign * Ey;  My = -sign * Ex
            Ntheta += (Jx * cosT * cosP + Jy * cosT * sinP) * area
            Nphi   += (-Jx * sinP       + Jy * cosP)        * area
            Ltheta += (Mx * cosT * cosP + My * cosT * sinP) * area
            Lphi   += (-Mx * sinP       + My * cosP)        * area

        # ── x-faces: n̂ = ±x̂, tangential = Ey, Ez, Hy, Hz ──────────────
        for face, sign, area in (('xhi', +1.0, area_yz), ('xlo', -1.0, area_yz)):
            Ey = self._dft_accum[face]['E'][1]
            Ez = self._dft_accum[face]['E'][2]
            Hy = self._dft_accum[face]['H'][1]
            Hz = self._dft_accum[face]['H'][2]
            # J_s = sign·x̂ × H = sign·(Hy·ẑ - Hz·ŷ) → wait...
            # x̂ × H = x̂ × (Hx·x̂+Hy·ŷ+Hz·ẑ) = Hy·(x̂×ŷ)+Hz·(x̂×ẑ) = Hy·ẑ - Hz·ŷ
            Jy = -sign * Hz;  Jz = sign * Hy
            Mz = sign * Ey;   My = -sign * Ez
            # θ-component: J·θ̂ where θ̂ = (cosT·cosP, cosT·sinP, -sinT)
            Ntheta += (Jy * cosT * sinP + Jz * (-sinT)) * area
            Nphi   += Jy * cosP                          * area
            Ltheta += (My * cosT * sinP + Mz * (-sinT)) * area
            Lphi   += My * cosP                          * area

        # ── y-faces: n̂ = ±ŷ, tangential = Ex, Ez, Hx, Hz ──────────────
        for face, sign, area in (('yhi', +1.0, area_xz), ('ylo', -1.0, area_xz)):
            Ex = self._dft_accum[face]['E'][0]
            Ez = self._dft_accum[face]['E'][2]
            Hx = self._dft_accum[face]['H'][0]
            Hz = self._dft_accum[face]['H'][2]
            # ŷ × H = Hz·x̂ - Hx·ẑ
            Jx = sign * Hz;  Jz = -sign * Hx
            Mx = sign * Ez;  Mz = -sign * Ex
            Ntheta += (Jx * cosT * cosP + Jz * (-sinT)) * area
            Nphi   += (-Jx * sinP)                       * area
            Ltheta += (Mx * cosT * cosP + Mz * (-sinT)) * area
            Lphi   += (-Mx * sinP)                       * area

        E_theta = -1j * k / (4.0 * PI) * (Ntheta + Lphi / ETA0)
        E_phi   =  1j * k / (4.0 * PI) * (Nphi   - Ltheta / ETA0)

        D_arr, D_max, D_dbi = compute_dir(E_theta, E_phi, theta, phi)
        gain_db = 10.0 * np.log10(np.abs(D_arr) + 1e-20)

        return {
            'E_theta':        E_theta,
            'E_phi':          E_phi,
            'gain_db':        gain_db,
            'directivity_db': gain_db,  # (no efficiency correction here)
        }
