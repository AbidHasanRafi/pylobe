"""FDTD solver subpackage — high-level FDTDSimulation API."""
import numpy as np
from tqdm import tqdm

from pylobe.constants import C0, PI
from pylobe.solver.fdtd.grid import FDTDGrid
from pylobe.solver.fdtd.update_equations import update_H, update_E
from pylobe.solver.fdtd.pml import PML
from pylobe.solver.fdtd.sources import modulated_gaussian, SoftSource, HardSource
from pylobe.solver.fdtd.probes import VoltageProbe, SParameterProbe, PointProbe
from pylobe.solver.fdtd.farfield import NearToFarField


class FDTDSimulation:
    """High-level FDTD simulation runner.

    Orchestrates: grid setup → material assignment → PML → source →
                  time-stepping → probe recording → NTFF → output.

    Parameters
    ----------
    freq_center : float
        Centre frequency [Hz].
    freq_span : float
        Frequency span for broadband excitation [Hz].
    cells_per_wavelength : int
        Spatial resolution: Δx = λ_min / cells_per_wavelength.
    pml_cells : int
        PML thickness in cells (minimum 8).
    domain_size_wavelengths : float
        Domain size as multiple of λ at freq_center.
    """

    def __init__(self, freq_center: float, freq_span: float,
                 cells_per_wavelength: int = 15, pml_cells: int = 10,
                 domain_size_wavelengths: float = 3.0):
        self.freq_center = freq_center
        self.freq_span   = freq_span
        self.cpw = cells_per_wavelength
        self.pml_cells = max(8, pml_cells)

        # Minimum wavelength (upper edge of band)
        f_max = freq_center + freq_span / 2.0
        lambda_min = C0 / f_max
        dx = lambda_min / cells_per_wavelength

        # Domain size (symmetric around centre freq wavelength)
        lambda_c = C0 / freq_center
        domain = domain_size_wavelengths * lambda_c
        total_cells = int(domain / dx) + 2 * pml_cells

        # Create grid
        self.grid = FDTDGrid(
            size_x=total_cells * dx,
            size_y=total_cells * dx,
            size_z=total_cells * dx,
            dx=dx, dy=dx, dz=dx,
        )
        self.dx = dx

        # Time step (99% of Courant limit for stability margin)
        self.dt = 0.99 * self.grid.courant_number()
        self._pml = PML(self.grid, thickness=self.pml_cells)
        self._CA, self._CB = self.grid.update_coefficients(
            self.dt, freq=freq_center
        )

        self._sources  = []
        self._probes   = []
        self._ntff     = NearToFarField(self.grid,
                                        surface_offset=self.pml_cells + 2,
                                        freq=freq_center)
        self._t        = 0.0
        self._step     = 0
        self._s_param_probe = None
        self._results  = {}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def add_geometry(self, geometry):
        """Map an AntennaGeometry onto the FDTD grid.

        Parameters
        ----------
        geometry : AntennaGeometry
        """
        self.grid.add_geometry(geometry, self.freq_center)
        # Recompute update coefficients after material assignment
        self._CA, self._CB = self.grid.update_coefficients(
            self.dt, freq=self.freq_center
        )
        self._geometry = geometry

    def add_source(self, source_type: str, position: tuple,
                   waveform_fn):
        """Add an excitation source.

        Parameters
        ----------
        source_type : str
            'hard', 'soft'.
        position : tuple of float (x, y, z) [m]
            Physical position of source.
        waveform_fn : callable
            waveform(t) → float.
        """
        g = self.grid
        # Apply geometry centering offset if add_geometry was called first
        off = getattr(g, '_geom_offset', (0.0, 0.0, 0.0))
        px = position[0] + off[0]
        py = position[1] + off[1]
        pz = position[2] + off[2]
        ix = int(px / g.dx)
        iy = int(py / g.dy)
        iz = int(pz / g.dz)
        # Clamp to grid interior (keep 1 cell away from domain walls)
        ix = int(np.clip(ix, 1, g.Nx - 2))
        iy = int(np.clip(iy, 1, g.Ny - 2))
        iz = int(np.clip(iz, 1, g.Nz - 2))

        if source_type == 'hard':
            src = HardSource(ix, iy, iz, 'Ez', waveform_fn)
        elif source_type == 'soft':
            src = SoftSource(ix, iy, iz, 'Ez', waveform_fn)
        else:
            raise ValueError(f"Unknown source_type '{source_type}'")
        self._sources.append(src)

        # Auto-add S-parameter probe at source location
        vp = VoltageProbe(ix, iy, max(iz - 1, 0), iz + 1)
        self._s_param_probe = SParameterProbe(vp, dt=self.dt)
        self._probes.append(vp)

    def add_probe(self, probe_type: str, position: tuple, component: str):
        """Add a field or voltage probe.

        Parameters
        ----------
        probe_type : str
            'point' or 'voltage'.
        position : tuple of float (x, y, z) [m]
        component : str
            Field component (for point probes).
        """
        g = self.grid
        ix = int(np.clip(position[0] / g.dx, 0, g.Nx - 1))
        iy = int(np.clip(position[1] / g.dy, 0, g.Ny - 1))
        iz = int(np.clip(position[2] / g.dz, 0, g.Nz - 1))

        if probe_type == 'point':
            probe = PointProbe(ix, iy, iz, component)
        elif probe_type == 'voltage':
            probe = VoltageProbe(ix, iy, max(iz - 1, 0), iz + 1)
        else:
            raise ValueError(f"Unknown probe_type '{probe_type}'")
        self._probes.append(probe)

    # ------------------------------------------------------------------
    # Main time-stepping loop
    # ------------------------------------------------------------------
    def run(self, n_steps: int = None, until_steady: bool = False,
            verbose: bool = True):
        """Execute the FDTD time-stepping loop.

        Parameters
        ----------
        n_steps : int or None
            Number of time steps. Auto-computed if None.
        until_steady : bool
            If True, run until steady state (not implemented; runs n_steps).
        verbose : bool
            Print progress bar.
        """
        if n_steps is None:
            # Auto: propagate for 10 × round-trip across domain
            domain = self.grid.size_x
            T_prop = domain / C0
            n_steps = int(10.0 * T_prop / self.dt)
            n_steps = max(n_steps, 500)

        if verbose:
            iterator = tqdm(range(n_steps), desc="FDTD", unit="step")
        else:
            iterator = range(n_steps)

        for step in iterator:
            self._t = step * self.dt

            # 1. Update H (n-1/2 → n+1/2)
            update_H(self.grid, self.dt)

            # 2. Inject sources (at E-field time n+1)
            for src in self._sources:
                src.inject(self.grid, (step + 0.5) * self.dt)

            # 3. Update E (n → n+1)
            update_E(self.grid, self._CA, self._CB, self.dt)

            # 4. Record probes
            for probe in self._probes:
                probe.record(self.grid, step)

            # 5. Accumulate NTFF DFT
            self._ntff.record_tangential_fields(
                self.grid, self.freq_center,
                t_now=self._t, dt=self.dt,
            )

            self._step = step

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def get_s_parameters(self, port: int = 0) -> tuple:
        """Extract S11 from recorded port signals.

        Returns
        -------
        tuple (freq_array [Hz], S11_complex)
        """
        if self._s_param_probe is None:
            raise RuntimeError("No S-parameter probe configured.")
        return self._s_param_probe.compute_s11()

    def get_radiation_pattern(self, freq: float,
                              n_theta: int = 181,
                              n_phi: int = 361) -> "RadiationPattern":
        """Compute radiation pattern via NTFF transform.

        Parameters
        ----------
        freq : float
            Frequency [Hz].
        n_theta, n_phi : int
            Pattern resolution.

        Returns
        -------
        RadiationPattern
        """
        import warnings
        from pylobe.analysis.radiation import RadiationPattern
        theta = np.linspace(0, PI, n_theta)
        phi   = np.linspace(0, 2 * PI, n_phi)
        ff    = self._ntff.compute_far_field(theta, phi, freq=freq)
        try:
            return RadiationPattern(ff['E_theta'], ff['E_phi'], theta, phi, freq)
        except ValueError as exc:
            warnings.warn(
                f"NTFF returned zero fields at {freq/1e9:.3f} GHz — "
                "the FDTD simulation may not have radiated (check source position "
                "and that sim.run() completed). Returning None.\n"
                f"  Detail: {exc}",
                UserWarning, stacklevel=2,
            )
            return None

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------
    @classmethod
    def estimate_memory(cls, freq_center: float, freq_span: float,
                        cells_per_wavelength: int = 15,
                        pml_cells: int = 10,
                        domain_size_wavelengths: float = 3.0,
                        verbose: bool = True) -> dict:
        """Estimate RAM required before constructing the simulation.

        Computes the number of Yee cells and the memory needed for all
        six field components (Ex, Ey, Ez, Hx, Hy, Hz) plus PML arrays,
        without allocating any arrays.

        Parameters
        ----------
        freq_center : float
            Centre frequency [Hz].
        freq_span : float
            Frequency span [Hz].
        cells_per_wavelength : int
            Spatial resolution.
        pml_cells : int
            PML thickness in cells.
        domain_size_wavelengths : float
            Domain size as multiple of λ at freq_center.
        verbose : bool
            If True, print a human-readable summary.

        Returns
        -------
        dict with keys:
            'n_cells', 'n_cells_per_dim', 'dx_mm',
            'field_memory_MB', 'total_memory_MB'

        Examples
        --------
        >>> from pylobe import FDTDSimulation
        >>> info = FDTDSimulation.estimate_memory(2.4e9, 1e9, verbose=True)
        Estimated FDTD memory: ...
        """
        import warnings
        f_max = freq_center + freq_span / 2.0
        lambda_min = C0 / f_max
        dx = lambda_min / cells_per_wavelength
        lambda_c = C0 / freq_center
        domain = domain_size_wavelengths * lambda_c
        N_dim = int(domain / dx) + 2 * pml_cells
        n_cells = N_dim ** 3

        # 6 field components, float64 (8 bytes each)
        field_MB = n_cells * 6 * 8 / 1e6
        # PML: ~6 additional arrays of same size
        pml_MB = n_cells * 6 * 8 / 1e6
        total_MB = field_MB + pml_MB

        result = {
            'n_cells':          n_cells,
            'n_cells_per_dim':  N_dim,
            'dx_mm':            dx * 1e3,
            'field_memory_MB':  field_MB,
            'total_memory_MB':  total_MB,
        }

        if verbose:
            print(
                f"Estimated FDTD memory:\n"
                f"  Grid size  : {N_dim}³ = {n_cells:,} Yee cells\n"
                f"  Cell size  : Δx = {dx*1e3:.3f} mm\n"
                f"  Fields     : {field_MB:.0f} MB\n"
                f"  PML        : {pml_MB:.0f} MB\n"
                f"  ── Total   : {total_MB:.0f} MB  ({total_MB/1024:.2f} GB)\n"
            )
            if total_MB > 4000:
                warnings.warn(
                    f"Estimated memory ({total_MB:.0f} MB) is large. "
                    "Consider reducing domain_size_wavelengths or cells_per_wavelength.",
                    UserWarning, stacklevel=2,
                )

        return result

    def save(self, filename: str):
        """Pickle simulation state for post-processing.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'grid':     self.grid,
                'dt':       self.dt,
                'freq_c':   self.freq_center,
                'n_steps':  self._step,
                'results':  self._results,
            }, f)
