"""Unified high-level workflow object for antenna design.

``AntennaDesign`` chains geometry → solver → analysis → visualisation →
report in a single object, reducing the boilerplate required for common
design flows.

New in this release
-------------------
* ``plot_dashboard()``         — comprehensive multi-panel 2-D dashboard.
* ``plot_2d_patterns()``       — full-pattern panel (E/H polar + Cartesian
                                 + heatmap + phase + multi-cut overlay).
* ``plot_frequency_response()``— S11 + VSWR + return loss + R + X + |Z|/∠Z
                                 + group delay + S-parameter table.
* ``export_all(prefix, fmt)``  — export all plots at ≥ 600 DPI.
* ``summary()``                — richer text output with beam solid angle,
                                 axial ratio, efficiency, and FNBW.

Usage
-----
>>> from pylobe import RectangularPatch, ROGERS4003
>>> from pylobe.design import AntennaDesign
>>> patch   = RectangularPatch(freq=2.4e9, substrate_material=ROGERS4003)
>>> design  = AntennaDesign(patch)
>>> design.solve()
>>> print(design.summary())
>>> fig_pat, fig_freq = design.plot_dashboard()
>>> design.export_all('results/my_patch')
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np


class AntennaDesign:
    """Unified workflow object for a single antenna design.

    Wraps a geometry object and an automatically selected (or user-specified)
    solver, and provides convenience methods for the most common design tasks.

    Parameters
    ----------
    geometry : AntennaGeometry
        Any PyLobe antenna geometry instance.
    solver : object or None
        Pre-constructed solver instance. If None, one is auto-selected
        based on the geometry type.
    freq : float or None
        Analysis frequency [Hz]. Defaults to ``geometry.freq_design``.

    Examples
    --------
    >>> from pylobe import RectangularPatch
    >>> from pylobe.design import AntennaDesign
    >>> patch = RectangularPatch(freq=2.4e9)
    >>> d = AntennaDesign(patch)
    >>> d.solve()
    >>> sm = d.pattern_summary()
    >>> sm.peak_gain_dbi > 0
    True
    """

    def __init__(self, geometry, solver=None, freq: float = None):
        self.geometry = geometry
        self.freq     = freq if freq is not None else geometry.freq_design
        self._solver  = solver
        self._radiation_pattern = None
        self._s11_data = None   # (freq_array, S11_complex)
        self._z_data   = None   # (freq_array, Z_complex)
        self._solved   = False

    # ──────────────────────────────────────────────────────────────────────────
    # Auto solver selection
    # ──────────────────────────────────────────────────────────────────────────
    def _get_solver(self):
        """Return the solver, auto-constructing from geometry type if needed."""
        if self._solver is not None:
            return self._solver

        from pylobe.geometry.patch   import RectangularPatch
        from pylobe.geometry.dipole  import HalfWaveDipole, FoldedDipole, BowTieDipole
        from pylobe.geometry.monopole import QuarterWaveMonopole

        geom = self.geometry
        if isinstance(geom, RectangularPatch):
            from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver
            self._solver = PatchAnalyticalSolver(geom, self.freq)
        elif isinstance(geom, HalfWaveDipole):
            from pylobe.solver.analytical.dipole_solver import DipoleSolver
            self._solver = DipoleSolver(geom, self.freq)
        elif isinstance(geom, (FoldedDipole, BowTieDipole)):
            from pylobe.geometry.dipole import HalfWaveDipole as HWD
            from pylobe.solver.analytical.dipole_solver import DipoleSolver
            warnings.warn(
                f"{type(geom).__name__} does not have a dedicated analytical solver. "
                "Using HalfWaveDipole-equivalent DipoleSolver as approximation.",
                UserWarning, stacklevel=2,
            )
            proxy = HWD(self.freq, wire_radius=getattr(geom, 'wire_radius', None))
            self._solver = DipoleSolver(proxy, self.freq)
        elif isinstance(geom, QuarterWaveMonopole):
            from pylobe.geometry.dipole import HalfWaveDipole as HWD
            from pylobe.solver.analytical.dipole_solver import DipoleSolver
            proxy = HWD(self.freq, length_factor=0.47)
            self._solver = DipoleSolver(proxy, self.freq)
        else:
            raise TypeError(
                f"No analytical solver available for {type(geom).__name__}. "
                "Pass a pre-constructed solver to AntennaDesign(geometry, solver=...)."
            )
        return self._solver

    # ──────────────────────────────────────────────────────────────────────────
    # Solving
    # ──────────────────────────────────────────────────────────────────────────
    def solve(self, n_theta: int = 181, n_phi: int = 361,
              n_freq: int = 200) -> 'AntennaDesign':
        """Run the solver and cache all results.

        Parameters
        ----------
        n_theta : int   Theta resolution (181 gives 1° steps).
        n_phi   : int   Phi   resolution (361 gives 1° steps).
        n_freq  : int   Frequency points for S11 / impedance sweeps.

        Returns
        -------
        AntennaDesign — self (for method chaining).
        """
        solver = self._get_solver()

        self._radiation_pattern = solver.radiation_pattern(
            n_theta=n_theta, n_phi=n_phi
        )
        if hasattr(solver, 's11'):
            self._s11_data = solver.s11(n_freq=n_freq)
        if hasattr(solver, 'impedance_sweep'):
            self._z_data = solver.impedance_sweep(n_freq=n_freq)
        elif self._s11_data is not None:
            freqs, S11 = self._s11_data
            Z0 = float(getattr(self.geometry, 'feed_impedance', 50.0))
            Z  = Z0 * (1.0 + S11) / np.clip(1.0 - S11, 1e-10, None)
            self._z_data = (freqs, Z)

        self._solved = True
        return self

    def _require_solved(self):
        if not self._solved:
            raise RuntimeError(
                "Call design.solve() before accessing results.\n"
                "  Example: design.solve().plot_dashboard()"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Results access
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def radiation_pattern(self):
        """The computed RadiationPattern object."""
        self._require_solved()
        return self._radiation_pattern

    def pattern_summary(self):
        """PatternSummary dataclass with all key metrics."""
        self._require_solved()
        return self._radiation_pattern.summary()

    def summary(self) -> str:
        """Human-readable design summary — richer output than before.

        Includes: peak gain, HPBW (E/H), FNBW, SLL, F/B ratio,
        beam solid angle, axial ratio at peak direction, radiation
        efficiency, resonant frequency, S11 min, VSWR, 10-dB bandwidth.
        """
        geom  = self.geometry
        lines = [
            '=' * 60,
            f'  AntennaDesign — {type(geom).__name__}',
            f'  Design frequency : {geom.freq_design / 1e9:.4f} GHz',
            f'  Analysis freq    : {self.freq / 1e9:.4f} GHz',
            f'  Material         : {geom.material.name}',
            '=' * 60,
        ]

        if not self._solved:
            lines.append('  (not solved — call .solve() first)')
            return '\n'.join(lines)

        rp = self._radiation_pattern
        sm = self.pattern_summary()

        # Extra pattern metrics
        from pylobe.analysis.metrics import (
            beamwidth_fnbw, beam_solid_angle as _bsa, axial_ratio,
        )
        theta_deg  = np.rad2deg(rp.theta)
        e_cut_lin  = rp.e_plane_cut()
        h_cut_lin  = rp.h_plane_cut()
        fnbw_e     = beamwidth_fnbw(e_cut_lin, theta_deg)
        fnbw_h     = beamwidth_fnbw(h_cut_lin, theta_deg)
        omega_a    = _bsa(rp.E_theta, rp.E_phi, rp.theta, rp.phi)
        pi_pk, pj_pk = np.unravel_index(
            np.argmax(rp.directivity_2d), rp.directivity_2d.shape)
        ar_peak = axial_ratio(
            complex(rp.E_theta[pi_pk, pj_pk]),
            complex(rp.E_phi[pi_pk, pj_pk]),
        )

        lines += [
            '  RADIATION PATTERN',
            f'    Peak gain        : {sm.peak_gain_dbi:.2f} dBi',
            f'    Peak direction   : θ = {sm.theta_max_deg:.1f}°,  '
            f'φ = {sm.phi_max_deg:.1f}°',
            f'    HPBW  E-plane    : {sm.hpbw_e_deg:.1f}°',
            f'    HPBW  H-plane    : {sm.hpbw_h_deg:.1f}°',
            f'    FNBW  E-plane    : {fnbw_e:.1f}°',
            f'    FNBW  H-plane    : {fnbw_h:.1f}°',
            f'    SLL              : {sm.sll_db:.1f} dB',
            f'    F/B Ratio        : {sm.fbr_db:.1f} dB',
            f'    Beam Solid Angle : {omega_a:.4f} sr',
            f'    Axial Ratio @peak: {ar_peak:.2f}  '
            f'({20 * np.log10(max(ar_peak, 1.0)):.1f} dB)',
        ]

        # Radiation efficiency
        solver = self._get_solver()
        if hasattr(solver, 'radiation_efficiency'):
            eff = solver.radiation_efficiency
            lines.append(f'    Radiation eff.   : {eff * 100:.1f} %')

        # S-parameter summary
        if self._s11_data is not None:
            freqs, S11   = self._s11_data
            S11_db       = 20.0 * np.log10(np.clip(np.abs(S11), 1e-10, None))
            vswr         = ((1 + np.abs(S11)) /
                            np.clip(1 - np.abs(S11), 1e-6, None))
            idx_res      = int(np.argmin(S11_db))
            below10      = S11_db < -10
            bw_mhz, fbw  = 0.0, 0.0
            if below10.any():
                lo    = int(np.argmax(below10))
                hi    = len(below10) - 1 - int(np.argmax(below10[::-1]))
                bw_mhz = (float(freqs[hi]) - float(freqs[lo])) / 1e6
                fc    = (float(freqs[lo]) + float(freqs[hi])) / 2.0
                fbw   = 200.0 * bw_mhz / (fc / 1e6 + 1e-30)
            lines += [
                '',
                '  S-PARAMETERS',
                f'    Resonant freq    : {freqs[idx_res] / 1e9:.4f} GHz',
                f'    S11 @ resonance  : {S11_db[idx_res]:.1f} dB',
                f'    Return loss      : {-S11_db[idx_res]:.1f} dB',
                f'    VSWR @ resonance : {vswr[idx_res]:.2f}',
                f'    10-dB bandwidth  : {bw_mhz:.1f} MHz  ({fbw:.2f} %)',
            ]

        lines.append('=' * 60)
        return '\n'.join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation — single plots (legacy compatibility)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_pattern(self, plane: str = 'both',
                     dyn_range: float = 40):
        """Plot radiation pattern (polar, E/H plane or 3-D).

        Parameters
        ----------
        plane : str  — ``'E'``, ``'H'``, ``'both'``, or ``'3d'``.
        dyn_range : float [dB]

        Returns
        -------
        Figure (matplotlib or Plotly depending on *plane*).
        """
        self._require_solved()
        rp = self._radiation_pattern
        if plane == '3d':
            from pylobe.visualization.lobe3d import plot_3d_radiation
            return plot_3d_radiation(rp)
        if plane in ('both', 'E', 'H'):
            from pylobe.visualization.polar import plot_e_h_plane, plot_polar
            if plane == 'both':
                return plot_e_h_plane(rp, dyn_range=dyn_range)
            phi   = 0.0 if plane == 'E' else 90.0
            cut   = rp.phi_cut(phi)
            p_max = float(np.max(cut)) or 1.0
            db    = 10.0 * np.log10(np.clip(cut / p_max, 1e-10, None))
            tdeg  = np.rad2deg(rp.theta)
            a360  = np.concatenate([tdeg, tdeg + 180.0])
            d360  = np.concatenate([db, db[::-1]])
            return plot_polar(d360, a360,
                              label=f'{plane}-plane | {rp.freq / 1e9:.3f} GHz',
                              dyn_range=dyn_range)
        raise ValueError(f"plane must be 'E', 'H', 'both', or '3d', got {plane!r}")

    def plot_s11(self, Z0: float = 50.0, threshold: float = -10.0):
        """Plot S11 vs frequency."""
        self._require_solved()
        if self._s11_data is None:
            raise RuntimeError("No S11 data — solver does not support s11().")
        from pylobe.visualization.heatmap import plot_s11_vs_freq
        freqs, S11 = self._s11_data
        s11_db     = 20.0 * np.log10(np.clip(np.abs(S11), 1e-10, None))
        return plot_s11_vs_freq(freqs, s11_db,
                                bandwidth_threshold=threshold,
                                title=f'S11 — {type(self.geometry).__name__}')

    def plot_impedance(self):
        """Plot input impedance (R + jX) vs frequency."""
        self._require_solved()
        if self._z_data is None:
            raise RuntimeError("No impedance data available.")
        from pylobe.visualization.heatmap import plot_impedance_vs_freq
        freqs, Z = self._z_data
        return plot_impedance_vs_freq(
            freqs, Z,
            title=f'Input Impedance — {type(self.geometry).__name__}',
        )

    def plot_smith(self):
        """Plot Smith chart with impedance trajectory."""
        self._require_solved()
        if self._z_data is None:
            raise RuntimeError("No impedance data available.")
        from pylobe.visualization.smith_chart import plot_smith_chart
        freqs, Z = self._z_data
        return plot_smith_chart(
            impedance_trace=Z,
            freq_labels=freqs,
            title=f'Smith Chart — {type(self.geometry).__name__}',
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation — comprehensive dashboards (new)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_2d_patterns(self, dyn_range: float = 40):
        """Full 2-D pattern dashboard.

        4-row × 3-column figure containing:
        * E/H polar patterns with HPBW, SLL, null annotations
        * E/H Cartesian patterns with −3/−10 dB references
        * Phase pattern (E-plane)
        * Full gain heatmap (θ vs φ, Turbo colormap)
        * Diagonal φ = 45°/135° polar cuts
        * Multi-cut Cartesian overlay

        Parameters
        ----------
        dyn_range : float  Dynamic range [dB].

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_solved()
        from pylobe.visualization.dashboard import plot_pattern_dashboard
        geom_name = type(self.geometry).__name__
        return plot_pattern_dashboard(
            self._radiation_pattern,
            dyn_range=dyn_range,
            title=geom_name,
        )

    def plot_frequency_response(self, Z0: float = 50.0):
        """Full frequency-response dashboard.

        3-row × 3-column figure containing:
        * |S11| (dB) with −10 dB threshold and resonance annotation
        * VSWR with matched-band highlight
        * Return loss with 10 dB threshold
        * Resistance R(Z) with Z₀ reference
        * Reactance X(Z) with inductive/capacitive fill
        * |Z| and ∠Z on dual axes
        * Group delay −dφ/dω
        * Phase of S₁₁
        * S-parameter summary table

        Parameters
        ----------
        Z0 : float  Reference impedance [Ω].

        Returns
        -------
        matplotlib.figure.Figure or None if no frequency data.
        """
        self._require_solved()
        if self._s11_data is None:
            warnings.warn("No S11 data — cannot create frequency dashboard.",
                          UserWarning, stacklevel=2)
            return None
        from pylobe.visualization.dashboard import plot_frequency_dashboard
        freqs, S11 = self._s11_data
        _, Z        = self._z_data
        return plot_frequency_dashboard(
            freqs, S11, Z,
            Z0=Z0,
            title=type(self.geometry).__name__,
        )

    def plot_dashboard(self, dyn_range: float = 40,
                       Z0: float = 50.0) -> tuple:
        """Create all dashboard figures for this design.

        Calls :func:`~pylobe.visualization.dashboard.plot_design_dashboard`
        internally.

        Returns
        -------
        tuple (fig_pattern, fig_freq)
            *fig_pattern* — matplotlib Figure (always available).
            *fig_freq*    — matplotlib Figure or None (if no freq data).
        """
        self._require_solved()
        from pylobe.visualization.dashboard import plot_design_dashboard
        return plot_design_dashboard(self, dyn_range=dyn_range, Z0=Z0)

    def plot_gain_heatmap(self, dyn_range: float = 40,
                          interactive: bool = True):
        """Gain heatmap: interactive Plotly or static matplotlib.

        Parameters
        ----------
        dyn_range : float  [dB]
        interactive : bool
            True → Plotly figure.  False → matplotlib at 600 DPI.

        Returns
        -------
        Figure
        """
        self._require_solved()
        rp = self._radiation_pattern
        if interactive:
            from pylobe.visualization.heatmap import plot_gain_heatmap
            return plot_gain_heatmap(rp,
                                     title=f'Gain Heatmap — {type(self.geometry).__name__}')
        else:
            from pylobe.visualization.heatmap import plot_gain_heatmap_mpl
            return plot_gain_heatmap_mpl(rp, dyn_range=dyn_range)

    def plot_waterfall(self, freq_list_ghz=None, n_theta: int = 181,
                       phi_cut_deg: float = 0.0):
        """Frequency–angle gain waterfall plot.

        If *freq_list_ghz* is None, the full S11 frequency range is used and
        the radiation pattern is re-computed at each point.  For quick
        visualisation, pass a short list of frequencies (e.g. 5–11 points).

        Parameters
        ----------
        freq_list_ghz : list of float or None
        n_theta : int
        phi_cut_deg : float

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_solved()
        solver = self._get_solver()

        if freq_list_ghz is None:
            if self._s11_data is None:
                raise RuntimeError(
                    "No frequency sweep data. Pass freq_list_ghz explicitly."
                )
            freqs_hz = self._s11_data[0]
            # Sub-sample to at most 21 points for speed
            idx      = np.round(np.linspace(0, len(freqs_hz) - 1, 21)).astype(int)
            freq_list_ghz = [float(freqs_hz[i]) / 1e9 for i in idx]

        theta_deg   = np.linspace(0, 180, n_theta)
        patterns_db = []
        for f_ghz in freq_list_ghz:
            f_hz = f_ghz * 1e9
            if hasattr(solver, 'radiation_pattern'):
                # Re-solve at this frequency — lightweight for analytical solvers
                try:
                    _cls    = type(solver)
                    _s_tmp  = _cls(self.geometry, f_hz)
                    _rp_tmp = _s_tmp.radiation_pattern(n_theta=n_theta, n_phi=1)
                    cut_lin = _rp_tmp.e_plane_cut()
                except Exception:
                    cut_lin = self._radiation_pattern.e_plane_cut()
            else:
                cut_lin = self._radiation_pattern.e_plane_cut()

            p_max = float(np.max(cut_lin)) or 1.0
            db_cut = 10.0 * np.log10(np.clip(cut_lin / p_max, 1e-10, None))
            patterns_db.append(db_cut[:n_theta])

        from pylobe.visualization.heatmap import plot_waterfall
        return plot_waterfall(
            freq_ghz   = np.array(freq_list_ghz),
            theta_deg  = theta_deg,
            patterns_db = np.array(patterns_db),
            phi_cut_deg = phi_cut_deg,
            title       = f'{type(self.geometry).__name__} — Gain Waterfall',
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────────
    def export_all(self, prefix: str = 'antenna',
                   fmt: str = 'png',
                   dpi: int = 600,
                   Z0: float = 50.0,
                   include_interactive: bool = True) -> list:
        """Export all plots at ≥ 600 DPI.

        Saved files (relative to the caller's CWD or absolute *prefix*):

        =============================  =====================
        ``{prefix}_pattern_dashboard.{fmt}``  4×3 pattern figure
        ``{prefix}_freq_dashboard.{fmt}``     3×3 frequency figure
        ``{prefix}_pattern_3d.html``          Plotly 3-D interactive
        ``{prefix}_smith.html``               Plotly Smith chart
        ``{prefix}_gain_heatmap.html``        Plotly interactive heatmap
        =============================  =====================

        Parameters
        ----------
        prefix : str   File path prefix (directories must exist).
        fmt    : str   Raster/vector format for matplotlib: ``'png'``,
                       ``'pdf'``, ``'svg'``.
        dpi    : int   Export DPI (minimum 600 enforced).
        Z0     : float Reference impedance [Ω].
        include_interactive : bool   Export Plotly HTML files.

        Returns
        -------
        list of str  — paths of all exported files.
        """
        self._require_solved()
        from pylobe.visualization.style import export_fig

        dpi   = max(600, dpi)
        out   = []
        geom  = type(self.geometry).__name__

        # ── Pattern dashboard ─────────────────────────────────────────────────
        fig_pat = self.plot_2d_patterns()
        path    = f'{prefix}_pattern_dashboard.{fmt}'
        export_fig(fig_pat, path, dpi=dpi, fmt=fmt)
        out.append(path)
        import matplotlib.pyplot as plt
        plt.close(fig_pat)

        # ── Frequency dashboard ───────────────────────────────────────────────
        if self._s11_data is not None:
            fig_freq = self.plot_frequency_response(Z0=Z0)
            if fig_freq is not None:
                path = f'{prefix}_freq_dashboard.{fmt}'
                export_fig(fig_freq, path, dpi=dpi, fmt=fmt)
                out.append(path)
                plt.close(fig_freq)

        # ── Interactive HTML exports ──────────────────────────────────────────
        if include_interactive:
            try:
                fig_3d = self.plot_pattern(plane='3d')
                path   = f'{prefix}_pattern_3d.html'
                export_fig(fig_3d, path)
                out.append(path)
            except Exception as exc:
                warnings.warn(f'3-D export failed: {exc}', UserWarning)

            if self._z_data is not None:
                try:
                    fig_smith = self.plot_smith()
                    path      = f'{prefix}_smith.html'
                    export_fig(fig_smith, path)
                    out.append(path)
                except Exception as exc:
                    warnings.warn(f'Smith chart export failed: {exc}', UserWarning)

            try:
                fig_hm = self.plot_gain_heatmap(interactive=True)
                path   = f'{prefix}_gain_heatmap.html'
                export_fig(fig_hm, path)
                out.append(path)
            except Exception as exc:
                warnings.warn(f'Gain heatmap export failed: {exc}', UserWarning)

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Report generation
    # ──────────────────────────────────────────────────────────────────────────
    def report(self, filename: str, include_3d: bool = False) -> None:
        """Generate a PDF design report.

        Parameters
        ----------
        filename : str  Output PDF path.
        include_3d : bool  If True, attempt to include a static 3-D image.
        """
        self._require_solved()
        from pylobe.export.report import generate_report
        generate_report(self, filename, include_3d=include_3d)

    def __repr__(self) -> str:
        status = 'solved' if self._solved else 'not solved'
        return (
            f"AntennaDesign({type(self.geometry).__name__}, "
            f"f={self.freq / 1e9:.3f} GHz, {status})"
        )
