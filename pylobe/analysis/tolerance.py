"""Monte Carlo manufacturing tolerance analysis for antenna designs.

Propagates dimensional uncertainty (±tolerance on each parameter) through
the solver and reports the statistical distribution of key metrics such as
resonant frequency, S11, gain, and HPBW.

Typical PCB etching tolerance: ±0.1 mm.
Typical substrate thickness tolerance: ±5% of nominal.

Usage
-----
>>> from pylobe import RectangularPatch, PatchAnalyticalSolver
>>> from pylobe.analysis.tolerance import ToleranceAnalyzer
>>> patch = RectangularPatch(freq=2.4e9)
>>> analyzer = ToleranceAnalyzer(patch, PatchAnalyticalSolver)
>>> results = analyzer.run(n_samples=500, tolerances={'h': 0.05e-3, 'eps_r': 0.1})
>>> results.summary()
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class ToleranceResult:
    """Container for Monte Carlo tolerance analysis results.

    Attributes
    ----------
    param_name : str
        Name of the metric being analysed.
    nominal : float
        Nominal (design) value.
    mean : float
        Sample mean.
    std : float
        Sample standard deviation.
    p5 : float
        5th percentile.
    p95 : float
        95th percentile.
    samples : ndarray
        All sample values.
    """
    param_name: str
    nominal:    float
    mean:       float
    std:        float
    p5:         float
    p95:        float
    samples:    np.ndarray = field(repr=False)

    @property
    def cpk(self) -> Optional[float]:
        """Process capability index Cpk for a ±10% acceptance band.

        Cpk = min(USL - μ, μ - LSL) / (3σ)
        where USL/LSL = nominal ± 10%.

        Returns
        -------
        float or None
            None if std is zero.
        """
        if self.std == 0:
            return None
        usl = self.nominal * 1.10
        lsl = self.nominal * 0.90
        return float(min(usl - self.mean, self.mean - lsl) / (3.0 * self.std))

    def __str__(self) -> str:
        cpk_str = f'{self.cpk:.2f}' if self.cpk is not None else 'N/A'
        return (
            f"{self.param_name}:\n"
            f"  Nominal    : {self.nominal:.4g}\n"
            f"  Mean ± σ   : {self.mean:.4g} ± {self.std:.4g}\n"
            f"  90% CI     : [{self.p5:.4g}, {self.p95:.4g}]\n"
            f"  Cpk (±10%) : {cpk_str}"
        )


class ToleranceResults:
    """Collection of ToleranceResult objects for multiple metrics.

    Attributes
    ----------
    results : dict of str → ToleranceResult
    n_samples : int
    tolerances : dict
    """

    def __init__(self, results: Dict[str, ToleranceResult],
                 n_samples: int, tolerances: dict):
        self.results = results
        self.n_samples = n_samples
        self.tolerances = tolerances

    def __getitem__(self, key: str) -> ToleranceResult:
        return self.results[key]

    def __repr__(self) -> str:
        return (
            f"ToleranceResults(n_samples={self.n_samples}, "
            f"metrics={list(self.results.keys())})"
        )

    def summary(self) -> None:
        """Print a human-readable summary of all metrics."""
        print(f"\nMonte Carlo Tolerance Analysis — {self.n_samples} samples")
        print(f"Parameter tolerances (±1σ): {self.tolerances}")
        print("─" * 55)
        for r in self.results.values():
            print(r)
            print()

    def plot(self) -> "matplotlib.figure.Figure":
        """Plot histograms of all sampled metrics.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from pylobe.visualization.style import _setup_style
        _setup_style()

        n = len(self.results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5 * cols, 3.5 * rows),
                                 squeeze=False)

        for ax, (name, r) in zip(axes.flat, self.results.items()):
            ax.hist(r.samples, bins=40, color='#3498db', edgecolor='white',
                    alpha=0.85, density=True)
            ax.axvline(r.nominal, color='#e74c3c', linewidth=2.0,
                       linestyle='--', label=f'Nominal={r.nominal:.4g}')
            ax.axvline(r.mean, color='#2ecc71', linewidth=1.5,
                       linestyle='-', label=f'Mean={r.mean:.4g}')
            # ±1σ shading
            ax.axvspan(r.mean - r.std, r.mean + r.std,
                       alpha=0.15, color='#2ecc71', label='±1σ')
            ax.set_xlabel(name, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_title(f'{name} distribution', fontsize=11, fontweight='bold')

        # Hide unused subplots
        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.suptitle('Monte Carlo Tolerance Analysis', fontsize=13,
                     fontweight='bold')
        fig.tight_layout()
        return fig


class ToleranceAnalyzer:
    """Monte Carlo tolerance propagation for antenna geometry + solver.

    Each run creates ``n_samples`` perturbed copies of the antenna
    geometry (parameters drawn from independent Gaussian distributions)
    and evaluates the solver for each, building up a statistical picture
    of how manufacturing tolerances affect electrical performance.

    Parameters
    ----------
    geometry : AntennaGeometry
        Nominal design.
    solver_class : type
        Solver class (e.g. ``PatchAnalyticalSolver``, ``DipoleSolver``).
        Must accept ``(geometry, freq)`` as constructor arguments.
    freq : float or None
        Analysis frequency [Hz]. Uses ``geometry.freq_design`` if None.
    seed : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> from pylobe import RectangularPatch, PatchAnalyticalSolver
    >>> from pylobe.analysis.tolerance import ToleranceAnalyzer
    >>> patch = RectangularPatch(freq=2.4e9)
    >>> ta = ToleranceAnalyzer(patch, PatchAnalyticalSolver)
    >>> results = ta.run(n_samples=200, tolerances={'h': 0.05e-3})
    >>> results.summary()
    """

    def __init__(self, geometry, solver_class,
                 freq: float = None, seed: int = None):
        self.geometry = geometry
        self.solver_class = solver_class
        self.freq = freq if freq is not None else geometry.freq_design
        self.rng = np.random.default_rng(seed)

    def run(self, n_samples: int = 500,
            tolerances: Dict[str, float] = None,
            metrics: List[str] = None) -> ToleranceResults:
        """Run Monte Carlo analysis.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples.
        tolerances : dict of str → float
            ``{'param_name': sigma_value}`` — Gaussian standard deviations
            for each perturbed geometry attribute. Only attributes that
            exist on the geometry object will be perturbed.
            Example: ``{'h': 0.05e-3, 'eps_r': 0.1, 'W': 0.1e-3}``
        metrics : list of str or None
            Solver attributes to collect. Default:
            ``['resonant_frequency', 'directivity_dbi', 'gain_dbi',
               'radiation_efficiency']``

        Returns
        -------
        ToleranceResults
        """
        if tolerances is None:
            # Sensible defaults: ±0.1 mm PCB tolerance + 2% eps_r tolerance
            tolerances = {}
            for attr in ('W', 'L'):
                if hasattr(self.geometry, attr):
                    tolerances[attr] = 0.1e-3
            if hasattr(self.geometry, 'h'):
                tolerances['h'] = 0.05e-3
            if hasattr(self.geometry, 'eps_r'):
                tolerances['eps_r'] = 0.1

        if metrics is None:
            metrics = [
                'resonant_frequency',
                'directivity_dbi',
                'gain_dbi',
                'radiation_efficiency',
            ]

        # Collect nominal values
        nominal_solver = self.solver_class(self.geometry, self.freq)
        nominals = {}
        for m in metrics:
            try:
                val = getattr(nominal_solver, m)
                if callable(val):
                    val = val()
                nominals[m] = float(val)
            except Exception:
                nominals[m] = np.nan

        # Collect samples
        sample_data: Dict[str, List[float]] = {m: [] for m in metrics}
        failed = 0

        for _ in range(n_samples):
            try:
                perturbed = self._perturb(tolerances)
                solver = self.solver_class(perturbed, self.freq)
                for m in metrics:
                    val = getattr(solver, m)
                    if callable(val):
                        val = val()
                    sample_data[m].append(float(val))
            except Exception:
                failed += 1
                for m in metrics:
                    sample_data[m].append(np.nan)

        if failed > 0:
            warnings.warn(
                f"{failed}/{n_samples} Monte Carlo samples failed "
                "(geometry or solver error). Failed samples are NaN.",
                UserWarning, stacklevel=2,
            )

        # Build results
        results = {}
        for m in metrics:
            s = np.array(sample_data[m], dtype=float)
            valid = s[~np.isnan(s)]
            if len(valid) == 0:
                continue
            results[m] = ToleranceResult(
                param_name=m,
                nominal=nominals.get(m, np.nan),
                mean=float(np.mean(valid)),
                std=float(np.std(valid)),
                p5=float(np.percentile(valid, 5)),
                p95=float(np.percentile(valid, 95)),
                samples=valid,
            )

        return ToleranceResults(results, n_samples=n_samples,
                                tolerances=tolerances)

    def _perturb(self, tolerances: Dict[str, float]):
        """Return a shallow copy of the geometry with perturbed attributes."""
        import copy
        geom = copy.copy(self.geometry)
        # Re-run __init__ is complex; instead patch attributes directly and
        # recompute derived quantities for RectangularPatch-like geometries.
        for attr, sigma in tolerances.items():
            if hasattr(geom, attr):
                nominal = getattr(geom, attr)
                delta = self.rng.normal(0.0, sigma)
                setattr(geom, attr, float(nominal) + delta)

        # Recompute derived patch dimensions if this is a rectangular patch
        if hasattr(geom, '_rebuild_derived'):
            geom._rebuild_derived()
        elif hasattr(geom, 'eps_eff') and hasattr(geom, 'W') and hasattr(geom, 'h'):
            self._recompute_patch_dims(geom)

        return geom

    @staticmethod
    def _recompute_patch_dims(geom) -> None:
        """Recompute eps_eff, delta_L, L, G1, y0 after parameter perturbation."""
        from pylobe.constants import C0, PI
        try:
            eps_r = geom.eps_r
            h = geom.h
            W = geom.W
            freq = geom.freq_design

            geom.eps_eff = (
                (eps_r + 1.0) / 2.0
                + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)
            )
            # Hammerstad length extension (Balanis Eq. 14-2)
            geom.delta_L = (
                0.412 * h
                * (geom.eps_eff + 0.3) * (W / h + 0.264)
                / ((geom.eps_eff - 0.258) * (W / h + 0.8))
            )
            L_eff = C0 / (2.0 * freq * np.sqrt(geom.eps_eff))
            geom.L = L_eff - 2.0 * geom.delta_L
            lambda0 = C0 / freq
            k0 = 2.0 * PI * freq / C0
            geom.G1 = (W / (120.0 * lambda0)) * (1.0 - (k0 * h) ** 2 / 24.0)
            geom.Rin_edge = 1.0 / (2.0 * geom.G1)
            ratio = np.clip(geom.feed_impedance / geom.Rin_edge, 0.0, 1.0)
            if geom.inset_feed:
                geom.y0 = (geom.L / PI) * np.arccos(np.sqrt(ratio))
        except Exception:
            pass  # Partial recompute is better than no recompute
