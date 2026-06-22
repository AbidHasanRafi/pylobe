"""Visualization style presets — publication-quality antenna figures.

All presets enforce **600 DPI** on save (``savefig.dpi = 600``) while
keeping the interactive-display DPI at 120 so Jupyter/Qt windows are
not huge.  Plotly static exports also use 600 DPI (scale = 600/96).

Available presets
-----------------
``'ieee'``         IEEE Transactions single-column (88 mm), serif, 600 dpi.
``'ieee_wide'``    IEEE Transactions two-column (181 mm), serif, 600 dpi.
``'presentation'`` Beamer/PowerPoint, 14 pt sans-serif, 600 dpi on save.
``'dark'``         Dark background — demos and dashboards, 600 dpi on save.
``'default'``      Screen-friendly defaults with 600 dpi on save.

Usage
-----
>>> from pylobe.visualization.style import set_style, export_fig
>>> set_style('ieee')
>>> fig = plot_polar(...)
>>> export_fig(fig, 'pattern.pdf')   # saves at 600 DPI automatically
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ── Minimum export resolution ─────────────────────────────────────────────────
EXPORT_DPI    = 600      # all matplotlib savefig calls
PLOTLY_SCALE  = EXPORT_DPI / 96.0   # kaleido scale factor (Plotly default = 96 dpi)

# ── Style presets ─────────────────────────────────────────────────────────────
_STYLES = {
    'ieee': {
        'figure.figsize':      (3.46, 2.8),
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'DejaVu Serif'],
        'font.size':           8,
        'axes.titlesize':      8,
        'axes.labelsize':      8,
        'xtick.labelsize':     7,
        'ytick.labelsize':     7,
        'legend.fontsize':     7,
        'lines.linewidth':     1.5,
        'axes.linewidth':      0.8,
        'grid.alpha':          0.4,
        'grid.linewidth':      0.5,
        'figure.dpi':          120,
        'savefig.dpi':         EXPORT_DPI,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.02,
        'axes.prop_cycle': matplotlib.cycler(
            color=['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#56B4E9', '#E69F00', '#000000']),
    },
    'ieee_wide': {
        'figure.figsize':      (7.13, 3.5),
        'font.family':         'serif',
        'font.serif':          ['Times New Roman', 'DejaVu Serif'],
        'font.size':           9,
        'axes.titlesize':      9,
        'axes.labelsize':      9,
        'xtick.labelsize':     8,
        'ytick.labelsize':     8,
        'legend.fontsize':     8,
        'lines.linewidth':     1.5,
        'axes.linewidth':      0.8,
        'grid.alpha':          0.4,
        'grid.linewidth':      0.5,
        'figure.dpi':          120,
        'savefig.dpi':         EXPORT_DPI,
        'savefig.bbox':        'tight',
        'savefig.pad_inches':  0.02,
        'axes.prop_cycle': matplotlib.cycler(
            color=['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#56B4E9', '#E69F00', '#000000']),
    },
    'presentation': {
        'figure.figsize':      (10, 7),
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Arial', 'DejaVu Sans'],
        'font.size':           14,
        'axes.titlesize':      16,
        'axes.labelsize':      14,
        'xtick.labelsize':     12,
        'ytick.labelsize':     12,
        'legend.fontsize':     12,
        'lines.linewidth':     2.5,
        'axes.linewidth':      1.5,
        'grid.alpha':          0.3,
        'grid.linewidth':      0.8,
        'figure.dpi':          120,
        'savefig.dpi':         EXPORT_DPI,
        'savefig.bbox':        'tight',
        'axes.prop_cycle': matplotlib.cycler(
            color=['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#56B4E9', '#E69F00', '#000000']),
    },
    'dark': {
        'figure.figsize':      (10, 7),
        'figure.facecolor':    '#1A1A2E',
        'axes.facecolor':      '#16213E',
        'axes.edgecolor':      '#4A4A7A',
        'axes.labelcolor':     '#E0E0E0',
        'xtick.color':         '#C0C0C0',
        'ytick.color':         '#C0C0C0',
        'text.color':          '#E0E0E0',
        'grid.color':          '#2A2A4A',
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Arial', 'DejaVu Sans'],
        'font.size':           12,
        'axes.titlesize':      14,
        'axes.labelsize':      12,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.fontsize':     10,
        'legend.facecolor':    '#1A1A2E',
        'legend.edgecolor':    '#4A4A7A',
        'lines.linewidth':     2.2,
        'axes.linewidth':      1.2,
        'grid.alpha':          0.4,
        'grid.linewidth':      0.6,
        'figure.dpi':          120,
        'savefig.dpi':         EXPORT_DPI,
        'savefig.bbox':        'tight',
        'axes.prop_cycle': matplotlib.cycler(
            color=['#4FC3F7', '#FF8A65', '#A5D6A7', '#CE93D8',
                   '#FFF176', '#80DEEA', '#FFAB91']),
    },
    'default': {
        'figure.figsize':      (10, 7),
        'figure.facecolor':    'white',
        'axes.facecolor':      '#F8F9FA',
        'axes.edgecolor':      '#555555',
        'axes.labelcolor':     '#2C3E50',
        'xtick.color':         '#2C3E50',
        'ytick.color':         '#2C3E50',
        'text.color':          '#2C3E50',
        'grid.color':          '#888888',
        'font.family':         'sans-serif',
        'font.sans-serif':     ['DejaVu Sans', 'Arial'],
        'font.size':           11,
        'axes.titlesize':      13,
        'axes.labelsize':      12,
        'xtick.labelsize':     10,
        'ytick.labelsize':     10,
        'legend.fontsize':     10,
        'lines.linewidth':     2.0,
        'axes.linewidth':      1.2,
        'grid.alpha':          0.40,
        'grid.linewidth':      0.6,
        'figure.dpi':          120,
        'savefig.dpi':         EXPORT_DPI,
        'savefig.bbox':        'tight',
        'axes.prop_cycle': matplotlib.cycler(
            color=['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#56B4E9', '#E69F00', '#000000']),
    },
}

_CURRENT_STYLE = 'default'


def set_style(name: str = 'default') -> None:
    """Apply a named style globally (updates ``matplotlib.rcParams``).

    Parameters
    ----------
    name : str
        One of ``'ieee'``, ``'ieee_wide'``, ``'presentation'``,
        ``'dark'``, ``'default'``.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> set_style('ieee')      # publication quality, 88 mm column, 600 dpi
    >>> set_style('dark')      # dark background for screen demos
    >>> set_style('default')   # restore screen-friendly defaults
    """
    global _CURRENT_STYLE
    # Friendly aliases
    _ALIASES = {'paper': 'ieee_wide', 'journal': 'ieee', 'screen': 'default'}
    name = _ALIASES.get(name, name)
    if name not in _STYLES:
        raise ValueError(
            f'Unknown style {name!r}. '
            f'Available: {list(_STYLES.keys()) + list(_ALIASES.keys())}'
        )
    plt.rcParams.update(_STYLES[name])
    _CURRENT_STYLE = name


def get_style() -> str:
    """Return the name of the active style."""
    return _CURRENT_STYLE


def export_fig(fig, filename: str, dpi: int = None,
               fmt: str = None) -> None:
    """Save a Matplotlib or Plotly figure at ≥ 600 DPI.

    For Matplotlib: calls ``fig.savefig()`` with ``dpi=max(600, dpi)``.
    For Plotly HTML: calls ``fig.write_html()`` (no kaleido needed).
    For Plotly raster/vector: calls ``fig.write_image()`` via kaleido at
    scale = ``max(600, dpi) / 96``.

    Parameters
    ----------
    fig : matplotlib Figure or plotly Figure
    filename : str
        Output path; extension sets the format unless *fmt* is given.
        Supported: ``.png``, ``.pdf``, ``.svg``, ``.eps``, ``.html``.
    dpi : int or None
        Minimum 600.  Pass a higher value for ultra-high-resolution export.
    fmt : str or None
        Override the format (e.g. ``'pdf'``).
    """
    effective_dpi = max(EXPORT_DPI, dpi or 0)

    try:
        import plotly.graph_objects as go
        _is_plotly = isinstance(fig, go.Figure)
    except ImportError:
        _is_plotly = False

    if _is_plotly:
        ext = fmt or filename.rsplit('.', 1)[-1].lower()
        if ext == 'html':
            fig.write_html(filename, include_plotlyjs='cdn')
        else:
            _export_plotly(fig, filename, dpi=effective_dpi, fmt=fmt)
    else:
        # bbox_inches='tight' triggers matplotlib's tight-layout probe, which
        # warns when the figure contains polar axes.  The layout is already set
        # correctly by explicit GridSpec spacing, so the warning is harmless.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='This figure includes Axes that are not compatible'
                        ' with tight_layout',
                category=UserWarning,
            )
            fig.savefig(filename, dpi=effective_dpi, format=fmt,
                        bbox_inches='tight')


def _export_plotly(fig, filename: str, dpi: int = EXPORT_DPI,
                   fmt: str = None) -> None:
    """Export a Plotly figure to a static image via kaleido at ≥ 600 DPI."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        raise ImportError(
            'Exporting Plotly figures requires kaleido.\n'
            'Install:  pip install kaleido   or   pip install pylobe[vis]'
        )
    ext   = fmt or filename.rsplit('.', 1)[-1].lower()
    scale = max(PLOTLY_SCALE, dpi / 96.0)
    fig.write_image(filename, format=ext, scale=scale)


def _get_current_rcparams() -> dict:
    keys = ['figure.figsize', 'font.size', 'lines.linewidth',
            'figure.dpi', 'savefig.dpi']
    return {k: plt.rcParams[k] for k in keys if k in plt.rcParams}


# Apply default style on import
set_style('default')
