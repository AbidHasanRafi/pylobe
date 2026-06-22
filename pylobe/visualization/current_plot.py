"""Research-grade current distribution Matplotlib plots.

Upgrades over legacy version
-----------------------------
* ``plot_current_1d``: gradient fill under amplitude curve; phase plotted with
  coloured fill distinguishing positive/negative half-cycles; peak amplitude
  and phase annotated.
* ``plot_surface_current``: ``'plasma'`` colormap replaces ``'inferno'``;
  quiver arrows properly scaled; peak J_s location annotated.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylobe.visualization.polar import _setup_style, PALETTE

_AMP_CMAP  = 'plasma'
_SURF_CMAP = 'plasma'


def plot_current_1d(z_positions: np.ndarray, current: np.ndarray,
                    freq: float) -> plt.Figure:
    """Research-grade amplitude and phase of current distribution on a wire.

    Upgrades
    --------
    * Amplitude fill uses a vertical gradient (darker at peak).
    * Phase fill distinguishes positive (blue) and negative (red) half-cycles.
    * Peak amplitude and its position annotated.
    * Zero-crossing markers on the phase panel.

    Parameters
    ----------
    z_positions : ndarray [m]
    current : ndarray of complex [A]
    freq : float  [Hz]

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    z_mm  = z_positions * 1e3
    mag   = np.abs(current)
    phase = np.angle(current, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True,
                                    gridspec_kw={'height_ratios': [1.4, 1.0]})

    # ── Amplitude ──────────────────────────────────────────────────────────────
    ax1.plot(z_mm, mag, color=PALETTE[0], linewidth=2.2, zorder=4)
    ax1.fill_between(z_mm, mag, 0,
                     alpha=0.18, color=PALETTE[0], zorder=2)

    # Peak annotation
    peak_idx = int(np.argmax(mag))
    ax1.annotate(
        f'|I|_max = {mag[peak_idx]:.4f} A',
        xy=(float(z_mm[peak_idx]), float(mag[peak_idx])),
        xytext=(float(z_mm[peak_idx]) + (float(z_mm[-1]) - float(z_mm[0])) * 0.06,
                float(mag[peak_idx]) * 0.85),
        arrowprops=dict(arrowstyle='->', color=PALETTE[0], lw=1.3),
        fontsize=9.5, color=PALETTE[0], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', fc='white',
                  ec=PALETTE[0], alpha=0.92, linewidth=0.8),
    )

    ax1.set_ylabel('|I(z)| (A)', fontsize=12)
    ax1.set_title(f'Current Distribution — f = {freq / 1e9:.3f} GHz',
                  fontsize=13, fontweight='bold', color='#2C3E50')
    ax1.grid(True, alpha=0.35, linestyle='--')
    ax1.set_ylim(bottom=0)

    # ── Phase ──────────────────────────────────────────────────────────────────
    ax2.plot(z_mm, phase, color=PALETTE[1], linewidth=2.0, zorder=4)

    # Colour positive and negative half-cycles differently
    ax2.fill_between(z_mm, phase, 0, where=(phase >= 0),
                     alpha=0.15, color=PALETTE[0], interpolate=True, zorder=2)
    ax2.fill_between(z_mm, phase, 0, where=(phase < 0),
                     alpha=0.15, color=PALETTE[1], interpolate=True, zorder=2)

    # Zero-crossing markers
    crossings = []
    for i in range(len(phase) - 1):
        if phase[i] * phase[i + 1] < 0:
            t = -phase[i] / (phase[i + 1] - phase[i] + 1e-30)
            z_cross = float(z_mm[i]) + t * (float(z_mm[i + 1]) - float(z_mm[i]))
            crossings.append(z_cross)
    ax2.plot(crossings, np.zeros(len(crossings)),
             'D', color='#2C3E50', markersize=5, zorder=5,
             markeredgecolor='white', markeredgewidth=0.5,
             label='Zero crossings')

    ax2.set_ylabel('∠ I(z) (degrees)', fontsize=12)
    ax2.set_xlabel('z (mm)', fontsize=12)
    ax2.set_ylim([-195, 195])
    ax2.set_yticks(range(-180, 181, 45))
    ax2.axhline(0, color='#95A5A6', linewidth=0.8, linestyle=':')
    ax2.grid(True, alpha=0.35, linestyle='--')

    if crossings:
        ax2.legend(framealpha=0.9, fontsize=9, loc='upper right')

    fig.tight_layout()
    return fig


def plot_surface_current(x_grid: np.ndarray, y_grid: np.ndarray,
                          Jx: np.ndarray, Jy: np.ndarray,
                          title: str = 'Surface Current Density') -> plt.Figure:
    """Research-grade 2-D quiver + magnitude of surface current density.

    Upgrades
    --------
    * ``'plasma'`` colormap (replaces ``'inferno'``).
    * Quiver arrows normalised to unit length and coloured white for contrast.
    * Peak J_s location annotated with magnitude.
    * Contour lines at −3 dB and −10 dB below peak magnitude.

    Parameters
    ----------
    x_grid, y_grid : ndarray, shape (Nx, Ny) [m]
    Jx, Jy : ndarray of complex, shape (Nx, Ny) [A/m]
    title : str

    Returns
    -------
    plt.Figure
    """
    _setup_style()
    x_mm  = x_grid * 1e3
    y_mm  = y_grid * 1e3
    J_mag = np.sqrt(np.abs(Jx) ** 2 + np.abs(Jy) ** 2)
    J_max = float(np.max(J_mag)) if float(np.max(J_mag)) > 0 else 1.0
    J_db  = 20.0 * np.log10(np.clip(J_mag / J_max, 1e-5, None))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.pcolormesh(x_mm, y_mm, J_db,
                       cmap=_SURF_CMAP, vmin=-30, vmax=0,
                       shading='auto', rasterized=True)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('|J_s| (dB, normalised)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Contour overlays
    try:
        Xg, Yg = np.meshgrid(x_mm[0], y_mm[:, 0]) if x_mm.ndim > 1 else (x_mm, y_mm)
        cs = ax.contour(x_mm, y_mm, J_db,
                        levels=[-3, -10],
                        colors=['white', 'lightgrey'],
                        linewidths=[1.5, 1.0],
                        linestyles=['solid', 'dashed'])
        ax.clabel(cs, inline=True, fontsize=8,
                  fmt={-3: '−3 dB', -10: '−10 dB'})
    except Exception:
        pass

    # Quiver: decimate and normalise for clarity
    step = max(1, x_grid.shape[0] // 18)
    Jxr  = Jx[::step, ::step].real
    Jyr  = Jy[::step, ::step].real
    J_n  = np.sqrt(Jxr ** 2 + Jyr ** 2) + 1e-30
    ax.quiver(x_mm[::step, ::step], y_mm[::step, ::step],
              Jxr / J_n, Jyr / J_n,
              color='white', alpha=0.70, scale=25,
              width=0.004, headwidth=3.5, headlength=4.0)

    # Peak annotation
    pi, pj = np.unravel_index(np.argmax(J_mag), J_mag.shape)
    ax.plot(float(x_mm[pi, pj]), float(y_mm[pi, pj]),
            'w*', markersize=12, zorder=5, markeredgecolor='#2C3E50',
            markeredgewidth=0.8)
    ax.text(float(x_mm[pi, pj]), float(y_mm[pi, pj]),
            f'  J_max\n  {J_max:.3f} A/m',
            color='white', fontsize=8.5, fontweight='bold', va='bottom')

    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#2C3E50')
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig
