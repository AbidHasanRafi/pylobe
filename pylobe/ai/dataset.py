"""Dataset generation for surrogate model training."""
import numpy as np


PATCH_FEATURE_NAMES = ['freq_hz', 'eps_r', 'h_m']
PATCH_TARGET_NAMES  = ['gain_dbi', 'bandwidth_frac', 'Rin_edge_ohm', 'eps_eff']


def generate_patch_dataset(freq_range: tuple = (1e9, 6e9),
                            eps_r_range: tuple = (2.0, 10.0),
                            substrate_range: tuple = None,
                            h_range: tuple = (0.5e-3, 3e-3),
                            n_samples: int = 500,
                            seed: int = 42) -> tuple:
    """Generate labelled dataset for rectangular patch surrogate training.

    Samples geometry parameters, runs analytical solver, extracts features.

    Parameters
    ----------
    freq_range : tuple (f_lo, f_hi) [Hz]
    eps_r_range : tuple
        Substrate relative permittivity range. Alias: ``substrate_range``.
    substrate_range : tuple or None
        Alias for ``eps_r_range``.  Takes precedence when provided.
    h_range : tuple [m]
    n_samples : int
    seed : int

    Returns
    -------
    tuple (X, y, feature_names, target_names)
        X : ndarray, shape (N, 3)  — [freq_hz, eps_r, h_m]
        y : ndarray, shape (N, 4)  — [gain_dbi, bandwidth_frac, Rin_edge_ohm, eps_eff]
        feature_names : list[str]
        target_names  : list[str]
    """
    if substrate_range is not None:
        eps_r_range = substrate_range
    from pylobe.geometry.patch import RectangularPatch
    from pylobe.solver.analytical.patch_solver import PatchAnalyticalSolver

    rng = np.random.default_rng(seed)
    freqs   = rng.uniform(*freq_range, n_samples)
    eps_rs  = rng.uniform(*eps_r_range, n_samples)
    hs      = rng.uniform(*h_range, n_samples)

    X_list, y_list = [], []
    for f, er, h in zip(freqs, eps_rs, hs):
        try:
            patch  = RectangularPatch(freq=f, eps_r=er, h=h)
            solver = PatchAnalyticalSolver(patch, freq=f)
            gain   = solver.directivity_dbi
            bw     = patch.bandwidth_approx
            rin    = patch.Rin_edge
            eps_eff = patch.eps_eff

            X_list.append([f, er, h])
            y_list.append([gain, bw, rin, eps_eff])
        except Exception:
            continue

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, PATCH_FEATURE_NAMES, PATCH_TARGET_NAMES


def generate_dipole_dataset(freq_range: tuple = (0.1e9, 3e9),
                             lf_range: tuple = (0.3, 0.7),
                             n_samples: int = 300,
                             seed: int = 42) -> tuple:
    """Generate (X, y) dataset for half-wave dipole surrogate.

    Parameters
    ----------
    freq_range : tuple [Hz]
    lf_range : tuple
        Length factor range.
    n_samples : int

    Returns
    -------
    tuple (X, y)
        X : ndarray, shape (N, 2)  — [freq, length_factor]
        y : ndarray, shape (N, 3)  — [directivity_dbi, Rr, Xin_imag]
    """
    from pylobe.geometry.dipole import HalfWaveDipole
    from pylobe.solver.analytical.dipole_solver import DipoleSolver

    rng   = np.random.default_rng(seed)
    freqs = rng.uniform(*freq_range, n_samples)
    lfs   = rng.uniform(*lf_range,   n_samples)

    X_list, y_list = [], []
    for f, lf in zip(freqs, lfs):
        try:
            dipole  = HalfWaveDipole(freq=f, length_factor=lf)
            solver  = DipoleSolver(dipole, freq=f)
            D_dbi   = solver.directivity_dbi
            Rr      = solver.radiation_resistance
            Zin     = solver.input_impedance
            X_list.append([f, lf])
            y_list.append([D_dbi, Rr, Zin.imag])
        except Exception:
            continue

    return np.array(X_list), np.array(y_list)


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_fraction: float = 0.2,
                     seed: int = 0) -> tuple:
    """Simple random train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
