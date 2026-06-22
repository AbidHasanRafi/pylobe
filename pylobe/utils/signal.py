"""Signal processing utilities for FDTD post-processing."""
import numpy as np
from scipy.signal import welch


def fft_with_freq(time_signal: np.ndarray, dt: float):
    """FFT with frequency axis.

    Applies Hann window before FFT to reduce spectral leakage.
    Returns only positive frequencies.

    Parameters
    ----------
    time_signal : ndarray, shape (N,)
        Time-domain signal.
    dt : float
        Time step [s].

    Returns
    -------
    freq : ndarray
        Frequency array [Hz], positive only.
    spectrum : ndarray
        Complex spectrum (one-sided, normalised).
    """
    n = len(time_signal)
    window = np.hanning(n)
    windowed = time_signal * window
    spectrum = np.fft.rfft(windowed)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, spectrum


def psd(time_signal: np.ndarray, dt: float):
    """Power spectral density via Welch's method.

    Parameters
    ----------
    time_signal : ndarray
        Time-domain signal.
    dt : float
        Time step [s].

    Returns
    -------
    freq : ndarray
        Frequency [Hz].
    psd_values : ndarray
        Power spectral density [V²/Hz or A²/Hz].
    """
    fs = 1.0 / dt
    freq, psd_values = welch(time_signal, fs=fs, nperseg=min(256, len(time_signal)))
    return freq, psd_values


def nextpow2(n: int) -> int:
    """Smallest power of 2 greater than or equal to n.

    Parameters
    ----------
    n : int
        Input integer.

    Returns
    -------
    int
    """
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()
