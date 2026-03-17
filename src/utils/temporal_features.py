# src/utils/temporal_features.py

import numpy as np

def _ensure_time_first(x: np.ndarray) -> np.ndarray:
    """Accept (T,H,W) or (C,H,W). Return (T,H,W)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    # assume time is first; keep for clarity/future extension
    return x

def temporal_std(x: np.ndarray) -> np.ndarray:
    """
    Per-pixel temporal standard deviation (shift-invariant).
    x: (T,H,W) time series.
    returns: (H,W)
    """
    x = _ensure_time_first(x)
    return np.std(x, axis=0)

def temporal_tv(x: np.ndarray) -> np.ndarray:
    """
    Per-pixel temporal total variation with circular wrap (shift-invariant).
    x: (T,H,W)
    returns: (H,W)
    """
    x = _ensure_time_first(x)
    x_roll = np.roll(x, shift=-1, axis=0)
    return np.sum(np.abs(x_roll - x), axis=0)

def dft_bandpower_excl_dc(x: np.ndarray) -> np.ndarray:
    """
    Per-pixel DFT band power excluding DC (shift-invariant).
    Uses rfft along time (real FFT).
    x: (T,H,W)
    returns: (H,W)
    """
    x = _ensure_time_first(x)
    X = np.fft.rfft(x, axis=0)
    # bins: 0..T//2; exclude k=0 (DC)
    mag2 = (np.abs(X) ** 2)
    if mag2.shape[0] <= 1:
        return np.zeros_like(mag2[0])
    return np.sum(mag2[1:], axis=0)

def dft_magnitudes_bins(x: np.ndarray, bins=(1, 2, 3)) -> np.ndarray:
    """
    Per-pixel DFT magnitudes at specific bins (shift-invariant).
    x: (T,H,W)
    returns: (len(bins), H, W)
    """
    x = _ensure_time_first(x)
    X = np.fft.rfft(x, axis=0)
    T = x.shape[0]
    max_bin = X.shape[0] - 1  # rfft length
    sel = []
    for k in bins:
        if 0 <= k <= max_bin:
            sel.append(np.abs(X[k]))
        else:
            sel.append(np.zeros_like(X[0]))
    return np.stack(sel, axis=0)
