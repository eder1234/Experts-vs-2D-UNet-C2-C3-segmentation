# src/utils/temporal.py
import numpy as np

__all__ = ["reorder_temporal_images"]

def reorder_temporal_images(phase: np.ndarray,
                            mag:   np.ndarray,
                            shift: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Cyclically shift the temporal dimension of 32-slice phase & magnitude stacks.

    Parameters
    ----------
    phase, mag : (32, H, W)  numpy arrays
    shift      : 0‥31 → deterministic cyclic shift
                 -1   → random permutation (kept for completeness but *not* used)

    Returns
    -------
    phase_s, mag_s : shifted copies
    idx            : list of slice indices that were selected
    """
    assert phase.shape[0] == mag.shape[0] == 32

    if shift == -1:
        idx = np.random.permutation(32).tolist()
    else:
        idx = list(range(shift, 32)) + list(range(shift))

    return phase[idx], mag[idx], idx
