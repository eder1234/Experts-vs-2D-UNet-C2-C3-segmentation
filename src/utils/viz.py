"""
Quick-and-dirty visual QA helpers.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Return an RGB overlay of `mask` on grayscale `img`.
    `img`  : (H, W) float [0,1]  |  `mask`: (H, W) bool/0-1
    """
    img_rgb = np.stack([img, img, img], axis=-1)
    mask_rgb = np.zeros_like(img_rgb)
    mask_rgb[..., 0] = mask  # red channel
    return np.clip((1 - alpha) * img_rgb + alpha * mask_rgb, 0, 1)


def save_triplet(
    mag_slice: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    Save a 1×3 panel: raw magnitude slice | pred overlay | GT overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["Magnitude", "Prediction", "Ground-truth"]
    imgs = [
        mag_slice,
        overlay(mag_slice, pred_mask),
        overlay(mag_slice, gt_mask),
    ]
    for ax, t, im in zip(axes, titles, imgs):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t)
        ax.axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
