# src/inference/visualize_test.py
"""
Visualise random test subjects: overlay GT (yellow) and
prediction (blue) on phase & magnitude slice-0.

Run:
    python -m src.inference.visualize_test --num 3 --out outputs/figures/test_vis.png
"""

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


# ---------- helper ---------------------------------------------------- #
def load_subject(path: Path, pred_dir: Path):
    """Return phase-0, mag-0, GT mask (full), pred mask (full)."""
    phase = np.load(path / "phase.npy")  # (32,240,240)
    mag   = np.load(path / "mag.npy")    # (32,240,240)
    gt    = np.load(path / "mask.npy")   # (240,240)

    pred_file = pred_dir / f"{path.name}_pred.npy"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_file}")
    pred  = np.load(pred_file)           # (240,240)

    return phase[0], mag[0], gt, pred


def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha=.4):
    """
    Overlay GT in yellow and pred in blue on a grayscale background.
    """
    img_norm = (img - img.min()) / (img.ptp() + 1e-8)
    rgb = np.stack([img_norm]*3, axis=-1)

    # GT → yellow  (R+G)
    mask_gt = (gt > 0.5)
    rgb[..., 0][mask_gt] = (1-alpha) * rgb[..., 0][mask_gt] + alpha  # red
    rgb[..., 1][mask_gt] = (1-alpha) * rgb[..., 1][mask_gt] + alpha  # green

    # Pred → blue
    mask_pred = (pred > 0.5)
    rgb[..., 2][mask_pred] = (1-alpha) * rgb[..., 2][mask_pred] + alpha  # blue

    # where both overlap → white-ish (R+G+B)
    return np.clip(rgb, 0, 1)


# ---------- main ------------------------------------------------------ #
def main(test_dir: str, pred_dir: str, num: int, out_path: str):
    test_dir = Path(test_dir)
    pred_dir = Path(pred_dir)
    subjects: List[Path] = [p for p in test_dir.iterdir() if p.is_dir()]

    if num > len(subjects):
        num = len(subjects)
    random.shuffle(subjects)
    chosen = subjects[:num]

    rows, cols = 2, num
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 8), squeeze=False)

    for idx, subj in enumerate(chosen):
        phase0, mag0, gt, pred = load_subject(subj, pred_dir)

        axes[0, idx].imshow(overlay(phase0, gt, pred))
        axes[0, idx].set_title(f"{subj.name} – phase0"), axes[0, idx].axis("off")

        axes[1, idx].imshow(overlay(mag0, gt, pred))
        axes[1, idx].set_title(f"{subj.name} – mag0"), axes[1, idx].axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Figure saved → {out_path.resolve()}")


# ---------- CLI ------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="data/test", help="folder with test subjects")
    parser.add_argument("--pred_dir", default="outputs/preds", help="folder with *_pred.npy files")
    parser.add_argument("--num", type=int, default=3, help="how many random subjects to show")
    parser.add_argument("--out", default="outputs/figures/test_vis.png", help="output PNG path")
    args = parser.parse_args()

    main(args.test_dir, args.pred_dir, args.num, args.out)
