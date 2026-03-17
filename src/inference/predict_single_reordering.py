# src/inference/predict_single_reordering.py
"""
Predict a single test subject and visualise the result.

Example
-------
python -m src.inference.predict_single_reordering \
       --subject Patient_2202021349 \
       --config  config.yaml \
       --weights outputs/checkpoints/best.pt \
       --out_dir outputs/preds_single \
       --fig     outputs/figures/Patient_2202021349.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt

import numpy as np

def reorder_temporal_images(phase: np.ndarray, mag: np.ndarray, shift: int):
    """
    Reorder the temporal sequence of phase and magnitude images.

    Parameters
    ----------
    phase : np.ndarray
        Phase image sequence of shape (T, H, W), where T = 32.
    mag : np.ndarray
        Magnitude image sequence of shape (T, H, W), where T = 32.
    shift : int
        - If shift == -1: random shuffle of the temporal images.
        - If shift ∈ [0, 31]: circular shift by 'shift' positions.

    Returns
    -------
    phase_reordered : np.ndarray
        Reordered phase images.
    mag_reordered : np.ndarray
        Reordered magnitude images.
    indices : list of int
        List of indices used for reordering.
    """
    assert phase.shape[0] == 32 and mag.shape[0] == 32, "Expected 32 images per sequence"
    assert phase.shape == mag.shape, "Phase and magnitude must have same shape"

    if shift == -1:
        indices = np.random.permutation(32).tolist()
    elif 0 <= shift < 32:
        indices = list(range(shift, 32)) + list(range(shift))
    else:
        raise ValueError("Shift must be -1 (for random shuffle) or an integer in [0, 31]")

    phase_reordered = phase[indices]
    mag_reordered = mag[indices]
    return phase_reordered, mag_reordered, indices


# --------------------------------------------------------------------- #
class SingleSubjectDataset(CSFVolumeDataset):
    """Wrap CSFVolumeDataset but keep only one subject (by folder name)."""

    def __init__(self,
                 root_dir: Path,
                 subject_id: str,
                 crop_size: int):
        super().__init__(root_dir=root_dir,
                         split="test",
                         crop_size=crop_size)

        idxs = [i for i, p in enumerate(self.subjects)
                if p.name == subject_id]
        if not idxs:
            raise FileNotFoundError(f"No subject '{subject_id}' under {root_dir}")
        self.keep = idxs[0]
        self.subj_path = self.subjects[self.keep]          # ← NEW: keep raw path

    def __len__(self): return 1
    def __getitem__(self, idx): return super().__getitem__(self.keep)


# --------------------------------------------------------------------- #
def pad_to_full(mask_crop: np.ndarray, crop_size: int,
                full_size: int = 240) -> np.ndarray:
    pad = (full_size - crop_size) // 2
    full = np.zeros((full_size, full_size), dtype=mask_crop.dtype)
    full[pad: pad + crop_size, pad: pad + crop_size] = mask_crop
    return full


def overlay(img: np.ndarray, gt: np.ndarray,
            pred: np.ndarray, alpha: float = .4) -> np.ndarray:
    img_norm = (img - img.min()) / (img.ptp() + 1e-8)
    rgb = np.stack([img_norm] * 3, -1)

    gt_mask = gt > .5
    rgb[..., 0][gt_mask] = (1 - alpha) * rgb[..., 0][gt_mask] + alpha
    rgb[..., 1][gt_mask] = (1 - alpha) * rgb[..., 1][gt_mask] + alpha

    pr_mask = pred > .5
    rgb[..., 2][pr_mask] = (1 - alpha) * rgb[..., 2][pr_mask] + alpha
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------------- #
def main(subject: str,
         cfg_path: str | Path,
         weights_path: str | Path,
         out_dir: str | Path,
         fig_path: str | Path | None,
         thresh: float,
         shift: int) -> None:

    cfg = load_yaml(cfg_path)
    crop = cfg["data"]["crop_size"]

    # ---------------- dataset (training preprocessing) ---------------- #
    ds = SingleSubjectDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["test_dir"],
        subject_id=subject,
        crop_size=crop,
    )
    sample   = ds[0]
    img = sample["image"]              # (64, 80, 80)
    phase = img[:32]                   # (32, 80, 80)
    mag   = img[32:]                   # (32, 80, 80)

    # --- Step 2: Apply reordering ---
    phase_reordered, mag_reordered, idxs = reorder_temporal_images(phase.numpy(), mag.numpy(), shift=shift)

    # --- Step 3: Stack and convert back to tensor ---
    img_reordered = np.concatenate([phase_reordered, mag_reordered], axis=0)  # (64, 80, 80)
    img = torch.from_numpy(img_reordered).unsqueeze(0).float()

    gt_crop  = sample["mask"].squeeze().numpy()            # (80,80)
    gt_full  = pad_to_full(gt_crop, crop)                  # ← NEW
    phase0_full = np.load(ds.subj_path / "phase.npy")[0]   # (240,240)
    mag0_full   = np.load(ds.subj_path / "mag.npy")[0]

    # --------------------------- model ------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=cfg["model"]["in_channels"],
                   out_channels=cfg["model"]["out_channels"],
                   base_channels=cfg["model"]["base_channels"]).to(device)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    ckpt = load_ckpt(weights_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --------------------------- predict ----------------------------- #
    with torch.no_grad():
        logits = model(img.to(device))
        probs_crop = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (80,80)

    probs_full = pad_to_full(probs_crop, crop)
    pred_mask  = (probs_full >= thresh).astype(np.uint8)

    # ---------------------- save .npy outputs ------------------------ #
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{subject}_probs.npy", probs_full.astype(np.float32))
    np.save(out_dir / f"{subject}_pred.npy",  pred_mask.astype(np.uint8))
    print(f"Saved → {out_dir}/{subject}_probs.npy and _pred.npy")

    # ------------------------ visualisation -------------------------- #
    if fig_path:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #### include the title as the indices
        fig.suptitle(f"Subject: {subject} | Shift: {shift} | Indices: {idxs}")
        ax[0].imshow(pred_mask, cmap="gray")
        ax[0].set_title("Binary mask"); ax[0].axis("off")

        ax[1].imshow(overlay(phase0_full, gt_full, pred_mask))   # ← FIX
        ax[1].set_title("Overlay – phase₀"); ax[1].axis("off")

        ax[2].imshow(overlay(mag0_full, gt_full, pred_mask))     # ← FIX
        ax[2].set_title("Overlay – mag₀");   ax[2].axis("off")

        fig.tight_layout()
        fig_path = Path(fig_path)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=200)
        print(f"Figure saved → {fig_path.resolve()}")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True,
                   help="folder under data/test/, e.g. Patient_2202021349")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--weights", default="outputs/checkpoints/best.pt")
    p.add_argument("--out_dir", default="outputs/preds_single")
    p.add_argument("--fig",     default=None,
                   help="optional PNG path for overlays")
    p.add_argument("--thresh",  type=float, default=0.5,
                   help="probability threshold")
    p.add_argument("--shift",   type=int, default=0,
                   help="reorder shift: -1 for random, 0-31 for circular shift")
    args = p.parse_args()

    main(args.subject, args.config, args.weights,
         args.out_dir, args.fig, args.thresh, args.shift)
