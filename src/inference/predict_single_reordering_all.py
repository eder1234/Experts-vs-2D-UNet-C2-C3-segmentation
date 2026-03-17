# src/inference/predict_single_reordering_all.py
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.temporal import reorder_temporal_images
from src.utils.temporal_features import (
    temporal_std, temporal_tv, dft_bandpower_excl_dc, dft_magnitudes_bins
)

def _first_pc(vol: np.ndarray) -> np.ndarray:
    c, h, w = vol.shape
    x = vol.reshape(c, -1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, 0:1].T @ x).reshape(h, w)

def pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad : pad + crop, pad : pad + crop] = mask_crop
    return out

def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = .4) -> np.ndarray:
    img_norm = (img - img.min()) / (img.ptp() + 1e-8)
    rgb = np.stack([img_norm]*3, -1)
    rgb[..., 0][gt > .5]  = (1-alpha)*rgb[..., 0][gt > .5]  + alpha  # GT → red
    rgb[..., 2][pred > .5] = (1-alpha)*rgb[..., 2][pred > .5] + alpha  # PR → blue
    return np.clip(rgb, 0, 1)

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return 2*inter / (pred.sum()+gt.sum()+1e-8)

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top : top + size, left : left + size]

def _in_channels_for_mode(mode: str) -> int:
    if mode == "full":
        return 64
    if mode == "pca":
        return 1
    if mode in ("dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

def _build_input_for_mode(input_mode: str, phase: np.ndarray, mag: np.ndarray, crop: int) -> np.ndarray:
    if input_mode == "full":
        vol = np.concatenate([phase, mag], axis=0)
        return _center_crop(vol, crop)
    if input_mode == "pca":
        vol = np.concatenate([phase, mag], axis=0)
        img = _first_pc(vol)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "dft_power":
        img = dft_bandpower_excl_dc(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "tvt":
        img = temporal_tv(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "std":
        img = temporal_std(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "dft_k123":
        img = dft_magnitudes_bins(phase, bins=(1,2,3))
        return _center_crop(img, crop)
    raise ValueError(f"Unknown input_mode '{input_mode}'")

def main(args) -> None:
    cfg = load_yaml(args.config)
    crop = cfg["data"]["crop_size"]
    input_mode = cfg["data"].get("input_mode", "full")

    # ------------------------------ data -------------------------------- #
    ds_root = Path(cfg["data"]["root"]) / cfg["data"]["test_dir"]

    # ground-truth crop/full via dataset helper (consistent transforms)
    ds = CSFVolumeDataset(ds_root, split="test", crop_size=crop, input_mode=input_mode)
    matches = [i for i, p in enumerate(ds.subjects) if p.name == args.subject]
    if not matches:
        raise FileNotFoundError(f"No subject '{args.subject}' in {ds_root}")
    idx = matches[0]
    gt_crop = ds[idx]["mask"].squeeze().numpy()
    gt_full = pad_to_full(gt_crop, crop)

    subj_dir = ds.subjects[idx]
    phase = np.load(subj_dir / "phase.npy")        # (32,240,240)
    mag   = np.load(subj_dir / "mag.npy")          # (32,240,240)

    # ------------------------------ model ------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = _in_channels_for_mode(input_mode)
    model = UNet2D(in_channels=in_ch,
                   out_channels=cfg["model"]["out_channels"],
                   base_channels=cfg["model"]["base_channels"]).to(device)
    model.load_state_dict(load_ckpt(args.weights, map_location=device)["state_dict"])
    model.eval()

    # --------------------------- visual loop ---------------------------- #
    n_variants = 33
    fig, axs = plt.subplots(n_variants, 3, figsize=(12, 4*n_variants))
    shifts = [-1] + list(range(32))
    dice_scores = []

    for i, shift in enumerate(shifts):
        ph_s, mag_s, _ = reorder_temporal_images(phase, mag, shift)
        vol = _build_input_for_mode(input_mode, ph_s, mag_s, crop)

        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        inp = torch.from_numpy(vol).unsqueeze(0).float().to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(inp)).cpu().numpy()[0, 0]

        pred_crop = (probs >= args.thresh).astype(np.uint8)
        dice_scores.append(compute_dice(pred_crop, gt_crop))

        pred_full = pad_to_full(pred_crop, crop)

        axs[i, 0].imshow(pred_full, cmap="gray");         axs[i, 0].axis("off")
        axs[i, 0].set_title(f"Mask | shift {shift}")

        axs[i, 1].imshow(overlay(phase[0], gt_full, pred_full)); axs[i, 1].axis("off")
        axs[i, 1].set_title("Overlay – phase₀")

        axs[i, 2].imshow(overlay(mag[0], gt_full, pred_full));   axs[i, 2].axis("off")
        axs[i, 2].set_title("Overlay – mag₀")

    fig.tight_layout()
    if args.fig:
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=200)
        print(f"Figure saved → {Path(args.fig).resolve()}")

    # --------------------------- radar plot ---------------------------- #
    import matplotlib.pyplot as plt2
    from matplotlib.patches import Circle
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    dice_shifted, dice_rand = dice_scores[1:], dice_scores[0]

    fig2 = plt2.figure(figsize=(6,6)); ax2 = fig2.add_subplot(111, polar=True)
    ax2.plot(np.append(angles, angles[0]),
             np.append(dice_shifted, dice_shifted[0]), lw=2)
    ax2.fill(np.append(angles, angles[0]),
             np.append(dice_shifted, dice_shifted[0]), alpha=.3, label="Shifts")
    ax2.add_patch(Circle((0,0), radius=dice_rand,
                         transform=ax2.transData._b, color='r', alpha=.25,
                         label=f"Random ({dice_rand:.3f})"))
    ax2.set_ylim(0,1); ax2.set_title("Dice across shifts"); ax2.legend()
    fig2.tight_layout()
    fig2.savefig("outputs/figures/dice_radar.png", dpi=200)
    print("Radar plot saved → outputs/figures/dice_radar.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--weights", default="outputs/checkpoints/best.pt")
    p.add_argument("--fig",     default=None, help="optional path for PNG")
    p.add_argument("--thresh",  type=float, default=0.5)
    args = p.parse_args()

    main(args)
