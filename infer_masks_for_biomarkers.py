# inference_mask_for_biomarkers.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt

# ---- tiny local utils (no extra deps) --------------------------------
def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top + size, left:left + size]

def pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad: pad + crop, pad: pad + crop] = mask_crop
    return out

def build_full_input(phase: np.ndarray, mag: np.ndarray, crop: int) -> np.ndarray:
    # (64, H, W) = [32 phase + 32 magnitude], center-cropped
    vol = np.concatenate([phase, mag], axis=0)
    return center_crop(vol, crop)

def main():
    ap = argparse.ArgumentParser("Run inference on test set and save predicted masks")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", required=True,
                    help="Path to best_model.pt (e.g., outputs/unet2d_full_c80_b32_dice/checkpoints/best_model.pt)")
    ap.add_argument("--out_dir", default="outputs/preds_biomarkers",
                    help="Where to save <subject>_pred.npy")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="Threshold for binarizing probabilities (FlowDice often uses 0.15)")
    ap.add_argument("--phase_scale", choices=["auto", "none"], default="auto",
                    help="If 'auto', convert [0,1] phase to [-1,1] by x*2-1")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    crop = int(cfg["data"]["crop_size"])
    data_root = Path(cfg["data"]["root"]) / cfg["data"]["test_dir"]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=64,
                   out_channels=cfg["model"]["out_channels"],
                   base_channels=cfg["model"]["base_channels"]).to(device)

    ckpt = load_ckpt(args.weights, map_location=device)
    state_dict = ckpt.get("state_dict") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint has no 'state_dict' nor 'model' keys: {list(ckpt.keys())}")
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in data_root.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} test subjects in {data_root}")

    with torch.no_grad():
        for subj_dir in subjects:
            sid = subj_dir.name
            phase = np.load(subj_dir / "phase.npy")  # (32,240,240)
            mag   = np.load(subj_dir / "mag.npy")    # (32,240,240)

            if args.phase_scale == "auto":
                # common case: your stored phase is [0,1] → convert to [-1,1]
                phase = phase * 2.0 - 1.0

            inp = build_full_input(phase, mag, crop)   # (64, crop, crop)
            # z-score normalise per sample
            inp = (inp - inp.mean()) / (inp.std() + 1e-8)

            x = torch.from_numpy(inp).unsqueeze(0).float().to(device)  # (1,64,crop,crop)
            probs = torch.sigmoid(model(x)).cpu().numpy()[0, 0]        # (crop,crop)

            pred_crop = (probs >= args.thresh).astype(np.uint8)
            pred_full = pad_to_full(pred_crop, crop)

            np.save(out_dir / f"{sid}_pred.npy", pred_full)
            print(f"[OK] {sid}: saved → {out_dir / f'{sid}_pred.npy'}")

    print("Done.")

if __name__ == "__main__":
    main()
