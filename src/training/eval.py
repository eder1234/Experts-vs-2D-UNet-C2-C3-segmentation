# src/training/eval.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

# direct submodule imports (avoid src.__init__ re-exports)
from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.metrics import compute_all

def _pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad : pad + crop, pad : pad + crop] = mask_crop
    return out

def _in_channels_for_mode(mode: str) -> int:
    mode = (mode or "full").lower()
    if mode == "full": return 64
    if mode in ("pca", "dft_power", "tvt", "std"): return 1
    if mode == "dft_k123": return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a single checkpoint on val or test split."
    )
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["val", "test"])
    ap.add_argument("--best_model", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for binarization")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device","cuda")=="cuda" else "cpu")

    input_mode = cfg["data"].get("input_mode", "full")
    crop = int(cfg["data"]["crop_size"])
    in_ch = _in_channels_for_mode(input_mode)

    # dataset
    if args.split == "test":
        root_dir = Path(cfg["data"]["root"]) / cfg["data"]["test_dir"]
        ds = CSFVolumeDataset(root_dir, split="test", crop_size=crop, input_mode=input_mode,
                              val_split=float(cfg["data"]["val_split"]))
    else:
        root_dir = Path(cfg["data"]["root"]) / cfg["data"]["train_dir"]
        ds = CSFVolumeDataset(root_dir, split="val", crop_size=crop, input_mode=input_mode,
                              val_split=float(cfg["data"]["val_split"]))
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds), num_workers=2)

    # model + weights
    model = UNet2D(in_channels=in_ch,
                   out_channels=int(cfg["model"]["out_channels"]),
                   base_channels=int(cfg["model"]["base_channels"])).to(device)
    ckpt = load_ckpt(args.best_model, map_location=device)
    state_dict = ckpt.get("state_dict") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint missing 'state_dict' or 'model' keys. Keys: {list(ckpt.keys())}")
    model.load_state_dict(state_dict)
    model.eval()

    # eval loop
    metrics = []
    with torch.no_grad():
        for batch in loader:
            gt_crop = batch["mask"].numpy()[0, 0]
            gt_full = _pad_to_full(gt_crop, crop)

            x = batch["image"].to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()[0, 0]

            # threshold HERE
            pred_bin = (probs >= args.thresh).astype(np.uint8)
            pred_full = _pad_to_full(pred_bin, crop)

            m = compute_all(pred_full, gt_full)
            metrics.append(m)

    # summarize
    if not metrics:
        print("No samples found.")
        return
    keys = list(metrics[0].keys())
    means = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
    print(f"Split: {args.split} — subjects: {len(metrics)} | mode: {input_mode}")
    for k in ["dice","iou","sensitivity","specificity"]:
        if k in means:
            print(f"{k:<12}: {means[k]:.4f}")

if __name__ == "__main__":
    main()
