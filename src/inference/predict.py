"""
Run inference on unseen test subjects and save padded full-res masks.

Usage:
    python -m src.inference.predict --config config.yaml --weights outputs/checkpoints/best.pt
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt


def pad_to_full(mask64: np.ndarray, full_size: int = 240) -> np.ndarray:
    crop = cfg["data"]["crop_size"]
    pad = (full_size - crop) // 2
    full = np.zeros((full_size, full_size), dtype=mask64.dtype)
    full[pad : pad + crop, pad : pad + crop] = mask64
    return full


# --------------------------------------------------------------------- #
def main(config, weights: str | Path, save_dir: str | Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CSFVolumeDataset(
        root_dir=Path(config["data"]["root"]) / config["data"]["test_dir"],
        split="test",
        crop_size=config["data"]["crop_size"],
    )
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds))

    model = UNet2D(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    ckpt = load_ckpt(weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            subj_id = batch["id"][0]
            img = batch["image"].to(device)

            logits = model(img)
            pred = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (64,64)
            pred_full = pad_to_full(pred)

            np.save(save_dir / f"{subj_id}_pred.npy", pred_full.astype(np.float32))
            print(f"{subj_id}  →  saved.")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--save_dir", type=str, default="outputs/preds")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    main(cfg, weights=args.weights, save_dir=args.save_dir)
