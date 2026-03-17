# src/training/combinatory_eval.py
from __future__ import annotations

import argparse
import copy
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt

# Direct submodule imports (avoid src.__init__ re-exports)
from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.metrics import compute_all
from src.utils.temporal import reorder_temporal_images
from src.utils.temporal_features import (
    temporal_std,
    temporal_tv,
    dft_bandpower_excl_dc,
    dft_magnitudes_bins,
)

# ------------------------- discovery helpers -------------------------

RUN_RE = re.compile(
    r"^unet2d_(?P<mode>.+?)_c(?P<crop>\d+)_b(?P<base>\d+)_(?P<loss>dice|tversky|focal_dice|flow_dice)$"
)
SUPPORTED_MODES = {"tvt", "std", "dft_k123", "pca", "full", "dft_power"}


def parse_run_name(name: str) -> Tuple[str, int, int, str] | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    mode = m.group("mode")
    if mode not in SUPPORTED_MODES:
        return None
    return mode, int(m.group("crop")), int(m.group("base")), m.group("loss")


# ------------------------- data helpers -------------------------

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top : top + size, left : left + size]


def _first_pc(vol: np.ndarray) -> np.ndarray:
    C, H, W = vol.shape
    X = vol.reshape(C, H * W).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc_scores = (S[0] * Vt[0]).reshape(H, W)
    pc_scores = (pc_scores - pc_scores.mean()) / (pc_scores.std() + 1e-8)
    return pc_scores


def pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad : pad + crop, pad : pad + crop] = mask_crop
    return out


def _in_channels_for_mode(mode: str) -> int:
    if mode == "full":
        return 64
    if mode in ("pca", "dft_power", "tvt", "std"):
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
        img = dft_magnitudes_bins(phase, bins=(1, 2, 3))
        return _center_crop(img, crop)
    raise ValueError(f"Unknown input_mode '{input_mode}'")


# ------------------------- evaluation core -------------------------

def evaluate_one_run(
    base_cfg: Dict,
    run_dir: Path,
    mode: str,
    loss_name: str,
    crop: int,
    split: str,
    cyclic_search: bool = False,
    thresh: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a single trained run on the requested split.
    Returns mean metrics dict with keys: dice, iou, sensitivity, specificity, subjects, mode, loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset for this split (GT & IDs); we rebuild inputs to match mode.
    cfg = copy.deepcopy(base_cfg)
    root = Path(cfg["data"]["root"])
    data_root = root / (cfg["data"]["test_dir"] if split == "test" else cfg["data"]["train_dir"])
    ds = CSFVolumeDataset(
        root_dir=data_root,
        split=split,
        crop_size=crop,
        val_split=cfg["data"]["val_split"],
        input_mode=mode,  # keeps loader consistent
    )
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds), num_workers=2)

    # Model + checkpoint
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = load_ckpt(ckpt_path, map_location=device)

    # accept either {"state_dict": ...} or {"model": ...}
    state_dict = ckpt.get("state_dict") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no 'state_dict' or 'model' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    in_ch = _in_channels_for_mode(mode)
    model = UNet2D(
        in_channels=in_ch,
        out_channels=base_cfg["model"]["out_channels"],
        base_channels=base_cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    metrics_list: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            subj_id = batch["id"][0]
            gt_crop = batch["mask"].numpy()[0]
            gt_full = pad_to_full(gt_crop, crop)

            subj_dir = data_root / subj_id
            phase = np.load(subj_dir / "phase.npy")  # (32,240,240)
            mag = np.load(subj_dir / "mag.npy")      # (32,240,240)

            shifts = range(32) if cyclic_search else (0,)
            best_m = None

            for shift in shifts:
                ph_s, mag_s, _ = reorder_temporal_images(phase, mag, shift=shift)
                vol = _build_input_for_mode(mode, ph_s, mag_s, crop)
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)

                inp = torch.from_numpy(vol).unsqueeze(0).float().to(device)
                probs = torch.sigmoid(model(inp)).cpu().numpy()[0, 0]
                pred_bin = (probs >= thresh).astype(np.uint8)
                pred_full = pad_to_full(pred_bin, crop)
                m = compute_all(pred_full, gt_full)
                if best_m is None or m["dice"] > best_m["dice"]:
                    best_m = m

            metrics_list.append(best_m)

    keys = metrics_list[0].keys()
    mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    mean["subjects"] = len(metrics_list)
    mean["mode"] = mode
    mean["loss"] = loss_name
    return mean


# ------------------------- reporting -------------------------

def plot_dice_bar(results: List[Dict[str, float]], title: str, out_path: Path) -> None:
    labels = [f"{r['mode']}-{r['loss']}" for r in results]
    dice = [r["dice"] for r in results]

    plt.figure(figsize=(max(10, 0.45 * len(labels)), 5))
    x = np.arange(len(labels))
    plt.bar(x, dice)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("DICE")
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_csv(results: List[Dict[str, float]], out_path: Path) -> None:
    if not results:
        return
    keys = ["mode", "loss"] + [
        k for k in results[0].keys() if k not in ("mode", "loss")
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate all trained (input_mode × loss) runs found in outputs/."
    )
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--outputs_root", type=str, default="outputs")
    ap.add_argument(
        "--splits", type=str, default="val,test",
        help="Comma-separated: any of {val,test}"
    )
    ap.add_argument(
        "--cyclic_search", action="store_true",
        help="Evaluate using best-of-32 temporal cyclic shifts (slower)."
    )
    ap.add_argument("--thresh", type=float, default=0.5,
                help="Probability threshold used to binarize predictions.")

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(args.outputs_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # discover runs
    runs = []
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_run_name(p.name)
        if parsed is None:
            print(f"[SKIP name] {p.name}")
            continue
        mode, crop, base, loss = parsed
        ck = p / "checkpoints" / "best_model.pt"
        if not ck.exists():
            print(f"[SKIP ckpt] {p.name} — missing {ck}")
            continue
        runs.append((p, mode, crop, base, loss))

    if not runs:
        print("No evaluated runs found (expected folders like unet2d_<mode>_c80_b32_<loss>).")
        return

    print(f"Found {len(runs)} runs to evaluate.")

    # per split evaluation
    for split in splits:
        split_results: List[Dict[str, float]] = []
        for run_dir, mode, crop, base, loss in sorted(runs, key=lambda x: (x[1], x[4])):
            # ensure model params matched to the trained run
            cfg["model"]["base_channels"] = base
            cfg["data"]["crop_size"] = crop

            res = evaluate_one_run(
                cfg, run_dir, mode, loss, crop, split,
                cyclic_search=args.cyclic_search, thresh=args.thresh
            )
            split_results.append(res)
            print(f"[{split}] {run_dir.name}: DICE={res['dice']:.4f}  IoU={res.get('iou', float('nan')):.4f}")

        # save CSV + figure (DICE only to keep it readable)
        save_csv(split_results, out_root / f"combinatory_eval_{split}.csv")
        plot_dice_bar(
            split_results,
            f"Combinatory Eval — {split.upper()} (DICE)",
            out_root / "figures" / f"combinatory_eval_{split}.png",
        )

    print("Done. CSVs in outputs/, figures in outputs/figures/")

if __name__ == "__main__":
    main()
