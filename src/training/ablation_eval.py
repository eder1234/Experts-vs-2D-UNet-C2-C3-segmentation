# src/training/ablation_eval.py

from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.metrics import compute_all
from src.utils.temporal_features import dft_bandpower_excl_dc, temporal_tv, temporal_std, dft_magnitudes_bins

def _pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad: pad + crop, pad: pad + crop] = mask_crop
    return out

def _in_ch(mode: str) -> int:
    if mode == "full": return 64
    if mode in ("pca","dft_power","tvt","std"): return 1
    if mode == "dft_k123": return 3
    raise ValueError(mode)

def _first_pc(vol):
    c,h,w = vol.shape
    x = vol.reshape(c,-1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    u,s,vt = np.linalg.svd(x, full_matrices=False)
    pc = (s[0]*vt[0]).reshape(h,w)
    pc = (pc - pc.mean())/(pc.std()+1e-8)
    return pc

def _build_input(mode: str, phase: np.ndarray, mag: np.ndarray) -> np.ndarray:
    if mode == "full":       return np.concatenate([phase, mag], axis=0)
    if mode == "pca":        return _first_pc(np.concatenate([phase,mag],0))[None,...]
    if mode == "dft_power":  return dft_bandpower_excl_dc(phase)[None,...]
    if mode == "tvt":        return temporal_tv(phase)[None,...]
    if mode == "std":        return temporal_std(phase)[None,...]
    if mode == "dft_k123":   return dft_magnitudes_bins(phase, bins=(1,2,3))
    raise ValueError(mode)

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h,w = arr.shape[-2:]
    top,left = (h-size)//2, (w-size)//2
    return arr[..., top:top+size, left:left+size]

def eval_one_run(cfg, run_dir: Path, split: str, thresh: float, variant: str | None):
    """variant: None, 'phase_only', or 'mag_only' for the FULL-mode ablation."""
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device","cuda")=="cuda" else "cpu")

    # parse mode/crop/base from folder name
    name = run_dir.name
    # expected: unet2d_<mode>_c<crop>_b<base>_...
    import re
    m = re.match(r"^unet2d_([^_]+)_c(\d+)_b(\d+)_", name)
    if not m:
        raise ValueError(f"Unrecognized run folder: {name}")
    mode, crop, base = m.group(1), int(m.group(2)), int(m.group(3))
    in_ch = _in_ch(mode)

    # dataset root
    root = Path(cfg["data"]["root"])
    data_root = root / (cfg["data"]["test_dir"] if split=="test" else cfg["data"]["train_dir"])
    ds = CSFVolumeDataset(data_root, split=("val" if split=="val" else "test"),
                          crop_size=crop, val_split=float(cfg["data"]["val_split"]),
                          input_mode=mode)
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds), num_workers=2)

    # model
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    sd = load_ckpt(str(ckpt_path), map_location=device)
    state = sd.get("state_dict") or sd.get("model")
    model = UNet2D(in_channels=in_ch,
                   out_channels=int(cfg["model"]["out_channels"]),
                   base_channels=base).to(device)
    model.load_state_dict(state); model.eval()

    mets = []
    with torch.no_grad():
        for batch in loader:
            sid = batch["id"][0]
            gt_crop = batch["mask"].numpy()[0,0]
            gt_full = _pad_to_full(gt_crop, crop)

            subj_dir = data_root / sid
            phase = np.load(subj_dir / "phase.npy")
            mag   = np.load(subj_dir / "mag.npy")

            vol = _build_input(mode, phase, mag)
            # phase/mag-only ablation for FULL mode
            if mode == "full" and variant in ("phase_only","mag_only"):
                if variant == "phase_only":
                    vol[32:] = 0.0       # zero magnitude stack
                else:
                    vol[:32] = 0.0       # zero phase stack

            vol = _center_crop(vol, crop)
            vol = (vol - vol.mean())/(vol.std()+1e-8)

            inp = torch.from_numpy(vol).unsqueeze(0).float().to(device)
            probs = torch.sigmoid(model(inp)).cpu().numpy()[0,0]
            pred = (probs >= thresh).astype(np.uint8)
            pred_full = _pad_to_full(pred, crop)

            mets.append(compute_all(pred_full, gt_full))

    mean = {k: float(np.mean([m[k] for m in mets])) for k in mets[0].keys()}
    mean["subjects"] = len(mets)
    mean["mode"] = mode
    mean["loss"] = name.split("_")[-1]
    if variant: mean["variant"] = variant
    return mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--runs", nargs="+", required=True, help="Run folders under outputs/ to evaluate")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--phase_mag_ablation", action="store_true", help="For FULL-mode runs, also compute phase-only and mag-only.")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results = []
    for rd in args.runs:
        run_dir = Path(rd)
        res = eval_one_run(cfg, run_dir, args.split, args.thresh, variant=None)
        results.append(res)
        # optional phase/mag-only
        if args.phase_mag_ablation:
            if res["mode"] == "full":
                results.append(eval_one_run(cfg, run_dir, args.split, args.thresh, variant="phase_only"))
                results.append(eval_one_run(cfg, run_dir, args.split, args.thresh, variant="mag_only"))

    out = Path(args.out_csv or f"outputs/ablation_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r.keys()}, key=lambda k: ["mode","loss","variant","dice","iou","sensitivity","specificity","subjects"].index(k) if k in ["mode","loss","variant","dice","iou","sensitivity","specificity","subjects"] else 100+k.count(k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[OK] Saved {out}")

if __name__ == "__main__":
    main()
