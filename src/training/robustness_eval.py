# src/training/robustness_eval.py

from __future__ import annotations
import argparse, csv, math, random
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, SequentialSampler

from src.datasets.csf_volume_dataset import CSFVolumeDataset
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.metrics import compute_all
from src.utils.temporal import reorder_temporal_images
from src.utils.temporal_features import dft_bandpower_excl_dc, temporal_tv, temporal_std, dft_magnitudes_bins

def _pad_to_full(x, crop, full=240):
    pad = (full - crop)//2
    out = np.zeros((full,full), dtype=x.dtype)
    out[pad:pad+crop, pad:pad+crop] = x
    return out

def _center_crop(arr, size):
    h,w = arr.shape[-2:]
    t,l = (h-size)//2, (w-size)//2
    return arr[..., t:t+size, l:l+size]

def _in_ch(mode):
    if mode=="full": return 64
    if mode in ("pca","dft_power","tvt","std"): return 1
    if mode=="dft_k123": return 3
    raise ValueError(mode)

def _first_pc(vol):
    c,h,w = vol.shape
    x = vol.reshape(c,-1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    u,s,vt = np.linalg.svd(x, full_matrices=False)
    pc = (s[0]*vt[0]).reshape(h,w)
    pc = (pc - pc.mean())/(pc.std()+1e-8)
    return pc

def _build_input(mode, phase, mag):
    if mode=="full":      return np.concatenate([phase,mag],0)
    if mode=="pca":       return _first_pc(np.concatenate([phase,mag],0))[None,...]
    if mode=="dft_power": return dft_bandpower_excl_dc(phase)[None,...]
    if mode=="tvt":       return temporal_tv(phase)[None,...]
    if mode=="std":       return temporal_std(phase)[None,...]
    if mode=="dft_k123":  return dft_magnitudes_bins(phase, bins=(1,2,3))
    raise ValueError(mode)

def _apply_spatial(img_t, mask_t, angle_deg=0.0, translate_frac=0.0):
    """Apply identical affine to image (C,H,W) and mask (1,H,W)."""
    if angle_deg==0 and translate_frac==0:
        return img_t, mask_t
    H,W = img_t.shape[-2:]
    max_trans = translate_frac*W
    tx = float(max_trans)  # +x right
    ty = float(max_trans)  # +y down
    # Use deterministic mid-extent translation (worst-case) so it's controlled,
    # not random every time; caller can sweep values.
    img = VF.affine(img_t, angle=angle_deg, translate=[tx,ty], scale=1.0, shear=[0.0,0.0])
    # For masks, use nearest-neighbor to keep binary
    mask = VF.affine(mask_t, angle=angle_deg, translate=[tx,ty], scale=1.0, shear=[0.0,0.0], interpolation=VF.InterpolationMode.NEAREST)
    return img, mask

def robustness_for_run(cfg, run_dir: Path, split: str, thresh: float,
                       sigmas, rot_degs, trans_fracs, temporal_modes):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device","cuda")=="cuda" else "cpu")

    import re
    m = re.match(r"^unet2d_([^_]+)_c(\d+)_b(\d+)_", run_dir.name)
    if not m: raise ValueError(f"Bad run dir {run_dir.name}")
    mode, crop, base = m.group(1), int(m.group(2)), int(m.group(3))
    in_ch = _in_ch(mode)

    root = Path(cfg["data"]["root"])
    data_root = root / (cfg["data"]["test_dir"] if split=="test" else cfg["data"]["train_dir"])
    ds = CSFVolumeDataset(data_root, split=("val" if split=="val" else "test"),
                          crop_size=crop, val_split=float(cfg["data"]["val_split"]), input_mode=mode)
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds), num_workers=2)

    ckpt = run_dir / "checkpoints" / "best_model.pt"
    sd = load_ckpt(str(ckpt), map_location=device)
    state = sd.get("state_dict") or sd.get("model")
    model = UNet2D(in_channels=in_ch, out_channels=int(cfg["model"]["out_channels"]),
                   base_channels=base).to(device)
    model.load_state_dict(state); model.eval()

    rows = []

    for batch in loader:
        sid = batch["id"][0]
        gt_crop = batch["mask"].numpy()[0,0]
        subj_dir = data_root / sid
        phase = np.load(subj_dir / "phase.npy")
        mag   = np.load(subj_dir / "mag.npy")

        base_vol = _build_input(mode, phase, mag)
        base_vol = _center_crop(base_vol, crop)

        # ---------- 1) Additive Gaussian noise ----------
        for s in sigmas:
            vol = base_vol.copy()
            vol = (vol - vol.mean())/(vol.std()+1e-8)
            vol_noisy = vol + s * np.random.randn(*vol.shape).astype(vol.dtype)
            with torch.no_grad():
                inp = torch.from_numpy(vol_noisy).unsqueeze(0).float().to(device)
                probs = torch.sigmoid(model(inp)).cpu().numpy()[0,0]
            pred = (probs >= thresh).astype(np.uint8)
            m = compute_all(_pad_to_full(pred, crop), _pad_to_full(gt_crop, crop))
            rows.append({"subject": sid, "run": run_dir.name, "perturb":"gauss", "level": s, **m})

        # ---------- 2) Spatial jitter (rotations / translations) ----------
        import torch as T
        vol_t  = T.from_numpy((base_vol - base_vol.mean())/(base_vol.std()+1e-8)).float()
        mask_t = T.from_numpy(gt_crop[None,...]).float()
        for ang in rot_degs:
            vt, mt = _apply_spatial(vol_t, mask_t, angle_deg=ang, translate_frac=0.0)
            with torch.no_grad():
                probs = torch.sigmoid(model(vt.unsqueeze(0).to(device))).cpu().numpy()[0,0]
            pred = (probs >= thresh).astype(np.uint8)
            m = compute_all(_pad_to_full(pred, crop), _pad_to_full(mt.numpy()[0], crop))
            rows.append({"subject": sid, "run": run_dir.name, "perturb":"rot", "level": ang, **m})
        for tr in trans_fracs:
            vt, mt = _apply_spatial(vol_t, mask_t, angle_deg=0.0, translate_frac=tr)
            with torch.no_grad():
                probs = torch.sigmoid(model(vt.unsqueeze(0).to(device))).cpu().numpy()[0,0]
            pred = (probs >= thresh).astype(np.uint8)
            m = compute_all(_pad_to_full(pred, crop), _pad_to_full(mt.numpy()[0], crop))
            rows.append({"subject": sid, "run": run_dir.name, "perturb":"trans", "level": tr, **m})

        # ---------- 3) Temporal corruption ----------
        for tm in temporal_modes:
            if tm == "shuffle":
                ph_s, mag_s, _ = reorder_temporal_images(phase, mag, shift=-1)  # random perm
            elif tm == "shift32_max":  # best/worst-of shifts isn’t defined; evaluate deterministic big shift
                ph_s, mag_s, _ = reorder_temporal_images(phase, mag, shift=16)
            else:
                raise ValueError(tm)
            vol = _build_input(mode, ph_s, mag_s)
            vol = _center_crop(vol, crop)
            vol = (vol - vol.mean())/(vol.std()+1e-8)
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.from_numpy(vol).unsqueeze(0).float().to(device))).cpu().numpy()[0,0]
            pred = (probs >= thresh).astype(np.uint8)
            m = compute_all(_pad_to_full(pred, crop), _pad_to_full(gt_crop, crop))
            rows.append({"subject": sid, "run": run_dir.name, "perturb":"temporal", "level": tm, **m})

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out_csv", default=None)
    # Defaults chosen to *extend* beyond your training augment ranges
    ap.add_argument("--sigmas", type=str, default="0,0.01,0.02,0.05,0.10")
    ap.add_argument("--rot_degs", type=str, default="0,15,30,45")
    ap.add_argument("--trans_fracs", type=str, default="0,0.05,0.10,0.15")  # fraction of crop
    ap.add_argument("--temporal_modes", type=str, default="shuffle,shift32_max")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sigmas = [float(x) for x in args.sigmas.split(",") if x!=""]
    rot_degs = [float(x) for x in args.rot_degs.split(",") if x!=""]
    trans_fracs = [float(x) for x in args.trans_fracs.split(",") if x!=""]
    temporal_modes = [x.strip() for x in args.temporal_modes.split(",") if x.strip()]

    rows = robustness_for_run(cfg, Path(args.run_dir), args.split, args.thresh,
                              sigmas, rot_degs, trans_fracs, temporal_modes)

    out_csv = Path(args.out_csv or f"outputs/robustness_{Path(args.run_dir).name}_{args.split}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    if rows:
        keys = ["subject","run","perturb","level","dice","iou","sensitivity","specificity"]
        with open(out_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k,"") for k in keys})
    print(f"[OK] Saved {out_csv}")

if __name__ == "__main__":
    main()
