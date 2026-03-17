#!/usr/bin/env python3
# compute_biomarkers_from_dicom.py
"""
Compute CSF biomarkers from a DICOMDIR with either:
  - manual mask from ROI .txt, or
  - automatic mask predicted by a selected segmentation model.

This version:
  • Matches CSFVolumeDataset preprocessing (per-sample z-score; feature modes).
  • Reads RAW DICOM arrays for model input; keeps [0,1] stacks for physics.
  • Handles PC-MRI series with 64 PHASE + 32 MAG by aligning PHASE→MAG times
    (TriggerTime/InstanceNumber), selecting exactly 32 phase frames.
  • Pads crop back to TRUE (H,W). Prints concise diagnostics.

Outputs:
  • CSV row: sample, roi_area_mm2, flow_amplitude_mm3_per_s, stroke_volume_mm3
  • Optional overlays and flow-curve image.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import yaml
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Project modules --------------------------------------------------------
from preprocess_dicom import (
    make_mask_from_txt,
    inventory_dicomdir,
    load_series_from_dicomdir,
    embed_mask_in_image,
    _norm01 as _norm01_img,   # ONLY for flow physics path
)
from csf_flow import compute_flow_and_stroke_volume
from src.models.unet2d import UNet2D
try:
    from src.utils.misc import load_ckpt
except Exception:
    load_ckpt = None

from src.utils.temporal_features import (
    temporal_std,
    temporal_tv,
    dft_bandpower_excl_dc,
    dft_magnitudes_bins,
)

# --------------------------- utilities --------------------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _phase01_to_unit(phase_stack_01: np.ndarray) -> np.ndarray:
    return phase_stack_01 * 2.0 - 1.0

def _extract_trigger_time(ds) -> Optional[float]:
    tt = getattr(ds, "TriggerTime", None)
    if tt is not None:
        try:
            return float(tt)
        except Exception:
            pass
    elem = ds.get((0x0018, 0x1060), None)  # Trigger Time
    if elem is not None:
        try:
            return float(str(elem.value).replace(",", "\\").split("\\")[0])
        except Exception:
            return None
    return None

def _extract_instance_number(ds) -> Optional[int]:
    inst = getattr(ds, "InstanceNumber", None)
    try:
        return int(inst) if inst is not None else None
    except Exception:
        return None

def _sort_temporally(datasets: List) -> List:
    def key(ds):
        tt = _extract_trigger_time(ds)
        inst = _extract_instance_number(ds)
        return (float(tt) if tt is not None else float("inf"),
                int(inst) if inst is not None else 1_000_000)
    return sorted(datasets, key=key)

def _align_phase_to_mag(phase_ds: List, mag_ds: List) -> List:
    """
    Align 64 PHASE frames to 32 MAG frames by time. For each MAG time, select
    the PHASE frame with the same/nearest TriggerTime; if two PHASE frames map
    to the same MAG time (common), pick the one with larger spatial std.
    Fallback (no times): try even indices, else odd; prefer the one with higher
    overall std across the selected 32.
    Returns the list of 32 PHASE datasets, ordered to match MAG time order.
    """
    # Build time tables
    mag_times = [(_extract_trigger_time(ds), i) for i, ds in enumerate(mag_ds)]
    ph_times  = [(_extract_trigger_time(ds), i) for i, ds in enumerate(phase_ds)]

    have_times = all(t is not None for t, _ in mag_times) and all(t is not None for t, _ in ph_times)

    if have_times:
        # Group phase indices by rounded ms time (to merge tiny jitter)
        def rkey(t): return int(round(t))
        from collections import defaultdict
        phase_by_time = defaultdict(list)
        for t, i in ph_times:
            phase_by_time[rkey(t)].append(i)

        selected_idx: List[int] = []
        for t_mag, _ in mag_times:
            rk = rkey(t_mag)
            candidates = phase_by_time.get(rk, [])
            if not candidates:
                # nearest neighbor in time
                diffs = [(abs(t_mag - t_ph), i) for (t_ph, i) in ph_times]
                diffs.sort()
                cand = diffs[0][1]
                selected_idx.append(cand)
            elif len(candidates) == 1:
                selected_idx.append(candidates[0])
            else:
                # choose candidate with larger spatial std
                stds = []
                for ci in candidates:
                    arr = phase_ds[ci].pixel_array.astype(np.float32)
                    stds.append((float(arr.std()), ci))
                stds.sort(reverse=True)
                selected_idx.append(stds[0][1])

        # ensure uniqueness and length 32; if duplicates, de-dup preserving order
        seen = set()
        uniq = []
        for i in selected_idx:
            if i not in seen:
                uniq.append(i); seen.add(i)
        if len(uniq) < 32:
            # fill with nearest unused indices by time order
            remaining = [i for _, i in ph_times if i not in seen]
            uniq.extend(remaining[: 32 - len(uniq)])
        aligned_phase = [phase_ds[i] for i in uniq[:32]]
        print(f"[align] used times: TriggerTime-based (unique {len(set(uniq[:32]))}/32).")
        return aligned_phase

    # Fallback: no times → try even vs. odd slices
    even_idx = list(range(0, len(phase_ds), 2))[:32]
    odd_idx  = list(range(1, len(phase_ds), 2))[:32]

    def mean_std(idxs):
        vals = []
        for i in idxs:
            arr = phase_ds[i].pixel_array.astype(np.float32)
            vals.append(float(arr.std()))
        return float(np.mean(vals)) if vals else 0.0

    even_std = mean_std(even_idx)
    odd_std  = mean_std(odd_idx)
    choice = "even" if even_std >= odd_std else "odd"
    idxs = even_idx if choice == "even" else odd_idx
    print(f"[align] no TriggerTime: chose {choice} indices (mean std {max(even_std, odd_std):.4f}).")
    return [phase_ds[i] for i in idxs]

def _series_to_full_stacks_RAW_and_01(split) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return RAW stacks (phase_raw, mag_raw) for model input and [0,1] stacks
    (phase_01, mag_01) for biomarker physics. Enforce 32+32 with alignment if needed.
    """
    # deterministic sorting
    phase_all = _sort_temporally(split.phase)
    mag_all   = _sort_temporally(split.mag)

    n_phase = len(phase_all)
    n_mag   = len(mag_all)
    print(f"[series] phase frames={n_phase} | mag frames={n_mag}")

    if n_mag < 32 or n_phase < 32:
        raise SystemExit(f"[ERROR] Need at least 32 PHASE + 32 MAG. Found {n_phase} / {n_mag}. Pick the correct PC-MRI series.")

    if n_phase == 64 and n_mag >= 32:
        # Align 64→32 phase frames to the first 32 MAG frames (training semantics)
        mag_sel  = mag_all[:32]
        phase_sel = _align_phase_to_mag(phase_all, mag_sel)
    else:
        # Use the first 32 of each (already time-sorted)
        phase_sel = phase_all[:32]
        mag_sel   = mag_all[:32]

    # Build RAW stacks
    phase_raw = np.stack([ds.pixel_array.astype(np.float32) for ds in phase_sel], axis=0)
    mag_raw   = np.stack([ds.pixel_array.astype(np.float32) for ds in mag_sel],   axis=0)

    # Physics stacks in [0,1]
    phase_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in phase_sel], axis=0)
    mag_01   = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in mag_sel],   axis=0)

    return phase_raw, mag_raw, phase_01, mag_01

def _overlay_and_save(img01: np.ndarray, mask: np.ndarray, out_path: Path, title: str = ""):
    img_u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8) if img01.dtype != np.uint8 else img01
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    surf = (mask > 0).astype(np.uint8)
    ys, xs = np.where(surf)
    if ys.size:
        overlay = rgb.copy()
        overlay[ys, xs] = (overlay[ys, xs] * 0.4 + np.array([255, 128, 0]) * 0.6).astype(np.uint8)
        rgb = cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0)
    if title:
        cv2.putText(rgb, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), rgb)

def _center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    if arr.ndim == 3:
        T, H, W = arr.shape
        y0 = (H - crop) // 2
        x0 = (W - crop) // 2
        return arr[:, y0:y0+crop, x0:x0+crop]
    elif arr.ndim == 2:
        H, W = arr.shape
        y0 = (H - crop) // 2
        x0 = (W - crop) // 2
        return arr[y0:y0+crop, x0:x0+crop]
    else:
        raise ValueError("Array must be 2-D or 3-D")

def _pad_crop_to_size(mask_crop: np.ndarray, H: int, W: int, crop: int) -> np.ndarray:
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    out = np.zeros((H, W), dtype=mask_crop.dtype)
    out[y0:y0+crop, x0:x0+crop] = mask_crop
    return out

def _zscore_per_sample(x: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + eps)

def _first_pc(vol: np.ndarray) -> np.ndarray:
    C, H, W = vol.shape
    X = vol.reshape(C, H * W).astype(np.float32)
    X = X - X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc = (S[0] * Vt[0]).reshape(H, W)
    pc = (pc - pc.mean()) / (pc.std() + 1e-8)
    return pc

def _build_input_from_mode_crop_RAW(
    phase_crop_raw: np.ndarray,
    mag_crop_raw:   np.ndarray,
    input_mode:     str = "full"
) -> np.ndarray:
    mode = (input_mode or "full").lower()

    def _fit32(x: np.ndarray) -> np.ndarray:
        T, H, W = x.shape
        out = np.zeros((32, H, W), dtype=x.dtype)
        t = min(T, 32)
        out[:t] = x[:t]
        return out

    phase = _fit32(phase_crop_raw)
    mag   = _fit32(mag_crop_raw) if mag_crop_raw.size else np.zeros_like(phase)

    if mode == "full":
        x = np.concatenate([phase, mag], axis=0)  # (64,Hc,Wc)
    elif mode == "pca":
        vol = np.concatenate([phase, mag], axis=0)
        x = _first_pc(vol)[None, ...]
    elif mode == "dft_power":
        x = dft_bandpower_excl_dc(phase)[None, ...]
    elif mode == "tvt":
        x = temporal_tv(phase)[None, ...]
    elif mode == "std":
        x = temporal_std(phase)[None, ...]
    elif mode == "dft_k123":
        x = dft_magnitudes_bins(phase, bins=(1, 2, 3))
    else:
        raise ValueError(f"Unknown input_mode '{input_mode}'")

    x = _zscore_per_sample(x).astype(np.float32)
    print(f"[input] mode={mode} shape={x.shape} mean={x.mean():.3f} std={x.std():.3f}")
    return x

def _predict_mask_from_crop_input(x_chw: np.ndarray, model_cfg: dict) -> np.ndarray:
    mode = (model_cfg.get("input_mode", "full") or "full").lower()
    in_ch = 64 if mode == "full" else (3 if mode == "dft_k123" else 1)
    base_ch = int(model_cfg.get("base_channels", 32))
    dev_str = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)

    model = UNet2D(in_channels=in_ch, out_channels=1, base_channels=base_ch).to(device)

    ckpt_path = Path(model_cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if load_ckpt is not None:
        ckpt = load_ckpt(str(ckpt_path), map_location=device)
        state_dict = ckpt.get("state_dict") or ckpt.get("model")
        if state_dict is None:
            raise KeyError(f"Checkpoint missing 'state_dict'/'model'. Keys: {list(ckpt.keys())}")
        model.load_state_dict(state_dict)
    else:
        sd = torch.load(str(ckpt_path), map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)

    thresh = float(model_cfg.get("threshold", 0.5))
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_chw[None, ...].astype(np.float32)).to(device)
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    print(f"[probs] min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f}")
    mask = (probs >= thresh).astype(np.uint8)
    print(f"[mask] sum={int(mask.sum())} (of {mask.size}) at thr={thresh}")
    return mask

def _print_series_inventory(dicomdir: Path, desc_filter: Optional[str] = None):
    inv = inventory_dicomdir(dicomdir)
    print("\nSeries inventory from DICOMDIR")
    print("---------------------------------------------------------------------------------------------")
    print(f"{'Series#':>7}  {'SeriesDescription':<20}  {'SeriesUID (truncated)':<26}  {'PHASE':>5}  {'MAG':>5}  {'TOTAL':>5}")
    print("---------------------------------------------------------------------------------------------")
    import re
    desc_re = re.compile(desc_filter, re.IGNORECASE) if desc_filter else None
    for s in inv:
        sd = s['series_description'] or ""
        if desc_re and not desc_re.search(sd):
            continue
        suid = (s['series_uid'] or "")[:24] + ("…" if s['series_uid'] and len(s['series_uid']) > 24 else "")
        sn = "-" if s['series_number'] is None else str(s['series_number'])
        print(f"{sn:>7}  {sd:<20.20}  {suid:<26}  {s['n_phase']:>5}  {s['n_mag']:>5}  {s['n_total']:>5}")
    print("---------------------------------------------------------------------------------------------\n")
    print("No series selected. Set 'series_selection.series_number' in config_bio.yaml and re-run.")
    raise SystemExit(0)

def _save_flow_plot(flow: np.ndarray, t: np.ndarray, out_path: Path, title: str = "Flow curve"):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, flow, linewidth=1.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Flow (mm³/s)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------------- biomarker core ---------------------------------

def _compute_biomarkers_from_full(
    phase_full01: np.ndarray,
    mag_full01:   np.ndarray,
    mask_full:    np.ndarray,
    meta: Dict[str, float],
    interpolate_n: int,
    flow_plot_cfg: dict,
    sample_id: str,
    out_dir: Path,
) -> Dict[str, float]:
    mask_full = (mask_full > 0).astype(np.uint8)
    if mask_full.sum() == 0:
        raise ValueError("Empty ROI mask → cannot compute flow.")

    phase_unit = _phase01_to_unit(phase_full01)

    res = compute_flow_and_stroke_volume(
        phase_vol=phase_unit,
        mask=mask_full,
        metadata={"v_enc": meta["v_enc"], "pixel_size": meta["pixel_size"], "trigger_delay": meta["delay_trigger"]},
        magnitude_vol=mag_full01,
        ref_mask=None,
        use_background_correction=True,
        interpolate_n=int(interpolate_n),
    )

    flow_interp = res["flow_interp"]
    t_interp    = res["t_interp"]

    pos = np.clip(flow_interp, 0.0, None)
    neg = np.clip(-flow_interp, 0.0, None)
    v_plus  = float(np.trapz(pos, t_interp))
    v_minus = float(np.trapz(neg, t_interp))
    stroke_vol_absint = v_plus + v_minus
    flow_amp          = float(np.max(flow_interp) - np.min(flow_interp))

    px = float(meta["pixel_size"])
    roi_area_mm2 = float(mask_full.astype(np.float32).sum()) * (px * px)

    if bool(flow_plot_cfg.get("save_flow_plot", False)):
        ext = (flow_plot_cfg.get("flow_plot_ext", "png") or "png").lower()
        if ext not in ("png", "jpg", "jpeg"):
            ext = "png"
        flow_name = f"{sample_id}_flow_curve.{ext}"
        _save_flow_plot(flow_interp, t_interp, out_dir / flow_name, title=f"Flow curve: {sample_id}")

    return {"roi_area_mm2": roi_area_mm2,
            "stroke_volume_mm3": stroke_vol_absint,
            "flow_amplitude_mm3_per_s": flow_amp}

# ---------------------------- runners ----------------------------------------

def run_manual(cfg: dict) -> Dict[str, float]:
    dicomdir = Path(cfg["paths"]["dicomdir"])
    mask_txt = Path(cfg["manual"]["mask_txt"])
    sample_id = cfg["common"]["sample_id"]
    out_dir = Path(cfg["paths"]["output_dir"])
    _ensure_dir(out_dir)

    sernum = cfg["series_selection"].get("series_number", None)
    if sernum is None:
        _print_series_inventory(dicomdir, desc_filter=cfg["series_selection"].get("desc_filter_regex", None))

    split  = load_series_from_dicomdir(dicomdir, int(sernum))
    # For manual mode we only need [0,1] stacks for physics
    phase_raw, mag_raw, phase_full01, mag_full01 = _series_to_full_stacks_RAW_and_01(split)
    if phase_full01.size == 0:
        raise RuntimeError("No phase images found for the selected series.")

    mask_small, cx, cy = make_mask_from_txt(mask_txt, one_based=bool(cfg["manual"].get("roi_one_based", False)))
    H, W = phase_full01.shape[1], phase_full01.shape[2]
    mask_full = embed_mask_in_image((H, W), mask_small, cx, cy).astype(np.uint8)

    ds = split.phase[-1] if split.phase else split.mag[-1]
    px = getattr(ds, "PixelSpacing", None)
    pixel_size = float(np.mean([float(v) for v in px])) if px is not None else float(cfg["fallbacks"]["pixel_size_mm"])
    trig = _extract_trigger_time(ds)
    delay_ms = float(trig) if trig is not None else float(cfg["fallbacks"]["delay_trigger_ms"])
    elem = ds.get((0x0018, 0x9199), None)  # VelocityEncoding (if present)
    venc = float(elem.value if (elem is not None and hasattr(elem, "value")) else cfg["fallbacks"]["v_enc"])

    biomarkers = _compute_biomarkers_from_full(
        phase_full01=phase_full01, mag_full01=mag_full01, mask_full=mask_full,
        meta={"v_enc": venc, "pixel_size": pixel_size, "delay_trigger": delay_ms},
        interpolate_n=int(cfg["biomarkers"].get("interpolate_n", 3201)),
        flow_plot_cfg=cfg.get("outputs", {}), sample_id=sample_id, out_dir=out_dir,
    )

    if bool(cfg["outputs"].get("save_overlays", True)):
        base = (mag_full01 if mag_full01.size else phase_full01)[(phase_full01.shape[0]-1)//2]
        _overlay_and_save(base, mask_full, out_dir / f"{sample_id}_manual_overlay.png", "Manual ROI (surface)")

    biomarkers["sample"] = sample_id
    return biomarkers


def run_auto(cfg: dict) -> Dict[str, float]:
    dicomdir = Path(cfg["paths"]["dicomdir"])
    sample_id = cfg["common"]["sample_id"]
    out_dir = Path(cfg["paths"]["output_dir"])
    _ensure_dir(out_dir)

    sernum = cfg["series_selection"].get("series_number", None)
    if sernum is None:
        _print_series_inventory(dicomdir, desc_filter=cfg["series_selection"].get("desc_filter_regex", None))

    split  = load_series_from_dicomdir(dicomdir, int(sernum))

    # Build BOTH paths: RAW for model, [0,1] for physics — with phase→mag alignment
    phase_raw, mag_raw, phase_full01, mag_full01 = _series_to_full_stacks_RAW_and_01(split)
    if phase_raw.size == 0:
        raise RuntimeError("No phase images found for the selected series.")

    crop = int(cfg["model"].get("crop_size", 80))
    H, W = phase_raw.shape[1], phase_raw.shape[2]

    # Crop RAW stacks for model input
    phase_crop_raw = _center_crop(phase_raw, crop)
    mag_crop_raw   = _center_crop(mag_raw,   crop)

    # Build features like dataset (from RAW), then z-score
    x_crop = _build_input_from_mode_crop_RAW(
        phase_crop_raw=phase_crop_raw,
        mag_crop_raw=mag_crop_raw,
        input_mode=cfg["model"].get("input_mode", "full"),
    )

    # Predict CROPPED mask, then pad back to (H,W)
    mask_crop = _predict_mask_from_crop_input(x_crop, cfg["model"])
    mask_full = _pad_crop_to_size(mask_crop, H=H, W=W, crop=crop).astype(np.uint8)

    # DICOM metadata
    ds = split.phase[-1] if split.phase else split.mag[-1]
    px = getattr(ds, "PixelSpacing", None)
    pixel_size = float(np.mean([float(v) for v in px])) if px is not None else float(cfg["fallbacks"]["pixel_size_mm"])
    trig = _extract_trigger_time(ds)
    delay_ms = float(trig) if trig is not None else float(cfg["fallbacks"]["delay_trigger_ms"])
    elem = ds.get((0x0018, 0x9199), None)
    venc = float(elem.value if (elem is not None and hasattr(elem, "value")) else cfg["fallbacks"]["v_enc"])

    biomarkers = _compute_biomarkers_from_full(
        phase_full01=phase_full01, mag_full01=mag_full01, mask_full=mask_full,
        meta={"v_enc": venc, "pixel_size": pixel_size, "delay_trigger": delay_ms},
        interpolate_n=int(cfg["biomarkers"].get("interpolate_n", 3201)),
        flow_plot_cfg=cfg.get("outputs", {}), sample_id=sample_id, out_dir=out_dir,
    )

    if bool(cfg["outputs"].get("save_overlays", True)):
        base = (mag_full01 if mag_full01.size else phase_full01)[(phase_full01.shape[0]-1)//2]
        _overlay_and_save(base, mask_full, out_dir / f"{sample_id}_auto_overlay.png", "Auto ROI (surface)")

    biomarkers["sample"] = sample_id
    return biomarkers

# --------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute CSF biomarkers (with BC, no aliasing) from a DICOMDIR.")
    ap.add_argument("--config", required=True, type=str, help="Path to config_bio.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mode = (cfg.get("mode", "manual") or "manual").lower()
    out_dir = Path(cfg["paths"]["output_dir"])
    _ensure_dir(out_dir)

    if mode == "manual":
        row = run_manual(cfg)
    elif mode == "auto":
        row = run_auto(cfg)
    else:
        raise ValueError("config.mode must be 'manual' or 'auto'")

    csv_path = out_dir / cfg["paths"]["output_csv"]
    write_header = not csv_path.exists()

    col_area = cfg["biomarkers"]["roi_area_name"]
    col_amp  = cfg["biomarkers"]["flow_amplitude_name"]
    col_sv   = cfg["biomarkers"]["stroke_volume_name"]
    fieldnames = ["sample", col_area, col_amp, col_sv]

    out_row = {
        "sample": row["sample"],
        col_area: row["roi_area_mm2"],
        col_amp:  row["flow_amplitude_mm3_per_s"],
        col_sv:   row["stroke_volume_mm3"],
    }
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(out_row)

    print(f"[OK] Saved biomarkers → {csv_path}")
    print(out_row)

if __name__ == "__main__":
    main()
