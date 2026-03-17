#!/usr/bin/env python3
# expert_consensus_model_novice_overlays.py
"""
Generate two figures from a PC-MRI DICOMDIR that compare:
  (A) Expert CONSENSUS mask (strict AND across the six expert masks),
  (B) Model segmentation (predicted with your Flow-Dice UNet2D),
  (C) Novice (Eder) session 1,
  (D) Novice (Eder) session 2.

Figures:
  1) Phase image + mask CONTOURS:
       - Expert (consensus): white, solid, thick
       - Model: green, solid
       - Novice (sess1): orange, solid
       - Novice (sess2): orange, dashed
     (Real names are NOT shown; legend uses generic labels.)

  2) Overlap HEAT MAP (0–4):
       Per-pixel count over {Expert consensus, Model, Novice-1, Novice-2}
       0 = black background; 1..4 = discrete colormap steps; labeled colorbar.

Notes
-----
• No external config file. Edit the “USER SETTINGS” section below.
• DICOM handling matches your pipeline (temporal sort, 64→32 alignment by TriggerTime;
  fallback by InstanceNumber/variance).
• ROI masks are built with your repo helpers (make_mask_from_txt + embed_mask_in_image).
• Model prediction uses your Flow-Dice UNet2D (input_mode='full' → 64 channels).

Outputs (saved in OUTPUT_DIR)
-----------------------------
  - expert_model_novice_contours.png
  - expert_model_novice_overlapmap.png
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import time

# =================== USER SETTINGS (EDIT THESE) ==============================
DICOMDIR       = Path("/media/rodriguez/easystore/patients_abstract/CAILLET Daniel/DICOMDIR")
SERIES_NUMBER  = 701                     # int, or set to None to print inventory and exit
OUTPUT_DIR     = Path("./outputs/overlays")

# ---- Expert masks (anonymous labels: Exp1, Exp2, Exp3) ----------------------
common_mask_path = "/media/rodriguez/easystore/Traitement_abstract_final/"
MASK_TXT_PATHS_EXPERTS: Dict[str, Tuple[Path, Path]] = {
    "Exp1": (Path(common_mask_path + "Kimi1/CAILLET DANIEL/2306261121/Segment/aqueduc.txt"),
             Path(common_mask_path + "Kimi2/CAILLET DANIEL/2306261121/Segment/aqueduc.txt")),
    "Exp2": (Path(common_mask_path + "Leo1/CAILLET DANIEL/2306261121/Segment/aqueduc.txt"),
             Path(common_mask_path + "Leo2/CAILLET DANIEL/2306261121/Segment/aqueduc.txt")),
    "Exp3": (Path(common_mask_path + "Olivier1/CAILLET DANIEL/2306261121/Segment/aqueduc.txt"),
             Path(common_mask_path + "Olivier2/CAILLET DANIEL/2306261121/Segment/aqueduc.txt")),
}

# ---- Novice (Eder) masks (anonymous in figure: "Novice") --------------------
EDER1_TXT = Path(common_mask_path + "Eder1/CAILLET DANIEL/2306261121/Segment/aqueduc.txt")
EDER2_TXT = Path(common_mask_path + "Eder2/CAILLET DANIEL/2306261121/Segment/aqueduc.txt")

# Are ROI indices 1-based in the .txt files?
ROI_ONE_BASED = False

# Which phase frame (0..31) to show as background
PHASE_FRAME_INDEX = 16

# ---- Model (Flow-Dice UNet2D) inference knobs --------------------------------
MODEL_CKPT       = Path("outputs/unet2d_full_c80_b32_flow_dice/checkpoints/best_model.pt")
INPUT_MODE       = "full"        # 'full' → 64 channels (32 phase + 32 mag)
BASE_CHANNELS    = 32
CROP_SIZE        = 80            # model crop size used at training
THRESHOLD        = 0.5           # sigmoid threshold
DEVICE_STR       = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================================

# ---- Import project helpers (present in your repo) --------------------------
from preprocess_dicom import (
    make_mask_from_txt,
    inventory_dicomdir,
    load_series_from_dicomdir,
    embed_mask_in_image,
    _norm01 as _norm01_img,
)

# UNet2D + checkpoint loader (same as in your project)
from src.models.unet2d import UNet2D
try:
    from src.utils.misc import load_ckpt
except Exception:
    load_ckpt = None

# ------------------------ Temporal helpers (as in pipeline) ------------------
def _extract_trigger_time(ds) -> Optional[float]:
    tt = getattr(ds, "TriggerTime", None)
    if tt is not None:
        try:
            return float(tt)
        except Exception:
            pass
    elem = ds.get((0x0018, 0x1060), None)
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
    mag_times = [(_extract_trigger_time(ds), i) for i, ds in enumerate(mag_ds)]
    ph_times  = [(_extract_trigger_time(ds), i) for i, ds in enumerate(phase_ds)]
    have_times = all(t is not None for t, _ in mag_times) and all(t is not None for t, _ in ph_times)

    if have_times:
        from collections import defaultdict
        def rkey(t): return int(round(t))
        phase_by_time = defaultdict(list)
        for t, i in ph_times:
            phase_by_time[rkey(t)].append(i)

        selected_idx: List[int] = []
        for t_mag, _ in mag_times:
            rk = rkey(t_mag)
            candidates = phase_by_time.get(rk, [])
            if not candidates:
                diffs = [(abs(t_mag - t_ph), i) for (t_ph, i) in ph_times]
                diffs.sort()
                selected_idx.append(diffs[0][1])
            elif len(candidates) == 1:
                selected_idx.append(candidates[0])
            else:
                stds = []
                for ci in candidates:
                    arr = phase_ds[ci].pixel_array.astype(np.float32)
                    stds.append((float(arr.std()), ci))
                stds.sort(reverse=True)
                selected_idx.append(stds[0][1])

        seen, uniq = set(), []
        for i in selected_idx:
            if i not in seen:
                uniq.append(i); seen.add(i)
        if len(uniq) < 32:
            remaining = [i for _, i in ph_times if i not in seen]
            uniq.extend(remaining[: 32 - len(uniq)])
        return [phase_ds[i] for i in uniq[:32]]

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
    idxs = even_idx if even_std >= odd_std else odd_idx
    return [phase_ds[i] for i in idxs]

def _series_to_phase_mag_01(split) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns aligned (phase_01, mag_01) stacks, each with 32 frames in [0,1].
    """
    phase_all = _sort_temporally(split.phase)
    mag_all   = _sort_temporally(split.mag)
    n_phase = len(phase_all)
    n_mag   = len(mag_all)
    if n_mag < 32 or n_phase < 32:
        raise SystemExit(f"[ERROR] Need at least 32 PHASE + 32 MAG. Found {n_phase}/{n_mag}.")

    if n_phase == 64 and n_mag >= 32:
        mag_sel   = mag_all[:32]
        phase_sel = _align_phase_to_mag(phase_all, mag_sel)
    else:
        phase_sel = phase_all[:32]
        mag_sel   = mag_all[:32]

    phase_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in phase_sel], axis=0)
    mag_01   = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in mag_sel],   axis=0)
    return phase_01, mag_01  # (32,H,W), (32,H,W)

# ---------------------------- Mask utilities ---------------------------------
def _load_and_embed_mask(txt_path: Path, shape_hw: Tuple[int, int], one_based=False) -> np.ndarray:
    m_small, cx, cy = make_mask_from_txt(txt_path, one_based=bool(one_based))
    H, W = shape_hw
    m_full = embed_mask_in_image((H, W), m_small, cx, cy).astype(np.uint8)
    return (m_full > 0).astype(np.uint8)

def _expert_consensus_from_six(expert_paths: Dict[str, Tuple[Path, Path]], shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Build strict consensus AND across ALL SIX expert masks (3 raters × 2 sessions).
    A pixel is 1 only if it is selected in each of the six masks.
    """
    H, W = shape_hw
    accum = np.ones((H, W), dtype=np.uint8)
    for rater, (p1, p2) in expert_paths.items():
        for sess_idx, p in enumerate((p1, p2), start=1):
            if not p.exists():
                print(f"[WARN] Missing expert mask: {rater} session {sess_idx} → treated as empty")
                current = np.zeros((H, W), dtype=np.uint8)
            else:
                current = _load_and_embed_mask(p, (H, W), one_based=ROI_ONE_BASED)
            accum = accum & current
    return accum.astype(np.uint8)

def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

# ---------------------------- Model prediction --------------------------------
def _center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    if arr.ndim == 3:
        T, H, W = arr.shape
        y0 = (H - crop) // 2
        x0 = (W - crop) // 2
        return arr[:, y0:y0+crop, x0:x0+crop]
    raise ValueError("Expected 3D array (T,H,W).")

def _zscore_per_sample(x: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + eps)

def _pad_crop_to_size(mask_crop: np.ndarray, H: int, W: int, crop: int) -> np.ndarray:
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    out = np.zeros((H, W), dtype=mask_crop.dtype)
    out[y0:y0+crop, x0:x0+crop] = mask_crop
    return out

def _build_input_from_mode_crop_RAW(phase_crop_raw: np.ndarray, mag_crop_raw: np.ndarray, input_mode: str = "full") -> np.ndarray:
    mode = (input_mode or "full").lower()

    def _fit32(x: np.ndarray) -> np.ndarray:
        T, H, W = x.shape
        out = np.zeros((32, H, W), dtype=x.dtype)
        t = min(T, 32)
        out[:t] = x[:t]
        return out

    # For 'full', model expects 64 channels: 32 phase + 32 mag
    phase = _fit32(phase_crop_raw)
    mag   = _fit32(mag_crop_raw)
    if mode != "full":
        raise ValueError("This script is set up for INPUT_MODE='full' (64-ch). Adjust if needed.")
    x = np.concatenate([phase, mag], axis=0)  # (64,Hc,Wc)

    x = _zscore_per_sample(x).astype(np.float32)
    print(f"[model input] mode={mode} shape={x.shape} mean={x.mean():.3f} std={x.std():.3f}")
    return x

def _predict_mask_from_crop_input(x_chw: np.ndarray) -> np.ndarray:
    device = torch.device(DEVICE_STR)
    model = UNet2D(in_channels=64, out_channels=1, base_channels=int(BASE_CHANNELS)).to(device)

    if not MODEL_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CKPT}")

    t_start_total = time.perf_counter()
    # Load checkpoint (measure)
    t0 = time.perf_counter()
    if load_ckpt is not None:
        ckpt = load_ckpt(str(MODEL_CKPT), map_location=device)
        state_dict = ckpt.get("state_dict") or ckpt.get("model")
        if state_dict is None:
            raise KeyError(f"Checkpoint missing 'state_dict'/'model'. Keys: {list(ckpt.keys())}")
        model.load_state_dict(state_dict)
    else:
        sd = torch.load(str(MODEL_CKPT), map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    t_load = time.perf_counter() - t0

    model.eval()
    # Inference (measure)
    t1 = time.perf_counter()
    with torch.no_grad():
        x_t = torch.from_numpy(x_chw[None, ...].astype(np.float32)).to(device)
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    t_inf = time.perf_counter() - t1
    t_total = time.perf_counter() - t_start_total

    mask = (probs >= float(THRESHOLD)).astype(np.uint8)
    print(f"[model probs] min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f} thr={THRESHOLD}")
    print(f"[model mask] sum={int(mask.sum())} / {mask.size}")
    print(f"[timing] checkpoint load: {t_load:.3f}s | inference: {t_inf:.3f}s | total (_predict): {t_total:.3f}s")
    return mask

# ------------------------------- Plotting ------------------------------------
def _plot_contours_over_phase(phase_img01: np.ndarray,
                              expert_cons: np.ndarray,
                              model_mask: np.ndarray,
                              novice1: np.ndarray,
                              novice2: np.ndarray,
                              out_path: Path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(phase_img01, cmap="gray", interpolation="nearest")
    ax.set_axis_off()

    def draw_cnt(m: np.ndarray, color, ls="-", lw=2.0, alpha=0.95, z=5):
        if m is None or m.sum() == 0:
            return
        for c in _find_contours(m):
            c = c.squeeze()
            if c.ndim != 2 or c.shape[0] < 2:
                continue
            ax.plot(c[:, 0], c[:, 1], color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=z)

    # Expert consensus (white, thick)
    draw_cnt(expert_cons, color="w", ls="-", lw=2.5, z=7)
    # Model (green)
    draw_cnt(model_mask, color="g", ls="-", lw=2.0, z=6)
    # Novice (orange)
    draw_cnt(novice1, color=(1.0, 0.5, 0.0), ls="-",  lw=1.8, z=6)
    draw_cnt(novice2, color=(1.0, 0.5, 0.0), ls="--", lw=1.8, z=6)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="w", lw=2.5, ls="-",  label="Expert (consensus)"),
        Line2D([0], [0], color="g", lw=2.0,  ls="-",  label="Model"),
        Line2D([0], [0], color=(1.0, 0.5, 0.0), lw=1.8, ls="-",  label="Novice (sess1)"),
        Line2D([0], [0], color=(1.0, 0.5, 0.0), lw=1.8, ls="--", label="Novice (sess2)"),
    ]
    #ax.legend(handles=legend_elems, loc="lower right", fontsize=12, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def _plot_overlap_map(overlap: np.ndarray, out_path: Path):
    """
    overlap ∈ {0,1,2,3,4} over (Expert consensus, Model, Novice1, Novice2).
    0 → black; 1..4 → stepped colors; labeled colorbar.
    """
    base = plt.get_cmap("viridis")
    steps = [base(x) for x in np.linspace(0.15, 0.95, 4)]  # 4 levels
    colors = [(0, 0, 0, 1.0)] + steps
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 4.5 + 1, 1.0)  # -0.5..4.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 5, 1))
    cbar.set_label("# of overlapping masks (0–4)", rotation=270, labelpad=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ------------------------------- Main ----------------------------------------
def _print_series_inventory_and_exit(dicomdir: Path):
    inv = inventory_dicomdir(dicomdir)
    print("\nSeries inventory from DICOMDIR")
    print("---------------------------------------------------------------------------------------------")
    print(f"{'Series#':>7}  {'SeriesDescription':<20}  {'SeriesUID (truncated)':<26}  {'PHASE':>5}  {'MAG':>5}  {'TOTAL':>5}")
    print("---------------------------------------------------------------------------------------------")
    for s in inv:
        sd = s['series_description'] or ""
        suid = (s['series_uid'] or "")[:24] + ("…" if s['series_uid'] and len(s['series_uid']) > 24 else "")
        sn = "-" if s['series_number'] is None else str(s['series_number'])
        print(f"{sn:>7}  {sd:<20.20}  {suid:<26}  {s['n_phase']:>5}  {s['n_mag']:>5}  {s['n_total']:>5}")
    print("---------------------------------------------------------------------------------------------\n")
    raise SystemExit(0)

def main():
    if SERIES_NUMBER is None:
        _print_series_inventory_and_exit(DICOMDIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load DICOM series and build phase/mag stacks (32 each, [0,1])
    split = load_series_from_dicomdir(DICOMDIR, int(SERIES_NUMBER))
    phase_01, mag_01 = _series_to_phase_mag_01(split)
    T, H, W = phase_01.shape
    idx = int(np.clip(PHASE_FRAME_INDEX, 0, T - 1))
    phase_bg = phase_01[idx]

    # ---- Expert consensus (strict AND across all six expert masks)
    expert_cons = _expert_consensus_from_six(MASK_TXT_PATHS_EXPERTS, (H, W))

    # ---- Novice (Eder) sessions
    novice1 = _load_and_embed_mask(EDER1_TXT, (H, W), one_based=ROI_ONE_BASED) if EDER1_TXT.exists() else np.zeros((H, W), np.uint8)
    novice2 = _load_and_embed_mask(EDER2_TXT, (H, W), one_based=ROI_ONE_BASED) if EDER2_TXT.exists() else np.zeros((H, W), np.uint8)
    if not EDER1_TXT.exists(): print(f"[WARN] Missing novice session 1: {EDER1_TXT}")
    if not EDER2_TXT.exists(): print(f"[WARN] Missing novice session 2: {EDER2_TXT}")

    # ---- Model segmentation (Flow-Dice UNet2D; input_mode='full')
    # Rebuild RAW-like input for model: use [0,1] then z-score (consistent with your pipeline’s features)
    # Center-crop for model input, then pad back to full size.
    phase_crop = _center_crop(phase_01, CROP_SIZE)
    mag_crop   = _center_crop(mag_01,   CROP_SIZE)
    x_crop = _build_input_from_mode_crop_RAW(phase_crop, mag_crop, input_mode=INPUT_MODE)
    model_mask_crop = _predict_mask_from_crop_input(x_crop)
    model_mask_full = _pad_crop_to_size(model_mask_crop, H=H, W=W, crop=CROP_SIZE).astype(np.uint8)

    # ---- Figure 1: phase + contours
    contours_path = OUTPUT_DIR / "expert_model_novice_contours.png"
    _plot_contours_over_phase(phase_bg, expert_cons, model_mask_full, novice1, novice2, contours_path)
    print(f"[OK] Saved contours overlay → {contours_path}")

    # ---- Figure 2: overlap heat map (0..4)
    stack = np.stack([
        (expert_cons      > 0).astype(np.uint8),
        (model_mask_full  > 0).astype(np.uint8),
        (novice1          > 0).astype(np.uint8),
        (novice2          > 0).astype(np.uint8),
    ], axis=0)
    overlap = stack.sum(axis=0)  # 0..4
    overlap_path = OUTPUT_DIR / "expert_model_novice_overlapmap.png"
    _plot_overlap_map(overlap, overlap_path)
    print(f"[OK] Saved overlap map → {overlap_path}")

if __name__ == "__main__":
    main()
