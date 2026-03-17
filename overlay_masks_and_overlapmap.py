#!/usr/bin/env python3
# overlay_masks_and_overlapmap.py
"""
Purpose
-------
Generate two figures from a PC-MRI DICOMDIR + six manual ROI masks (3 raters × 2 sessions):

  1) Phase image with the six mask CONTOURS overlaid:
     - Colors encode raters: Cyan = Kimi, Magenta = Léo, Yellow = Olivier
     - Line style encodes session: solid = session 1, dashed = session 2

  2) Overlap HEAT MAP (discrete 0..6) showing how many masks overlap per pixel:
     - 0 overlap = black background
     - 1..6 overlap = discrete, perceptually ordered colors
     - Includes a colorbar labeled “# of overlapping masks (0–6)”.

Notes
-----
• No external config file. Edit the “USER SETTINGS” block below.
• Uses the same series handling logic as your pipeline (64→32 phase alignment by TriggerTime,
  fallbacks by InstanceNumber and std-based choice).
• Uses your repo helpers from `preprocess_dicom`: make_mask_from_txt, inventory_dicomdir,
  load_series_from_dicomdir, embed_mask_in_image, _norm01.

Outputs
-------
Saves two PNGs in OUTPUT_DIR and also displays them:
  - figure_contours.png
  - figure_overlapmap.png
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ======== USER SETTINGS (EDIT THESE) =========================================
DICOMDIR       = Path("/media/rodriguez/easystore/patients_abstract/PATIENT-ID hard-case/DICOMDIR")
SERIES_NUMBER  = 701  # int, or set to None to list series inventory and exit
OUTPUT_DIR     = Path("./outputs/overlays")

# Raters (colors): Kimi = cyan, Léo = magenta, Olivier = yellow
# Provide six ROI .txt files: {rater: (session1_path, session2_path)}
common_mask_path = '/media/rodriguez/easystore/Traitement_abstract_final/'
MASK_TXT_PATHS: Dict[str, Tuple[Path, Path]] = {
    "Exp1":    (Path(common_mask_path+"Kimi1/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt"),    Path(common_mask_path+"Kimi2/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt")),
    "Exp2":     (Path(common_mask_path+"Leo1/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt"),     Path(common_mask_path+"Leo2/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt")),
    "Exp3": (Path(common_mask_path+"Olivier1/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt"), Path(common_mask_path+"Olivier2/PATIENT-ID hard-case/2301301227/Segment/aqueduc.txt")),
}
# If your .txt coordinates are 1-based, set this to True
ROI_ONE_BASED = False

# Which phase frame to display as background (index in the selected 32 frames)
PHASE_FRAME_INDEX = 16  # middle by default (0..31)
# ============================================================================

# ---- Import project helpers (from your repo) --------------------------------
from preprocess_dicom import (
    make_mask_from_txt,
    inventory_dicomdir,
    load_series_from_dicomdir,
    embed_mask_in_image,
    _norm01 as _norm01_img,
)

# ---- Temporal helpers (aligned with your pipeline) ---------------------------
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
    to the same MAG time, pick the one with larger spatial std. Fallback (no
    times): choose even vs odd indices by higher mean std.
    """
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
                cand = diffs[0][1]
                selected_idx.append(cand)
            elif len(candidates) == 1:
                selected_idx.append(candidates[0])
            else:
                stds = []
                for ci in candidates:
                    arr = phase_ds[ci].pixel_array.astype(np.float32)
                    stds.append((float(arr.std()), ci))
                stds.sort(reverse=True)
                selected_idx.append(stds[0][1])

        # de-dup and ensure length 32
        seen, uniq = set(), []
        for i in selected_idx:
            if i not in seen:
                uniq.append(i); seen.add(i)
        if len(uniq) < 32:
            remaining = [i for _, i in ph_times if i not in seen]
            uniq.extend(remaining[: 32 - len(uniq)])
        return [phase_ds[i] for i in uniq[:32]]

    # Fallback: no times → try even vs odd
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

def _series_to_phase01(split) -> np.ndarray:
    """
    Returns a 32-frame phase stack normalized to [0,1], aligned to MAG timing when needed.
    """
    phase_all = _sort_temporally(split.phase)
    mag_all   = _sort_temporally(split.mag)
    n_phase = len(phase_all)
    n_mag   = len(mag_all)
    if n_mag < 32 or n_phase < 32:
        raise SystemExit(f"[ERROR] Need at least 32 PHASE + 32 MAG. Found {n_phase} / {n_mag}. Pick the correct series.")

    if n_phase == 64 and n_mag >= 32:
        mag_sel   = mag_all[:32]
        phase_sel = _align_phase_to_mag(phase_all, mag_sel)
    else:
        phase_sel = phase_all[:32]

    phase_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in phase_sel], axis=0)
    return phase_01  # (32,H,W)

# ---- Mask utilities ----------------------------------------------------------
def _load_and_embed_mask(txt_path: Path, shape_hw: Tuple[int, int], one_based=False) -> np.ndarray:
    """
    Build full-resolution binary mask from a ROI .txt as in your pipeline:
      make_mask_from_txt(...) -> (mask_small, cx, cy) -> embed_mask_in_image((H,W), ...)
    """
    m_small, cx, cy = make_mask_from_txt(txt_path, one_based=bool(one_based))
    H, W = shape_hw
    m_full = embed_mask_in_image((H, W), m_small, cx, cy).astype(np.uint8)
    return (m_full > 0).astype(np.uint8)

def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

# ---- Plotting ----------------------------------------------------------------
def _plot_contours_over_phase(phase_img01: np.ndarray, masks: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path):
    """
    masks: { 'Exp1': (m_sess1, m_sess2), 'Exp2': (...), 'Exp3': (...) }
    Colors: {'Exp1':'c','Exp2':'m','Exp3':'y'} ; Linestyles: sess1='-', sess2='--'
    """
    color_map = {"Exp1": "c", "Exp2": "m", "Exp3": "y"}
    ls_map    = {0: "-", 1: "--"}  # 0=session1, 1=session2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(phase_img01, cmap="gray", interpolation="nearest")
    ax.set_axis_off()

    # draw contours
    for rater, (m1, m2) in masks.items():
        for sess_idx, m in enumerate([m1, m2]):
            if m is None or m.sum() == 0:
                print(f"[WARN] Empty mask for {rater} session {sess_idx+1}")
                continue
            cnts = _find_contours(m)
            for c in cnts:
                c = c.squeeze()
                if c.ndim != 2 or c.shape[0] < 2:
                    continue
                ax.plot(c[:, 0], c[:, 1],
                        color=color_map[rater],
                        linestyle=ls_map[sess_idx],
                        linewidth=1.5,
                        alpha=0.95,
                        zorder=5)

    # legend proxies
    from matplotlib.lines import Line2D
    legend_elems = []
    for rater, color in color_map.items():
        legend_elems.append(Line2D([0], [0], color=color, lw=2, ls='-', label=f"{rater} – sess1"))
        legend_elems.append(Line2D([0], [0], color=color, lw=2, ls='--', label=f"{rater} – sess2"))
    ax.legend(handles=legend_elems, loc="lower right", fontsize=12, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def _plot_overlap_map(overlap: np.ndarray, out_path: Path):
    """
    overlap ∈ {0,1,...,6}. Use a discrete colormap with 0→black and 1..6→perceptual steps.
    """
    # Build discrete colors: 0 black, then 6 steps from a perceptual map
    # Sample from 'viridis' but keep 0 as black for background.
    base = plt.get_cmap("viridis")
    steps = [base(x) for x in np.linspace(0.15, 0.95, 6)]  # 6 levels, avoid very dark
    colors = [(0, 0, 0, 1.0)] + steps  # prepend black for 0
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 7.5, 1.0)  # bins for 0..6
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(overlap, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, 7, 1))
    cbar.set_label("# of overlapping masks (0–6)", rotation=270, labelpad=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ---- Main --------------------------------------------------------------------
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

    # Load selected series
    split = load_series_from_dicomdir(DICOMDIR, int(SERIES_NUMBER))
    phase_01 = _series_to_phase01(split)  # (32,H,W)
    T, H, W = phase_01.shape
    idx = int(np.clip(PHASE_FRAME_INDEX, 0, T - 1))
    phase_bg = phase_01[idx]

    # Prepare masks per rater/session at full resolution
    full_masks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for rater, (p1, p2) in MASK_TXT_PATHS.items():
        m1 = _load_and_embed_mask(p1, (H, W), one_based=ROI_ONE_BASED) if p1 and p1.exists() else None
        m2 = _load_and_embed_mask(p2, (H, W), one_based=ROI_ONE_BASED) if p2 and p2.exists() else None
        if m1 is None:
            print(f"[WARN] Missing/invalid path for {rater} session 1: {p1}")
            m1 = np.zeros((H, W), dtype=np.uint8)
        if m2 is None:
            print(f"[WARN] Missing/invalid path for {rater} session 2: {p2}")
            m2 = np.zeros((H, W), dtype=np.uint8)
        full_masks[rater] = (m1, m2)

    # Figure 1: contours overlay
    contours_path = OUTPUT_DIR / "figure_contours.png"
    _plot_contours_over_phase(phase_bg, full_masks, contours_path)
    print(f"[OK] Saved contours overlay → {contours_path}")

    # Figure 2: overlap heatmap (sum of six masks)
    stacked = []
    for rater in ["Exp1", "Exp2", "Exp3"]:  # enforce consistent order
        m1, m2 = full_masks[rater]
        stacked.append((m1 > 0).astype(np.uint8))
        stacked.append((m2 > 0).astype(np.uint8))
    overlap = np.stack(stacked, axis=0).sum(axis=0)  # (H,W) in 0..6

    overlap_path = OUTPUT_DIR / "figure_overlapmap.png"
    _plot_overlap_map(overlap, overlap_path)
    print(f"[OK] Saved overlap map → {overlap_path}")

if __name__ == "__main__":
    main()
