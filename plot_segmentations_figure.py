#!/usr/bin/env python3

# plot_segmentation_figures.py
"""
Generate compact figure with qualitative overlays + flow curves
for 2 volunteers + 2 patients under:
  - No shift
  - Two random cyclic shifts (k1, k2; per-subject; seeded)
  - One random shuffle (per-subject; seeded)

Contours (image rows):
  Manual (green), Full-Dice (blue), Full-FlowDice (orange).

Flow curves (below each image row):
  Manual (green), Full-Dice (blue), Full-FlowDice (orange).

Figure:
  - Anonymized subject labels in-figure ("Volunteer 1/2", "Patient 1/2").
  - Real subject IDs are printed to the terminal.
  - Output: fig/best_segmentations.png (2-column friendly).
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from csf_flow import compute_flow_and_stroke_volume
from eval_biomarkers import load_metadata_table, build_annular_ref_mask
from src.utils.temporal import reorder_temporal_images

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_ROOT = Path("data/test")
PRED_DIR_DICE = Path("outputs/preds_biomarkers/full_dice")
PRED_DIR_FLOW = Path("outputs/preds_biomarkers/full_flowdice")
META_CSV = Path("merged_metadata_file.csv")
OUT_FIG = Path("fig/best_segmentations.png")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Thresholds (if preds are probabilities)
THRESH_DICE = 0.5
THRESH_FLOW = 0.15

N_VOL = 2
N_PAT = 2

# Colors (consistent mapping for contours and curves)
COLORS = {
    "manual": "green",
    "dice":   "blue",
    "flow":   "orange",
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def canon_flow_meta(meta: dict) -> dict:
    """
    Map CSV keys to the keys expected by csf_flow.compute_flow_and_stroke_volume.
    """
    if meta is None:
        raise KeyError("Missing metadata for subject")
    if "delay_trigger" in meta:
        td = float(meta["delay_trigger"])
    elif "trigger_delay" in meta:
        td = float(meta["trigger_delay"])
    else:
        raise KeyError("Neither 'delay_trigger' nor 'trigger_delay' found in metadata")

    return {
        "v_enc": float(meta["v_enc"]),
        "pixel_size": float(meta["pixel_size"]),
        "trigger_delay": td,  # canonical key
    }

def normalize_img(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

def pick_frame(phase: np.ndarray, mask_2d: np.ndarray) -> int:
    """Pick frame index with max |mean phase| inside the manual ROI."""
    vals = []
    roi = mask_2d.astype(bool)
    for t in range(phase.shape[0]):
        vals.append(abs(phase[t][roi]).mean() if roi.any() else 0.0)
    return int(np.argmax(vals))

def load_subject_data(sid: str):
    subj_dir = DATA_ROOT / sid
    phase = np.load(subj_dir / "phase.npy")  # (32,240,240)
    mag   = np.load(subj_dir / "mag.npy")    # (32,240,240)
    manual = np.load(subj_dir / "mask.npy")  # (240,240)
    dice_pred = np.load(PRED_DIR_DICE / f"{sid}_pred.npy")
    flow_pred = np.load(PRED_DIR_FLOW / f"{sid}_pred.npy")

    # Scale phase to [-1,1] if stored in [0,1]
    if phase.min() >= -1e-6 and phase.max() <= 1 + 1e-6:
        phase = phase * 2.0 - 1.0

    # Threshold predictions if arrays are probabilities (float)
    if dice_pred.dtype != np.uint8 and dice_pred.dtype != np.bool_:
        dice_mask = (dice_pred >= THRESH_DICE).astype(np.uint8)
    else:
        dice_mask = dice_pred.astype(np.uint8)

    if flow_pred.dtype != np.uint8 and flow_pred.dtype != np.bool_:
        flow_mask = (flow_pred >= THRESH_FLOW).astype(np.uint8)
    else:
        flow_mask = flow_pred.astype(np.uint8)

    return phase, mag, manual, dice_mask, flow_mask

def compute_flow_curves(phase, mag, masks, meta):
    """
    Compute baseline-corrected flow for each mask using annular static-tissue reference.
    Returns dict of {tag: flow_corr (T,)} with native 32-frame sampling.
    """
    flows = {}
    for tag, mask in masks.items():
        ref_mask = build_annular_ref_mask(mask, magnitude=mag)
        res = compute_flow_and_stroke_volume(phase, mask, meta, ref_mask=ref_mask)
        flows[tag] = res["flow_corr"]
    return flows

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Subject discovery and selection (seeded)
    all_subjects = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    volunteers = sorted([s for s in all_subjects if s.startswith("Volunteer")])
    patients   = sorted([s for s in all_subjects if s.startswith("Patient")])

    chosen_vols = random.sample(volunteers, N_VOL)
    chosen_pats = random.sample(patients,   N_PAT)
    chosen = chosen_vols + chosen_pats  # preserve order: vols then pats

    # Build anonymized labels and print mapping
    anon_labels = []
    print("Anonymization mapping:")
    for i, sid in enumerate(chosen_vols, 1):
        label = f"Volunteer {i}"
        anon_labels.append((sid, label))
        print(f"  {label}  \u2190  {sid}")
    for i, sid in enumerate(chosen_pats, 1):
        label = f"Patient {i}"
        anon_labels.append((sid, label))
        print(f"  {label}  \u2190  {sid}")

    # Metadata
    meta_map = load_metadata_table(META_CSV)
    first_flow_handles = None
    first_flow_labels = None
    legend_target_ax = None  # will hold the bottom-left MRI axes

    # Figure layout:
    # - Larger figsize
    # - Image rows get more height to enlarge MRI panels (contours more visible)
    fig = plt.figure(figsize=(12.0, 10.8))
    gs = GridSpec(
        len(chosen) * 2, 4, figure=fig,
        height_ratios=[1.9, 0.45] * len(chosen),  # ↑ image taller, flow shorter
        hspace=0.07, wspace=0.04
    )

    # Column headers (generic; per-subject k's are annotated inside panels)
    col_titles = ["No shift", "Shift k1", "Shift k2", "Shuffle"]

    # Iterate subjects / rows
    for r, (sid, anon) in enumerate(anon_labels):
        phase, mag, manual, dice_mask, flow_mask = load_subject_data(sid)
        raw_meta = meta_map.get(sid)
        meta = canon_flow_meta(raw_meta)

        # Per-subject random k1, k2, and shuffle perm (seeded)
        k1, k2 = np.random.choice(range(1, 32), 2, replace=False)
        perm = np.random.permutation(32)

        conds = [
            ("baseline", None),
            ("shift", k1),
            ("shift", k2),
            ("shuffle", perm),
        ]

        # Grid rows for this subject block
        row_img = 2 * r
        row_flow = row_img + 1

        for c, (ctype, param) in enumerate(conds):
            # Apply temporal transform
            if ctype == "baseline":
                ph, mg = phase, mag
                man, dm, fm = manual, dice_mask, flow_mask
            elif ctype == "shift":
                ph, mg, _ = reorder_temporal_images(phase, mag, int(param))
                man, dm, fm = manual, dice_mask, flow_mask  # masks are static (H,W)
            elif ctype == "shuffle":
                ph, mg = phase[perm], mag[perm]
                man, dm, fm = manual, dice_mask, flow_mask
            else:
                raise ValueError("Unknown condition")

            # Compute flows (native 32-frame)
            flows = compute_flow_curves(
                ph, mg,
                {"manual": man, "dice": dm, "flow": fm},
                meta
            )

            # Pick representative frame for overlays
            t_star = pick_frame(ph, man)

            # ---------- Image panel ----------
            ax_img = fig.add_subplot(gs[row_img, c])
            bg = normalize_img(mg[t_star])
            ax_img.imshow(bg, cmap="gray", interpolation="nearest")
            # Contours (slightly thicker for visibility)
            for tag, m in (("manual", man), ("dice", dm), ("flow", fm)):
                ax_img.contour(m.astype(float), levels=[0.5],
                               colors=[COLORS[tag]], linewidths=1.6)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            if r == 0:
                ax_img.set_title(col_titles[c], fontsize=10, pad=3)
            if c == 0:
                ax_img.set_ylabel(anon, fontsize=9)

            # Annotate k inside shift panels (per-subject values) with opaque white box
            if ctype == "shift":
                ax_img.text(
                    4, 12, f"k={int(param)}",
                    color="black", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8, alpha=1.0)
                )

            # Remember the bottom-left MRI axes (for legend placement later)
            if r == len(anon_labels) - 1 and c == 0:
                legend_target_ax = ax_img

            # ---------- Flow panel ----------
            ax_flow = fig.add_subplot(gs[row_flow, c])
            x = np.arange(32)

            ln1, = ax_flow.plot(x, flows["manual"], color=COLORS["manual"], lw=1.3, label="Manual")
            ln2, = ax_flow.plot(x, flows["dice"],   color=COLORS["dice"],   lw=1.3, label="Full-Dice")
            ln3, = ax_flow.plot(x, flows["flow"],   color=COLORS["flow"],   lw=1.3, label="Full-FlowDice")

            # Capture handles/labels only once (top-left flow panel)
            if r == 0 and c == 0:
                first_flow_handles = [ln1, ln2, ln3]
                first_flow_labels = ["Manual", "Full-Dice", "Full-FlowDice"]

            ax_flow.set_xlim(0, 31)

            # Ensure vertical gridlines visible at [0, 8, 16, 24, 31]
            major_ticks = [0, 8, 16, 24, 31]
            ax_flow.set_xticks(major_ticks)

            if r == len(anon_labels) - 1:
                ax_flow.set_xticklabels([str(t) for t in major_ticks], fontsize=7)
            else:
                ax_flow.set_xticklabels([])

            # Y-axis on all rows; concise label on first column only
            if c == 0:
                ax_flow.set_ylabel("Flow (mm³/s)", fontsize=8)

            ax_flow.tick_params(axis="both", labelsize=7)

            # Grid: vertical + light horizontal
            ax_flow.grid(axis="x", which="major", alpha=0.5, linewidth=0.7)
            ax_flow.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)

            # Slim borders
            for sp in ax_flow.spines.values():
                sp.set_linewidth(0.6)

        # Put the legend on the bottom-left MRI image
        if legend_target_ax is not None and first_flow_handles is not None:
            legend_target_ax.legend(
                first_flow_handles, first_flow_labels,
                loc="lower left",
                fontsize=8,
                frameon=True, fancybox=True, framealpha=1.0,
                borderpad=0.4, handlelength=2.2
            )


    # Tight layout with trimmed margins
    plt.subplots_adjust(top=0.97, bottom=0.08, left=0.055, right=0.99, hspace=0.1, wspace=0.045)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=400, bbox_inches="tight")
    print(f"[OK] Saved figure \u2192 {OUT_FIG.resolve()}")
