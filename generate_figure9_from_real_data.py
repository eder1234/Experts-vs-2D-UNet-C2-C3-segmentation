#!/usr/bin/env python3
"""
generate_figure9_from_real_data.py

Purpose
-------
Generate 9 panel-ready images from real data for three patients:
  3 patients × 3 images each:
    1) expert_variance_vs_model.png
    2) novice_model_vs_expert_consensus.png
    3) corrected_flow_waveforms.png

Also saves numeric results so the final figure can be post-edited manually:
  - biomarkers_long.csv
  - biomarkers_summary.csv
  - difficulty_stats.csv
  - biomarker_space_points.csv
  - biomarker_space_ellipses.csv
  - expert_overlap_count.npy / expert_overlap_fraction.npy / expert_std_map.npy per patient
  - flow_curves_long.csv
  - patient_manifest.csv

This script intentionally reuses your existing project logic wherever possible:
  - DICOM loading / mask embedding from preprocess_dicom.py
  - flow computation / background correction from csf_flow.py
  - UNet definition from unet2d.py (or src.models.unet2d if available)
  - temporal alignment logic adapted from your overlay script
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ------------------------------ USER SETTINGS ---------------------------------
PATIENTS = [
    ("BOUCHER", "BOUCHER Amelie"),
    ("CAILLET", "CAILLET Daniel"),
    ("GRONIER", "GRONIER Michel"),
]

DATA_ROOT = Path("/media/rodriguez/easystore/patients_abstract")
MASK_ROOT = Path("/media/rodriguez/easystore/Traitement_abstract_final")
OUTPUT_ROOT = Path("outputs/figure9_real_data")

# Model
MODEL_CKPT = Path("outputs/unet2d_full_c80_b32_flow_dice/checkpoints/best_model.pt")
INPUT_MODE = "full"
BASE_CHANNELS = 32
THRESHOLD = 0.5
CROP_SIZE = 80
FULL_SIZE = 240
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ROI txt indexing
ROI_ONE_BASED = False

# Background image frame for overlay panels
PHASE_FRAME_INDEX = 16

# Which novice session to show in the middle panel
NOVICE_SHOW_SESSION = 1

# Series auto-pick
# Strong preference: the actual cervical CSF series used in all 3 patients
DESC_REGEX_STRICT = re.compile(r"^\s*PCV\s*5\s*CervLCS\s*$", re.IGNORECASE)

# Fallback only if strict match does not exist; still exclude aqueduct and vascular series later
DESC_REGEX_FALLBACK = re.compile(r"cervlcs", re.IGNORECASE)

INTERPOLATE_N = 3201

# Debug
DEBUG_DICOM_SELECTION = True

# Plot styling
DPI = 220
FIGSIZE_OVERLAY = (4.1, 4.1)
FIGSIZE_FLOW = (5.1, 3.2)


def _ds_brief(ds, idx=None) -> str:
    fn = getattr(ds, "filename", "")
    inst = getattr(ds, "InstanceNumber", None)
    tt = getattr(ds, "TriggerTime", None)
    sd = getattr(ds, "SeriesDescription", "")
    sn = getattr(ds, "SeriesNumber", None)
    sop = getattr(ds, "SOPInstanceUID", "")
    sop_short = str(sop)[-12:] if sop else ""
    prefix = f"[{idx:02d}] " if idx is not None else ""
    return (
        f"{prefix}"
        f"file={fn} | SeriesNumber={sn} | SeriesDescription={sd} | "
        f"InstanceNumber={inst} | TriggerTime={tt} | SOP(last12)={sop_short}"
    )
# ------------------------------------------------------------------------------
# Imports from your repo
# ------------------------------------------------------------------------------
from preprocess_dicom import (
    make_mask_from_txt,
    inventory_dicomdir,
    load_series_from_dicomdir,
    embed_mask_in_image,
    _norm01 as _norm01_img,
)

from csf_flow import (
    compute_flow_and_stroke_volume,
    build_exclusion_mask,
    compute_reference_mask,
)

try:
    from src.models.unet2d import UNet2D
except Exception:
    from unet2d import UNet2D

try:
    from src.utils.misc import load_ckpt
except Exception:
    load_ckpt = None


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------
@dataclass
class PatientConfig:
    token: str
    folder_name: str
    display_name: str
    dicomdir: Path
    patient_upper: str


@dataclass
class BiomarkerRow:
    patient: str
    role: str
    rater_name: str
    session: str
    sample: str
    roi_area_mm2: float
    flow_amp_mm3_per_s: float
    stroke_vol_mm3: float
    mask_area_px: int
    centroid_x_px: float
    centroid_y_px: float


# ------------------------------------------------------------------------------
# Temporal helpers (reused from your overlay script)
# ------------------------------------------------------------------------------
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
        return (
            float(tt) if tt is not None else float("inf"),
            int(inst) if inst is not None else 1_000_000,
        )
    return sorted(datasets, key=key)


def _align_phase_to_mag(phase_ds: List, mag_ds: List) -> List:
    mag_times = [(_extract_trigger_time(ds), i) for i, ds in enumerate(mag_ds)]
    ph_times = [(_extract_trigger_time(ds), i) for i, ds in enumerate(phase_ds)]
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
                uniq.append(i)
                seen.add(i)
        if len(uniq) < 32:
            remaining = [i for _, i in ph_times if i not in seen]
            uniq.extend(remaining[: 32 - len(uniq)])
        return [phase_ds[i] for i in uniq[:32]]

    even_idx = list(range(0, len(phase_ds), 2))[:32]
    odd_idx = list(range(1, len(phase_ds), 2))[:32]

    def mean_std(idxs):
        vals = []
        for i in idxs:
            arr = phase_ds[i].pixel_array.astype(np.float32)
            vals.append(float(arr.std()))
        return float(np.mean(vals)) if vals else 0.0

    idxs = even_idx if mean_std(even_idx) >= mean_std(odd_idx) else odd_idx
    return [phase_ds[i] for i in idxs]


def _series_to_phase_mag_01(split, patient_name: str = "", series_number: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    phase_all = _sort_temporally(split.phase)
    mag_all = _sort_temporally(split.mag)
    n_phase, n_mag = len(phase_all), len(mag_all)

    if DEBUG_DICOM_SELECTION:
        print("\n" + "=" * 100)
        print(f"[DICOM DEBUG] patient={patient_name} | chosen_series={series_number}")
        print(f"[DICOM DEBUG] available counts -> PHASE={n_phase}, MAG={n_mag}")
        if len(phase_all) > 0:
            print(f"[DICOM DEBUG] PHASE first sorted: {_ds_brief(phase_all[0])}")
        if len(mag_all) > 0:
            print(f"[DICOM DEBUG] MAG   first sorted: {_ds_brief(mag_all[0])}")

    if n_mag < 32 or n_phase < 32:
        raise RuntimeError(f"Need at least 32 PHASE + 32 MAG, got {n_phase}/{n_mag}")

    if n_phase == 64 and n_mag >= 32:
        mag_sel = mag_all[:32]
        phase_sel = _align_phase_to_mag(phase_all, mag_sel)
        selection_mode = "64->32 aligned by TriggerTime/heuristic"
    else:
        phase_sel = phase_all[:32]
        mag_sel = mag_all[:32]
        selection_mode = "first 32 phase + first 32 mag after temporal sorting"

    if DEBUG_DICOM_SELECTION:
        print(f"[DICOM DEBUG] selection_mode={selection_mode}")
        print("[DICOM DEBUG] ---- SELECTED MAG FILES ----")
        for i, ds in enumerate(mag_sel):
            print(_ds_brief(ds, i))
        print("[DICOM DEBUG] ---- SELECTED PHASE FILES ----")
        for i, ds in enumerate(phase_sel):
            print(_ds_brief(ds, i))
        print("=" * 100 + "\n")

    phase_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in phase_sel], axis=0)
    mag_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in mag_sel], axis=0)
    return phase_01, mag_01


# ------------------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------------------
def _normalize_patient_upper(folder_name: str) -> str:
    return folder_name.upper()


def _choose_series_number(dicomdir: Path) -> int:
    inv = inventory_dicomdir(dicomdir)

    if DEBUG_DICOM_SELECTION:
        print(f"\n[INVENTORY] {dicomdir}")
        for s in inv:
            print(
                f"  series_number={s.get('series_number')} | "
                f"desc={s.get('series_description')} | "
                f"n_phase={s.get('n_phase')} | n_mag={s.get('n_mag')} | "
                f"series_uid={s.get('series_uid')}"
            )

    cands = [s for s in inv if int(s["n_phase"]) >= 32 and int(s["n_mag"]) >= 32]
    if not cands:
        raise RuntimeError(f"No valid 32/32 phase-mag series found in {dicomdir}")

    def desc_of(s):
        return str(s.get("series_description") or "").strip()

    # 1) Exact preferred series: PCV 5CervLCS
    strict_hits = [s for s in cands if DESC_REGEX_STRICT.match(desc_of(s))]
    if strict_hits:
        chosen = sorted(
            strict_hits,
            key=lambda s: int(s["series_number"]) if s["series_number"] is not None else 999999
        )[0]
        if DEBUG_DICOM_SELECTION:
            print(f"[SERIES SELECT] strict PCV 5CervLCS -> {chosen['series_number']} | {desc_of(chosen)}")
        return int(chosen["series_number"])

    # 2) Narrow fallback: anything with CervLCS, but explicitly reject aqueduct and vascular
    fallback_hits = []
    for s in cands:
        d = desc_of(s).lower()
        if DESC_REGEX_FALLBACK.search(d) and ("aqueduc" not in d and "acqueduc" not in d and "vasc" not in d):
            fallback_hits.append(s)

    if fallback_hits:
        chosen = sorted(
            fallback_hits,
            key=lambda s: int(s["series_number"]) if s["series_number"] is not None else 999999
        )[0]
        if DEBUG_DICOM_SELECTION:
            print(f"[SERIES SELECT] fallback CervLCS -> {chosen['series_number']} | {desc_of(chosen)}")
        return int(chosen["series_number"])

    # 3) Fail explicitly instead of silently picking the wrong series
    raise RuntimeError(
        f"Could not find a suitable cervical CSF series in {dicomdir}. "
        f"Expected something like 'PCV 5CervLCS'."
    )


def _find_single_txt(pattern: str) -> Path:
    hits = sorted(MASK_ROOT.glob(pattern))
    if len(hits) == 0:
        raise FileNotFoundError(f"No mask found for pattern: {pattern}")
    if len(hits) > 1:
        # keep deterministic choice, but be explicit
        print(f"[WARN] Multiple masks found for pattern {pattern}; using first: {hits[0]}")
    return hits[0]


def _discover_mask_paths(patient_upper: str) -> Dict[str, Dict[str, Path]]:
    raters = {
        "Exp1": ("Kimi1", "Kimi2"),
        "Exp2": ("Leo1", "Leo2"),
        "Exp3": ("Olivier1", "Olivier2"),
        "Novice": ("Eder1", "Eder2"),
    }
    out = {}
    for role, sessions in raters.items():
        out[role] = {}
        for sess_idx, sess_folder in enumerate(sessions, start=1):
            patt = f"{sess_folder}/{patient_upper}/*/Segment/aqueduc.txt"
            out[role][f"sess{sess_idx}"] = _find_single_txt(patt)
    return out


def _load_and_embed_mask(txt_path: Path, shape_hw: Tuple[int, int], one_based=False) -> np.ndarray:
    m_small, cx, cy = make_mask_from_txt(txt_path, one_based=bool(one_based))
    H, W = shape_hw
    m_full = embed_mask_in_image((H, W), m_small, cx, cy).astype(np.uint8)
    return (m_full > 0).astype(np.uint8)


def _center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    assert arr.ndim == 3
    T, H, W = arr.shape
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    return arr[:, y0:y0 + crop, x0:x0 + crop]


def _pad_crop_to_size(mask_crop: np.ndarray, H: int, W: int, crop: int) -> np.ndarray:
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    out = np.zeros((H, W), dtype=mask_crop.dtype)
    out[y0:y0 + crop, x0:x0 + crop] = mask_crop
    return out


def _zscore_per_sample(x: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + eps)


def _build_input_from_mode_crop_raw(phase_crop_raw: np.ndarray, mag_crop_raw: np.ndarray, input_mode: str = "full") -> np.ndarray:
    mode = (input_mode or "full").lower()

    def _fit32(x: np.ndarray) -> np.ndarray:
        T, H, W = x.shape
        out = np.zeros((32, H, W), dtype=x.dtype)
        t = min(T, 32)
        out[:t] = x[:t]
        return out

    phase = _fit32(phase_crop_raw)
    mag = _fit32(mag_crop_raw)
    if mode != "full":
        raise ValueError("This script currently expects INPUT_MODE='full'.")
    x = np.concatenate([phase, mag], axis=0)
    x = _zscore_per_sample(x).astype(np.float32)
    return x


def _predict_mask_from_crop_input(x_chw: np.ndarray) -> np.ndarray:
    device = torch.device(DEVICE_STR)
    model = UNet2D(in_channels=64, out_channels=1, base_channels=int(BASE_CHANNELS)).to(device)

    if not MODEL_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CKPT}")

    if load_ckpt is not None:
        ckpt = load_ckpt(str(MODEL_CKPT), map_location=device)
        state_dict = ckpt.get("state_dict") or ckpt.get("model")
        if state_dict is None:
            raise KeyError(f"Checkpoint missing state_dict/model keys: {list(ckpt.keys())}")
        model.load_state_dict(state_dict)
    else:
        sd = torch.load(str(MODEL_CKPT), map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x_chw[None, ...].astype(np.float32)).to(device)
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return (probs >= float(THRESHOLD)).astype(np.uint8)


def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def _draw_contours(ax, mask: np.ndarray, color, ls="-", lw=2.0, alpha=0.95, z=5):
    if mask is None or int(mask.sum()) == 0:
        return
    for c in _find_contours(mask):
        c = c.squeeze()
        if c.ndim != 2 or c.shape[0] < 2:
            continue
        ax.plot(c[:, 0], c[:, 1], color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=z)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    yy, xx = np.nonzero(mask > 0)
    if len(xx) == 0:
        return (float("nan"), float("nan"))
    return float(np.mean(xx)), float(np.mean(yy))


def _safe_dice(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter) / (denom + eps))


def _safe_iou(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + eps))


def _ellipse_from_points(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return {
            "mean_x": float("nan"),
            "mean_y": float("nan"),
            "eigval1": float("nan"),
            "eigval2": float("nan"),
            "axis1_1sd": float("nan"),
            "axis2_1sd": float("nan"),
            "angle_deg": float("nan"),
        }
    if len(x) == 1:
        return {
            "mean_x": float(x[0]),
            "mean_y": float(y[0]),
            "eigval1": 0.0,
            "eigval2": 0.0,
            "axis1_1sd": 0.0,
            "axis2_1sd": 0.0,
            "angle_deg": 0.0,
        }
    P = np.column_stack([x, y])
    mu = P.mean(axis=0)
    C = np.cov(P.T, ddof=1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    vals = np.maximum(vals, 0.0)
    return {
        "mean_x": float(mu[0]),
        "mean_y": float(mu[1]),
        "eigval1": float(vals[0]),
        "eigval2": float(vals[1]),
        "axis1_1sd": float(np.sqrt(vals[0])),
        "axis2_1sd": float(np.sqrt(vals[1])),
        "angle_deg": float(ang),
    }


def _build_metadata_from_dicom(split, phase_01: np.ndarray) -> Dict[str, float]:
    ds0 = _sort_temporally(split.phase)[0]
    pixel_spacing = getattr(ds0, "PixelSpacing", None)
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        pixel_size_mm = float(pixel_spacing[0])
    else:
        pixel_size_mm = 0.8

    venc_candidates = []
    for tag in [
        (0x2001, 0x101A),  # Philips private sometimes
        (0x0018, 0x9197),  # velocity encoding
    ]:
        try:
            elem = ds0.get(tag, None)
            if elem is not None:
                venc_candidates.append(float(elem.value))
        except Exception:
            pass

    # fallback from SeriesDescription if needed
    desc = str(getattr(ds0, "SeriesDescription", "") or "")
    if not venc_candidates:
        m = re.search(r"PCV\s*([0-9]+)", desc, re.IGNORECASE)
        if m:
            venc_candidates.append(float(m.group(1)) * 10.0)  # cm/s → mm/s

    v_enc = float(venc_candidates[0]) if len(venc_candidates) > 0 else 50.0
    trigger_delay = float(_extract_trigger_time(ds0) or 650.0)
    return {
        "v_enc": v_enc,
        "pixel_size": pixel_size_mm,
        "trigger_delay": trigger_delay,
    }


# ------------------------------------------------------------------------------
# Flow helpers
# ------------------------------------------------------------------------------
def _compute_shared_ref_mask(phase_full_m1_1: np.ndarray, mag_full_01: np.ndarray, ref_seed_mask: np.ndarray) -> Tuple[np.ndarray, str]:
    excl = build_exclusion_mask(ref_seed_mask.astype(np.uint8), radius_px=8)
    try:
        ref_mask = compute_reference_mask(
            phase_full_m1_1,
            mag_full_01,
            processed_mask=excl,
            n_ref_pixels=4000,
            min_component_size=50,
            verbose=False,
        )
        return ref_mask.astype(np.uint8), "default"
    except Exception as e:
        print(f"[WARN] default ref-mask failed -> {e}")
        try:
            ref_mask = compute_reference_mask(
                phase_full_m1_1,
                mag_full_01,
                processed_mask=excl,
                n_ref_pixels=2000,
                pct_mag_low=0, pct_mag_high=99,
                pct_mean_vel=99, pct_fft=99,
                pct_std_low=5, pct_std_high=99,
                radial_center=None,
                min_component_size=10,
                verbose=False,
            )
            return ref_mask.astype(np.uint8), "loose"
        except Exception as e2:
            print(f"[WARN] loose ref-mask failed -> {e2}")
            return None, "none"


def _compute_biomarkers_for_mask(
    phase_full_m1_1: np.ndarray,
    mag_full_01: np.ndarray,
    metadata: Dict[str, float],
    mask_full: np.ndarray,
    ref_mask: Optional[np.ndarray],
) -> Dict:
    return compute_flow_and_stroke_volume(
        phase_vol=phase_full_m1_1,
        mask=mask_full.astype(np.uint8),
        metadata=metadata,
        magnitude_vol=mag_full_01,
        ref_mask=ref_mask,
        use_background_correction=(ref_mask is not None),
        interpolate_n=INTERPOLATE_N,
    )


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
def save_panel1_expert_vs_model(phase_bg01: np.ndarray, expert_masks: List[np.ndarray], model_mask: np.ndarray, out_png: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE_OVERLAY)
    ax.imshow(phase_bg01, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    for m in expert_masks:
        _draw_contours(ax, m, color="#7FDBFF", ls="-", lw=1.2, alpha=0.95, z=6)
    _draw_contours(ax, model_mask, color="red", ls="-", lw=2.2, alpha=0.95, z=7)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], color="#7FDBFF", lw=1.5, label="Expert"),
        Line2D([0], [0], color="red", lw=2.2, label="DL model contour"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_panel2_novice_model_consensus(
    phase_bg01: np.ndarray,
    expert_mean_mask: np.ndarray,
    model_mask: np.ndarray,
    novice_mask: np.ndarray,
    out_png: Path,
):
    fig, ax = plt.subplots(figsize=FIGSIZE_OVERLAY)
    ax.imshow(phase_bg01, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    _draw_contours(ax, expert_mean_mask, color="#00FFFF", ls="-", lw=2.0, alpha=0.95, z=6)
    _draw_contours(ax, model_mask, color="#FF00FF", ls="-", lw=2.0, alpha=0.95, z=7)
    _draw_contours(ax, novice_mask, color="#3CB371", ls="--", lw=1.8, alpha=0.95, z=6)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], color="#00FFFF", lw=2.0, label="Expert consensus"),
        Line2D([0], [0], color="#FF00FF", lw=2.0, label="DL model contour"),
        Line2D([0], [0], color="#3CB371", lw=1.8, ls="--", label="Novice contour"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_panel3_flow(
    patient_token: str,
    flow_dict: Dict[str, Dict],
    out_png: Path,
):
    """
    flow_dict keys expected:
      expert_rows: list[dict], each with flow_corr/t_interp
      novice_rows: list[dict], each with flow_corr/t_interp
      model_row: dict
      summaries: dict with text values
    """
    expert_rows = flow_dict["expert_rows"]
    novice_rows = flow_dict["novice_rows"]
    model_row = flow_dict["model_row"]

    fig, ax = plt.subplots(figsize=FIGSIZE_FLOW)
    ax.grid(True, alpha=0.25)

    # expert mean ± 1 sd
    ex_curves = np.stack([r["flow_interp"] for r in expert_rows], axis=0)
    t = expert_rows[0]["t_interp"]
    ex_mu = np.mean(ex_curves, axis=0)
    ex_sd = np.std(ex_curves, axis=0, ddof=1) if ex_curves.shape[0] > 1 else np.zeros_like(ex_mu)
    ax.plot(t * 1000.0, ex_mu, color="#444444", lw=2.0, label="Expert mean")
    ax.fill_between(t * 1000.0, ex_mu - ex_sd, ex_mu + ex_sd, color="#888888", alpha=0.25, linewidth=0)

    # novice mean ± 1 sd over 2 sessions
    nv_curves = np.stack([r["flow_interp"] for r in novice_rows], axis=0)
    nv_mu = np.mean(nv_curves, axis=0)
    nv_sd = np.std(nv_curves, axis=0, ddof=1) if nv_curves.shape[0] > 1 else np.zeros_like(nv_mu)
    ax.plot(t * 1000.0, nv_mu, color="#2E8B57", lw=1.8, ls="--", label="Novice")
    if novice_rows and nv_curves.shape[0] > 1:
        ax.fill_between(t * 1000.0, nv_mu - nv_sd, nv_mu + nv_sd, color="#2E8B57", alpha=0.12, linewidth=0)

    # model
    ax.plot(model_row["t_interp"] * 1000.0, model_row["flow_interp"], color="#C33", lw=1.8, label="DL model")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Background-corrected flow (mm$^3$/s)")
    #ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------------------
def process_one_patient(cfg: PatientConfig):
    print(f"\n[INFO] Processing {cfg.display_name} ({cfg.token})")
    patient_out = OUTPUT_ROOT / cfg.token
    patient_out.mkdir(parents=True, exist_ok=True)

    # DICOM
    series_number = _choose_series_number(cfg.dicomdir)
    split = load_series_from_dicomdir(cfg.dicomdir, int(series_number))
    series_desc = getattr(_sort_temporally(split.mag)[0], "SeriesDescription", "") if len(split.mag) else ""
    print(f"[INFO] {cfg.token}: chosen series_number = {series_number} | description = {series_desc}")
    phase_01, mag_01 = _series_to_phase_mag_01(
        split,
        patient_name=cfg.display_name,
        series_number=series_number,
    )  # (32,H,W)
    T, H, W = phase_01.shape
    phase_full_m1_1 = phase_01 * 2.0 - 1.0
    metadata = _build_metadata_from_dicom(split, phase_01)
    phase_bg = phase_01[int(np.clip(PHASE_FRAME_INDEX, 0, T - 1))]

    # Discover mask paths
    mask_paths = _discover_mask_paths(cfg.patient_upper)
    with open(patient_out / "mask_paths.json", "w") as f:
        json.dump(
            {k: {kk: str(vv) for kk, vv in d.items()} for k, d in mask_paths.items()},
            f,
            indent=2,
        )

    # Load masks
    expert_masks = []
    expert_names = []
    for exp_name in ["Exp1", "Exp2", "Exp3"]:
        for sess_key in ["sess1", "sess2"]:
            txt = mask_paths[exp_name][sess_key]
            m = _load_and_embed_mask(txt, (H, W), one_based=ROI_ONE_BASED)
            expert_masks.append(m)
            expert_names.append(f"{exp_name}_{sess_key}")

    novice_masks = []
    for sess_key in ["sess1", "sess2"]:
        txt = mask_paths["Novice"][sess_key]
        m = _load_and_embed_mask(txt, (H, W), one_based=ROI_ONE_BASED)
        novice_masks.append(m)

    # Expert average / strict consensus / overlap stats
    stack_ex = np.stack(expert_masks, axis=0).astype(np.uint8)
    expert_overlap_count = stack_ex.sum(axis=0).astype(np.uint8)
    expert_overlap_fraction = expert_overlap_count.astype(np.float32) / float(stack_ex.shape[0])
    expert_mean_mask = (expert_overlap_fraction >= 0.5).astype(np.uint8)
    expert_and_mask = (expert_overlap_count == stack_ex.shape[0]).astype(np.uint8)
    expert_std_map = np.std(stack_ex.astype(np.float32), axis=0, ddof=1)

    np.save(patient_out / "expert_overlap_count.npy", expert_overlap_count)
    np.save(patient_out / "expert_overlap_fraction.npy", expert_overlap_fraction)
    np.save(patient_out / "expert_std_map.npy", expert_std_map)

    # Model
    phase_crop = _center_crop(phase_01, CROP_SIZE)
    mag_crop = _center_crop(mag_01, CROP_SIZE)
    x_crop = _build_input_from_mode_crop_raw(phase_crop, mag_crop, input_mode=INPUT_MODE)
    t0 = time.perf_counter()
    model_mask_crop = _predict_mask_from_crop_input(x_crop)
    infer_s = time.perf_counter() - t0
    model_mask = _pad_crop_to_size(model_mask_crop, H=H, W=W, crop=CROP_SIZE).astype(np.uint8)

    np.save(patient_out / "model_mask.npy", model_mask)

    # Shared ref mask from expert mean mask
    ref_mask, ref_mode = _compute_shared_ref_mask(phase_full_m1_1, mag_01, expert_mean_mask)
    if ref_mask is not None:
        np.save(patient_out / "ref_mask.npy", ref_mask.astype(np.uint8))

    # Compute biomarkers for all masks with shared reference
    biomarker_rows: List[BiomarkerRow] = []
    flow_rows_long = []

    def add_one(role: str, rater_name: str, session: str, sample: str, mask: np.ndarray, res: Dict):
        cx, cy = _mask_centroid(mask)
        biomarker_rows.append(
            BiomarkerRow(
                patient=cfg.token,
                role=role,
                rater_name=rater_name,
                session=session,
                sample=sample,
                roi_area_mm2=float(mask.sum() * (metadata["pixel_size"] ** 2)),
                flow_amp_mm3_per_s=float(res["flow_range"]),
                stroke_vol_mm3=float(res["stroke_vol"]),
                mask_area_px=int(mask.sum()),
                centroid_x_px=float(cx),
                centroid_y_px=float(cy),
            )
        )
        curve_df = pd.DataFrame({
            "patient": cfg.token,
            "sample": sample,
            "role": role,
            "rater_name": rater_name,
            "session": session,
            "t_s": res["t"],
            "flow_corr_mm3_per_s": res["flow_corr"],
            "flow_uncorr_mm3_per_s": res["flow"],
        })
        flow_rows_long.append(curve_df)

    expert_res_list = []
    for name, mask in zip(expert_names, expert_masks):
        res = _compute_biomarkers_for_mask(phase_full_m1_1, mag_01, metadata, mask, ref_mask)
        add_one("Expert", name.split("_")[0], name.split("_")[1], f"{name}-{cfg.token}", mask, res)
        expert_res_list.append(res)

    novice_res_list = []
    for i, mask in enumerate(novice_masks, start=1):
        res = _compute_biomarkers_for_mask(phase_full_m1_1, mag_01, metadata, mask, ref_mask)
        add_one("Novice", "Eder", f"sess{i}", f"Eder-{i}-{cfg.token}", mask, res)
        novice_res_list.append(res)

    model_res = _compute_biomarkers_for_mask(phase_full_m1_1, mag_01, metadata, model_mask, ref_mask)
    add_one("DL", "model", "model", f"model-{cfg.token}", model_mask, model_res)

    # Panel images
    save_panel1_expert_vs_model(
        phase_bg01=phase_bg,
        expert_masks=expert_masks,
        model_mask=model_mask,
        out_png=patient_out / "01_expert_variance_vs_model.png",
    )
    save_panel2_novice_model_consensus(
        phase_bg01=phase_bg,
        expert_mean_mask=expert_mean_mask,
        model_mask=model_mask,
        novice_mask=novice_masks[max(0, min(NOVICE_SHOW_SESSION - 1, 1))],
        out_png=patient_out / "02_novice_model_vs_expert_consensus.png",
    )

    # Interpolated curves for panel 3
    flow_dict = {
        "expert_rows": expert_res_list,
        "novice_rows": novice_res_list,
        "model_row": model_res,
    }
    save_panel3_flow(
        patient_token=cfg.token,
        flow_dict=flow_dict,
        out_png=patient_out / "03_corrected_flow_waveforms.png",
    )

    # Difficulty / biomarker space stats
    # 1) Geometry-based difficulty summaries from six expert masks
    areas_px = np.array([int(m.sum()) for m in expert_masks], dtype=float)
    centroids = np.array([_mask_centroid(m) for m in expert_masks], dtype=float)

    pair_rows = []
    for i in range(len(expert_masks)):
        for j in range(i + 1, len(expert_masks)):
            pair_rows.append({
                "patient": cfg.token,
                "i": expert_names[i],
                "j": expert_names[j],
                "dice": _safe_dice(expert_masks[i], expert_masks[j]),
                "iou": _safe_iou(expert_masks[i], expert_masks[j]),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(patient_out / "expert_pairwise_overlap.csv", index=False)

    # 2) Biomarker-space expert ellipses for manual panel editing
    bio_df = pd.DataFrame([vars(r) for r in biomarker_rows])
    ex_bio = bio_df[bio_df["role"] == "Expert"].copy()

    ellipse_specs = []
    for xcol, ycol, name in [
        ("roi_area_mm2", "stroke_vol_mm3", "roi_vs_sv"),
        ("roi_area_mm2", "flow_amp_mm3_per_s", "roi_vs_peak_flow"),
        ("stroke_vol_mm3", "flow_amp_mm3_per_s", "sv_vs_peak_flow"),
    ]:
        est = _ellipse_from_points(ex_bio[xcol].values, ex_bio[ycol].values)
        est.update({"patient": cfg.token, "space": name, "x_metric": xcol, "y_metric": ycol})
        ellipse_specs.append(est)

    ellipse_df = pd.DataFrame(ellipse_specs)
    ellipse_df.to_csv(patient_out / "biomarker_space_ellipses.csv", index=False)

    # 3) Difficulty stats
    difficulty = {
        "patient": cfg.token,
        "display_name": cfg.display_name,
        "series_number": int(series_number),
        "ref_mode": ref_mode,
        "model_inference_seconds": float(infer_s),
        "expert_n_masks": int(stack_ex.shape[0]),
        "expert_area_mean_px": float(np.mean(areas_px)),
        "expert_area_sd_px": float(np.std(areas_px, ddof=1)),
        "expert_area_mean_mm2": float(np.mean(areas_px) * metadata["pixel_size"] ** 2),
        "expert_area_sd_mm2": float(np.std(areas_px, ddof=1) * metadata["pixel_size"] ** 2),
        "expert_centroid_mean_x_px": float(np.nanmean(centroids[:, 0])),
        "expert_centroid_mean_y_px": float(np.nanmean(centroids[:, 1])),
        "expert_centroid_sd_x_px": float(np.nanstd(centroids[:, 0], ddof=1)),
        "expert_centroid_sd_y_px": float(np.nanstd(centroids[:, 1], ddof=1)),
        "expert_pairwise_dice_mean": float(pair_df["dice"].mean()),
        "expert_pairwise_dice_sd": float(pair_df["dice"].std(ddof=1)),
        "expert_pairwise_iou_mean": float(pair_df["iou"].mean()),
        "expert_pairwise_iou_sd": float(pair_df["iou"].std(ddof=1)),
        "expert_overlap_fraction_mean_on_union": float(expert_overlap_fraction[expert_overlap_count > 0].mean()),
        "expert_std_map_mean_on_union": float(expert_std_map[expert_overlap_count > 0].mean()),
        "expert_std_map_max": float(expert_std_map.max()),
        "expert_consensus_and_area_px": int(expert_and_mask.sum()),
        "expert_meanmask_area_px": int(expert_mean_mask.sum()),
        "expert_flow_amp_mean_mm3_per_s": float(ex_bio["flow_amp_mm3_per_s"].mean()),
        "expert_flow_amp_sd_mm3_per_s": float(ex_bio["flow_amp_mm3_per_s"].std(ddof=1)),
        "expert_sv_mean_mm3": float(ex_bio["stroke_vol_mm3"].mean()),
        "expert_sv_sd_mm3": float(ex_bio["stroke_vol_mm3"].std(ddof=1)),
    }
    with open(patient_out / "difficulty_stats.json", "w") as f:
        json.dump(difficulty, f, indent=2)

    manifest = pd.DataFrame([{
        "patient": cfg.token,
        "display_name": cfg.display_name,
        "folder_name": cfg.folder_name,
        "dicomdir": str(cfg.dicomdir),
        "series_number": int(series_number),
        "panel1": str(patient_out / "01_expert_variance_vs_model.png"),
        "panel2": str(patient_out / "02_novice_model_vs_expert_consensus.png"),
        "panel3": str(patient_out / "03_corrected_flow_waveforms.png"),
        "ref_mask": str(patient_out / "ref_mask.npy") if ref_mask is not None else "",
        "ref_mode": ref_mode,
    }])

    return {
        "biomarkers_df": pd.DataFrame([vars(r) for r in biomarker_rows]),
        "flow_long_df": pd.concat(flow_rows_long, ignore_index=True),
        "difficulty_df": pd.DataFrame([difficulty]),
        "ellipse_df": ellipse_df,
        "manifest_df": manifest,
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patient_cfgs = []
    for token, folder in PATIENTS:
        patient_cfgs.append(PatientConfig(
            token=token,
            folder_name=folder,
            display_name=folder,
            dicomdir=DATA_ROOT / folder / "DICOMDIR",
            patient_upper=_normalize_patient_upper(folder),
        ))

    all_bio = []
    all_flow = []
    all_diff = []
    all_ellipse = []
    all_manifest = []

    for cfg in patient_cfgs:
        if not cfg.dicomdir.exists():
            raise FileNotFoundError(f"DICOMDIR not found: {cfg.dicomdir}")
        out = process_one_patient(cfg)
        all_bio.append(out["biomarkers_df"])
        all_flow.append(out["flow_long_df"])
        all_diff.append(out["difficulty_df"])
        all_ellipse.append(out["ellipse_df"])
        all_manifest.append(out["manifest_df"])

    bio_df = pd.concat(all_bio, ignore_index=True)
    flow_df = pd.concat(all_flow, ignore_index=True)
    diff_df = pd.concat(all_diff, ignore_index=True)
    ellipse_df = pd.concat(all_ellipse, ignore_index=True)
    manifest_df = pd.concat(all_manifest, ignore_index=True)

    # Long-form outputs
    bio_df.to_csv(OUTPUT_ROOT / "biomarkers_long.csv", index=False)
    flow_df.to_csv(OUTPUT_ROOT / "flow_curves_long.csv", index=False)
    diff_df.to_csv(OUTPUT_ROOT / "difficulty_stats.csv", index=False)
    ellipse_df.to_csv(OUTPUT_ROOT / "biomarker_space_ellipses.csv", index=False)
    manifest_df.to_csv(OUTPUT_ROOT / "patient_manifest.csv", index=False)

    # Summary by patient x role
    summary = (
        bio_df.groupby(["patient", "role"])[["roi_area_mm2", "flow_amp_mm3_per_s", "stroke_vol_mm3", "mask_area_px"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_ROOT / "biomarkers_summary.csv", index=False)

    # Useful points for manual scatter panels later
    points = bio_df[[
        "patient", "role", "rater_name", "session", "sample",
        "roi_area_mm2", "flow_amp_mm3_per_s", "stroke_vol_mm3"
    ]].copy()
    points.to_csv(OUTPUT_ROOT / "biomarker_space_points.csv", index=False)

    print("\n[OK] Done.")
    print(f"[OK] Root output: {OUTPUT_ROOT.resolve()}")
    print("[OK] 9 images saved as:")
    for p in manifest_df.itertuples(index=False):
        print(f"  - {p.patient}:")
        print(f"      {p.panel1}")
        print(f"      {p.panel2}")
        print(f"      {p.panel3}")


if __name__ == "__main__":
    main()
