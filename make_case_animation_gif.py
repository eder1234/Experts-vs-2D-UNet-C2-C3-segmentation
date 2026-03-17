#!/usr/bin/env python3
"""
make_case_animation_gif.py

Generate an anonymous GIF for one selected patient showing, from left to right:
  1) Phase image with expert mean-mask contour + DL contour
  2) Magnitude image with expert mean-mask contour + DL contour
  3) Background-corrected flow curves for BOTH expert mean-mask and DL mask,
     with the current frame marker

Frames:
  - one GIF frame per cardiac frame (1..32)

Patient selection:
  - via terminal argument --patient {PATIENT-ID,PATIENT-ID,PATIENT-ID}
  - default = PATIENT-ID

Anonymous labeling:
  - PATIENT-ID -> Easy case
  - PATIENT-ID -> Medium case
  - PATIENT-ID -> Hard case
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_ROOT = Path("/media/rodriguez/easystore/patients_abstract")
MASK_ROOT = Path("/media/rodriguez/easystore/Traitement_abstract_final")
OUTPUT_ROOT = Path("outputs/case_gifs")

MODEL_CKPT = Path("outputs/unet2d_full_c80_b32_flow_dice/checkpoints/best_model.pt")
BASE_CHANNELS = 32
THRESHOLD = 0.5
CROP_SIZE = 80
ROI_ONE_BASED = False
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

DESC_REGEX_STRICT = re.compile(r"^\s*PCV\s*5\s*CervLCS\s*$", re.IGNORECASE)
DESC_REGEX_FALLBACK = re.compile(r"cervlcs", re.IGNORECASE)

PATIENT_MAP = {
    "PATIENT-ID": ("PATIENT-ID ", "Easy case"),
    "PATIENT-ID": ("PATIENT-ID ", "Medium case"),
    "PATIENT-ID": ("PATIENT-ID ", "Hard case"),
}

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

        def rkey(t):
            return int(round(t))

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


def _series_to_phase_mag_01(split) -> Tuple[np.ndarray, np.ndarray]:
    phase_all = _sort_temporally(split.phase)
    mag_all = _sort_temporally(split.mag)
    n_phase, n_mag = len(phase_all), len(mag_all)
    if n_mag < 32 or n_phase < 32:
        raise RuntimeError(f"Need at least 32 PHASE + 32 MAG, got {n_phase}/{n_mag}")

    if n_phase == 64 and n_mag >= 32:
        mag_sel = mag_all[:32]
        phase_sel = _align_phase_to_mag(phase_all, mag_sel)
    else:
        phase_sel = phase_all[:32]
        mag_sel = mag_all[:32]

    phase_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in phase_sel], axis=0)
    mag_01 = np.stack([_norm01_img(ds.pixel_array.astype(np.float32)) for ds in mag_sel], axis=0)
    return phase_01, mag_01


def _find_single_txt(pattern: str) -> Path:
    hits = sorted(MASK_ROOT.glob(pattern))
    if len(hits) == 0:
        raise FileNotFoundError(f"No mask found for pattern: {pattern}")
    return hits[0]


def _discover_mask_paths(patient_upper: str) -> Dict[str, Dict[str, Path]]:
    raters = {
        "Exp1": ("Kimi1", "Kimi2"),
        "Exp2": ("Leo1", "Leo2"),
        "Exp3": ("Olivier1", "Olivier2"),
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


def _build_input_from_mode_crop_raw(phase_crop_raw: np.ndarray, mag_crop_raw: np.ndarray) -> np.ndarray:
    def _fit32(x: np.ndarray) -> np.ndarray:
        T, H, W = x.shape
        out = np.zeros((32, H, W), dtype=x.dtype)
        out[:min(T, 32)] = x[:min(T, 32)]
        return out

    phase = _fit32(phase_crop_raw)
    mag = _fit32(mag_crop_raw)
    x = np.concatenate([phase, mag], axis=0)
    return _zscore_per_sample(x).astype(np.float32)


def _predict_mask_from_crop_input(x_chw: np.ndarray) -> np.ndarray:
    device = torch.device(DEVICE_STR)
    model = UNet2D(in_channels=64, out_channels=1, base_channels=int(BASE_CHANNELS)).to(device)

    if load_ckpt is not None:
        ckpt = load_ckpt(str(MODEL_CKPT), map_location=device)
        state_dict = ckpt.get("state_dict") or ckpt.get("model")
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


def _choose_series_number(dicomdir: Path) -> int:
    inv = inventory_dicomdir(dicomdir)
    cands = [s for s in inv if int(s["n_phase"]) >= 32 and int(s["n_mag"]) >= 32]
    if not cands:
        raise RuntimeError(f"No valid 32/32 phase-mag series found in {dicomdir}")

    def desc_of(s):
        return str(s.get("series_description") or "").strip()

    strict_hits = [s for s in cands if DESC_REGEX_STRICT.match(desc_of(s))]
    if strict_hits:
        return int(sorted(strict_hits, key=lambda s: int(s["series_number"]))[0]["series_number"])

    fallback_hits = []
    for s in cands:
        d = desc_of(s).lower()
        if DESC_REGEX_FALLBACK.search(d) and ("aqueduc" not in d and "acqueduc" not in d and "vasc" not in d):
            fallback_hits.append(s)

    if fallback_hits:
        return int(sorted(fallback_hits, key=lambda s: int(s["series_number"]))[0]["series_number"])

    raise RuntimeError(f"Could not find suitable CervLCS series in {dicomdir}")


def _build_metadata_from_dicom(split) -> Dict[str, float]:
    ds0 = _sort_temporally(split.phase)[0]
    pixel_spacing = getattr(ds0, "PixelSpacing", None)
    pixel_size_mm = float(pixel_spacing[0]) if pixel_spacing is not None and len(pixel_spacing) >= 2 else 0.8

    venc_candidates = []
    for tag in [(0x2001, 0x101A), (0x0018, 0x9197)]:
        try:
            elem = ds0.get(tag, None)
            if elem is not None:
                venc_candidates.append(float(elem.value))
        except Exception:
            pass

    desc = str(getattr(ds0, "SeriesDescription", "") or "")
    if not venc_candidates:
        m = re.search(r"PCV\s*([0-9]+)", desc, re.IGNORECASE)
        if m:
            venc_candidates.append(float(m.group(1)) * 10.0)

    v_enc = float(venc_candidates[0]) if venc_candidates else 50.0
    trigger_delay = float(_extract_trigger_time(ds0) or 650.0)
    return {"v_enc": v_enc, "pixel_size": pixel_size_mm, "trigger_delay": trigger_delay}


def _compute_shared_ref_mask(phase_full_m1_1: np.ndarray, mag_full_01: np.ndarray, ref_seed_mask: np.ndarray):
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
    except Exception:
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


def _render_frame(
    phase_img: np.ndarray,
    mag_img: np.ndarray,
    expert_mean_mask: np.ndarray,
    model_mask: np.ndarray,
    flow_t: np.ndarray,
    flow_expert: np.ndarray,
    flow_dl: np.ndarray,
    current_frame_idx: int,
    case_label: str,
    out_tmp: Path,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    ax1, ax2, ax3 = axes

    # Phase
    ax1.imshow(phase_img, cmap="gray", interpolation="nearest")
    _draw_contours(ax1, expert_mean_mask, color="#00FFFF", ls="-", lw=2.0, z=6)
    _draw_contours(ax1, model_mask, color="#FF00FF", ls="-", lw=2.0, z=7)
    ax1.set_title(f"{case_label} — Phase", fontsize=10)
    ax1.axis("off")

    # Magnitude
    ax2.imshow(mag_img, cmap="gray", interpolation="nearest")
    _draw_contours(ax2, expert_mean_mask, color="#00FFFF", ls="-", lw=2.0, z=6)
    _draw_contours(ax2, model_mask, color="#FF00FF", ls="-", lw=2.0, z=7)
    ax2.set_title("Magnitude", fontsize=10)
    ax2.axis("off")

    # Flow curves: expert + DL
    idx = max(0, min(current_frame_idx, len(flow_t) - 1))
    ax3.plot(flow_t, flow_expert, color="#00A6D6", lw=1.8, label="Expert mean")
    ax3.plot(flow_t, flow_dl, color="#C2185B", lw=1.8, label="DL")

    ax3.scatter([flow_t[idx]], [flow_expert[idx]], s=32, color="#00A6D6", zorder=5)
    ax3.scatter([flow_t[idx]], [flow_dl[idx]], s=32, color="#C2185B", zorder=5)
    ax3.axvline(flow_t[idx], color="red", ls="--", lw=1.0, alpha=0.75, label=f"Frame {idx+1}")

    ax3.set_title("Flow waveform", fontsize=10)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Flow (mm$^3$/s)")
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize=8, framealpha=0.9)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#00FFFF", lw=2.0, label="Expert mean contour"),
        Line2D([0], [0], color="#FF00FF", lw=2.0, label="DL contour"),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    # Reduced spacing between subplots
    fig.subplots_adjust(left=0.015, right=0.995, top=0.90, bottom=0.14, wspace=0.03)
    fig.savefig(out_tmp, dpi=140, bbox_inches="tight", pad_inches=0.21)
    plt.close(fig)
    return imageio.imread(out_tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="PATIENT-ID", choices=["PATIENT-ID", "PATIENT-ID", "PATIENT-ID"])
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    patient_key = args.patient.upper()
    folder_name, case_label = PATIENT_MAP[patient_key]
    dicomdir = DATA_ROOT / folder_name / "DICOMDIR"
    patient_upper = folder_name.upper()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_gif = Path(args.out) if args.out else (OUTPUT_ROOT / f"{patient_key.lower()}_anonymous_case.gif")
    tmp_png = OUTPUT_ROOT / "_tmp_case_frame.png"

    series_number = _choose_series_number(dicomdir)
    split = load_series_from_dicomdir(dicomdir, int(series_number))
    phase_01, mag_01 = _series_to_phase_mag_01(split)
    phase_full_m1_1 = phase_01 * 2.0 - 1.0
    metadata = _build_metadata_from_dicom(split)

    # Expert mean mask
    mask_paths = _discover_mask_paths(patient_upper)
    expert_masks = []
    for exp_name in ["Exp1", "Exp2", "Exp3"]:
        for sess_key in ["sess1", "sess2"]:
            txt = mask_paths[exp_name][sess_key]
            expert_masks.append(_load_and_embed_mask(txt, phase_01.shape[1:], one_based=ROI_ONE_BASED))
    stack_ex = np.stack(expert_masks, axis=0).astype(np.uint8)
    expert_mean_mask = ((stack_ex.mean(axis=0)) >= 0.5).astype(np.uint8)

    # DL mask
    phase_crop = _center_crop(phase_01, CROP_SIZE)
    mag_crop = _center_crop(mag_01, CROP_SIZE)
    x_crop = _build_input_from_mode_crop_raw(phase_crop, mag_crop)
    model_mask_crop = _predict_mask_from_crop_input(x_crop)
    model_mask = _pad_crop_to_size(model_mask_crop, H=phase_01.shape[1], W=phase_01.shape[2], crop=CROP_SIZE).astype(np.uint8)

    # Shared ref mask
    ref_mask, ref_mode = _compute_shared_ref_mask(phase_full_m1_1, mag_01, expert_mean_mask)

    # Flow for expert mean mask
    expert_res = compute_flow_and_stroke_volume(
        phase_vol=phase_full_m1_1,
        mask=expert_mean_mask.astype(np.uint8),
        metadata=metadata,
        magnitude_vol=mag_01,
        ref_mask=ref_mask,
        use_background_correction=True,
        interpolate_n=3201,
    )

    # Flow for DL mask
    model_res = compute_flow_and_stroke_volume(
        phase_vol=phase_full_m1_1,
        mask=model_mask.astype(np.uint8),
        metadata=metadata,
        magnitude_vol=mag_01,
        ref_mask=ref_mask,
        use_background_correction=True,
        interpolate_n=3201,
    )

    # Use native 32-frame corrected flows for frame markers
    t_ms = expert_res["t"] * 1000.0
    flow_expert = expert_res["flow_corr"]
    flow_dl = model_res["flow_corr"]

    frames = []
    for i in range(32):
        img = _render_frame(
            phase_img=phase_01[i],
            mag_img=mag_01[i],
            expert_mean_mask=expert_mean_mask,
            model_mask=model_mask,
            flow_t=t_ms,
            flow_expert=flow_expert,
            flow_dl=flow_dl,
            current_frame_idx=i,
            case_label=case_label,
            out_tmp=tmp_png,
        )
        frames.append(img)

    imageio.mimsave(out_gif, frames, duration=1.0 / max(args.fps, 0.1), loop=0)

    if tmp_png.exists():
        tmp_png.unlink()

    meta = {
        "patient_key": patient_key,
        "anonymous_label": case_label,
        "folder_name": folder_name,
        "series_number": int(series_number),
        "ref_mode": ref_mode,
        "gif_path": str(out_gif),
    }
    with open(out_gif.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved GIF -> {out_gif}")
    print(f"[OK] Metadata  -> {out_gif.with_suffix('.json')}")


if __name__ == "__main__":
    main()
