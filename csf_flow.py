# csf_flow.py
# -----------------------------------------------------------------------------
# Robust background correction defaults + safe exclusion from ROI
# -----------------------------------------------------------------------------

from __future__ import annotations
import re
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
import cv2

from skimage.morphology import binary_dilation, remove_small_objects, binary_opening, disk

__all__ = [
    "compute_flow_and_stroke_volume",
    "pad_to_full",
    "build_exclusion_mask",
    "process_mask",            # kept for backward-compatibility
    "compute_reference_mask",
    "row_to_metadata",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def pad_to_full(mask_crop: np.ndarray, crop_size: int, full_size: int = 240) -> np.ndarray:
    """Pad a square crop mask to the full image centre."""
    pad = (full_size - crop_size) // 2
    full = np.zeros((full_size, full_size), dtype=mask_crop.dtype)
    full[pad : pad + crop_size, pad : pad + crop_size] = mask_crop
    return full


def build_exclusion_mask(roi_mask_full: np.ndarray, radius_px: int = 8) -> np.ndarray:
    """
    Conservative exclusion: return a mask with 1=exclude, 0=allow by dilating the ROI.

    Parameters
    ----------
    roi_mask_full : (H,W) binary (0/1) ROI
    radius_px     : int, dilation radius in pixels (default: 8)

    Returns
    -------
    np.ndarray of shape (H,W), dtype uint8, with 1=exclude, 0=allow
    """
    se = disk(int(max(0, radius_px)))
    excl = binary_dilation(roi_mask_full.astype(bool), se).astype(np.uint8)
    return excl  # 1=exclude, 0=allow


def process_mask(input_mask: np.ndarray, *, legacy: bool = False, radius_px: int = 8) -> np.ndarray:
    """
    Backward-compatible API for the exclusion mask used by BC.
    By default it returns the conservative ROI-dilation exclusion.
    Set legacy=True to use the previous ring-fill + central disk + dilate(10) behavior.

    Returns a binary mask with 1=exclude, 0=allow.
    """
    if not legacy:
        return build_exclusion_mask(input_mask, radius_px=radius_px)

    # ---- LEGACY BEHAVIOR (was causing over-exclusion on your data) ----
    assert input_mask.ndim == 2, "Mask must be 2D"
    H, W = input_mask.shape
    cy, cx = H // 2, W // 2
    center_val = input_mask[cy, cx]

    output_mask = input_mask.copy()

    if center_val == 0:
        # Assume it's a ring – fill the center with 1s
        mask_floodfill = output_mask.copy().astype(np.uint8)
        mask_bordered = cv2.copyMakeBorder(mask_floodfill, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        floodfill_mask = np.zeros((mask_bordered.shape[0] + 2, mask_bordered.shape[1] + 2), np.uint8)
        cv2.floodFill(mask_bordered, floodfill_mask, (cx + 1, cy + 1), 1)
        output_mask = mask_bordered[1:-1, 1:-1]
    else:
        # Not a ring – draw filled circle at center
        output_mask = np.zeros_like(input_mask, dtype=np.uint8)
        cv2.circle(output_mask, (cx, cy), 15, 1, thickness=-1)

    # Dilate the mask by 10 pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # 2*10+1=21
    output_mask = cv2.dilate(output_mask, kernel, iterations=1)

    return output_mask  # still 1=exclude, 0=allow


def percentile_mask(arr, lower_pct, upper_pct):
    lower = np.percentile(arr, lower_pct)
    upper = np.percentile(arr, upper_pct)
    return (arr > lower) & (arr < upper)


def compute_reference_mask(
    phase_full: np.ndarray,
    mag_full: np.ndarray,
    processed_mask: Optional[np.ndarray] = None,
    *,
    n_ref_pixels: int = 2000,
    pct_mag_low: float = 10,
    pct_mag_high: float = 95,
    pct_mean_vel: float = 90,
    pct_fft: float = 90,
    pct_std_low: float = 60,
    pct_std_high: float = 90,
    radial_center: Optional[tuple[int, int]] = None,
    radial_inner_exclusion: int = 0,
    radial_flat: int = 0,
    min_component_size: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build a static-tissue reference pixel mask (H,W) with True/1 at selected pixels.

    Safer defaults:
      • small-component removal threshold is modest (min_component_size=50)
      • exclusion mask expected with 1=exclude, 0=allow

    Raises ValueError if no pixels survive.
    """
    T, H, W = phase_full.shape
    eps = 1e-6

    if processed_mask is None:
        processed_mask = np.zeros((H, W), dtype=np.uint8)
    else:
        if processed_mask.dtype != np.uint8:
            processed_mask = processed_mask.astype(np.uint8)

    # 1) Magnitude gate
    mean_mag = mag_full.mean(axis=0)
    mask_mag = percentile_mask(mean_mag, pct_mag_low, pct_mag_high).astype(np.float32)
    if verbose:
        print(f"[BC] mag gate → {int(mask_mag.sum())} px")

    # 2) Mean-velocity (abs phase) gate
    mean_vel = np.abs(phase_full.mean(axis=0))
    thr_vel = np.percentile(mean_vel, pct_mean_vel)
    mask_mean_vel = (mean_vel < thr_vel).astype(np.float32)
    if verbose:
        print(f"[BC] mean-vel gate → {int(mask_mean_vel.sum())} px")

    combined_mask = mask_mag * mask_mean_vel

    # 3) FFT flatness on magnitude
    fft_mag = np.abs(np.fft.rfft(mag_full, axis=0))
    num = fft_mag[1:3].sum(axis=0)
    den = fft_mag[3:].sum(axis=0) + eps
    fft_ratio = num / den
    thr_fft = np.percentile(fft_ratio, pct_fft)
    mask_fft = (fft_ratio < thr_fft).astype(np.float32)
    combined_mask *= mask_fft
    if verbose:
        print(f"[BC] fft gate → {int(mask_fft.sum())} px; combined so far → {int((combined_mask>0).sum())} px")

    # 4) std weighting (clipped)
    phase_std = phase_full.std(axis=0)
    std_low, std_high = np.percentile(phase_std, [pct_std_low, pct_std_high])
    phase_std_clipped = np.clip(phase_std, std_low, std_high)
    weight_std = (std_high - phase_std_clipped) / (std_high + eps)
    weight_map = combined_mask * weight_std  # float32

    # 5) optional radial attenuation
    if radial_center is not None:
        rr, cc = np.ogrid[:H, :W]
        dist = np.sqrt((rr - radial_center[0])**2 + (cc - radial_center[1])**2)
        flat = radial_flat
        dist_mask = np.where(dist < flat, flat, dist)
        dist_mask = dist_mask.max() - dist_mask
        dist_mask = dist_mask.astype(np.float32)
        dist_mask /= dist_mask.max() + eps
        dist_mask[dist < radial_inner_exclusion] = 0.0
        weight_map *= dist_mask

    # 6) exclude ROI vicinity: processed_mask is 1=exclude, 0=allow
    weight_map = weight_map * (1.0 - processed_mask.astype(np.float32))
    if verbose:
        print(f"[BC] after exclusion → {(weight_map > 0).sum()} px")

    # 7) choose top n_ref_pixels
    flat_w = weight_map.flatten()
    keep = flat_w > 0
    if keep.sum() == 0:
        raise ValueError("All weights are zero after masking.")

    n_keep = min(n_ref_pixels, int(keep.sum()))
    idx_top = np.argpartition(flat_w, -n_keep)[-n_keep:]
    ref_pixel_mask = np.zeros_like(flat_w, dtype=bool)
    ref_pixel_mask[idx_top] = True
    ref_pixel_mask = ref_pixel_mask.reshape(H, W)

    # 8) clean up: remove tiny specks (safer default)
    if min_component_size > 1:
        ref_pixel_mask = remove_small_objects(ref_pixel_mask, min_size=int(min_component_size))
        # mild opening to break 1px bridges
        ref_pixel_mask = binary_opening(ref_pixel_mask, footprint=np.ones((1, 1), dtype=bool))

    if not ref_pixel_mask.any():
        raise ValueError("Reference mask became empty after cleanup.")

    return ref_pixel_mask.astype(np.uint8)


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def compute_flow_and_stroke_volume(
    phase_vol: np.ndarray,
    mask: np.ndarray,
    metadata: Dict,
    *,
    magnitude_vol: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
    interpolate_n: int = 3201,
    use_background_correction: bool = True,
    background_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Compute CSF flow curve and stroke volume from phase-contrast MRI.

    Parameters
    ----------
    phase_vol : (T,H,W), scaled to [-1,1]
    mask      : (H,W) binary ROI
    metadata  : dict with keys
                • "v_enc" (mm/s)    • "pixel_size" (mm)    • "trigger_delay" (ms)
    magnitude_vol : (T,H,W) required if constructing ref mask internally
    ref_mask      : (H,W) binary static tissue mask (1=ref pixel, 0=other).
                    If provided, it is used directly (recommended for stability).
    use_background_correction : if True, apply baseline correction using ref mask.
    background_kwargs : forwarded to compute_reference_mask when ref_mask is None.

    Returns
    -------
    dict with keys:
      - stroke_vol : unsigned stroke volume, SV = 0.5 * (V+ + V−)
      - net_vol    : signed net transported volume, V+ - V−
      - pos_area / v_plus  : forward volume V+
      - neg_area / v_minus : reverse-volume magnitude V−
      plus time/flow arrays
    """
    # ----------------------------- unpack & checks -----------------------------
    v_enc = float(metadata["v_enc"]) * -10  # NOTE: keeping your original sign scaling
    pixel_size = float(metadata["pixel_size"])
    trigger_delay_ms = float(metadata["trigger_delay"])  # ms

    if phase_vol.ndim != 3:
        raise ValueError("phase_vol must be 3-D (T,H,W)")
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D (H,W)")
    if np.any(phase_vol < -1.1) or np.any(phase_vol > 1.1):
        raise ValueError("phase_vol must be scaled to the [-1, 1] range")

    T, H, W = phase_vol.shape
    if mask.shape != (H, W):
        raise ValueError("mask shape must match (H, W) of phase_vol")

    roi_pixels = int(mask.sum())
    if roi_pixels <= 0:
        raise ValueError("Empty ROI mask → cannot compute flow.")

    # ---------------- reference mask handling (recommended: pass ref_mask) ----
    if ref_mask is None and use_background_correction:
        if magnitude_vol is None:
            raise ValueError("magnitude_vol is required to compute internal background correction.")

        # SAFE exclusion by default (dilate ROI a bit; avoids wiping out everything)
        proc_exclusion = build_exclusion_mask(mask, radius_px=8)  # 1=exclude, 0=allow

        kw = dict(n_ref_pixels=4000, min_component_size=50)  # safer defaults
        if background_kwargs:
            kw.update(background_kwargs)

        # Try to build a robust reference mask
        ref_mask = compute_reference_mask(
            phase_full=phase_vol,
            mag_full=magnitude_vol,
            processed_mask=proc_exclusion,
            **kw,
        )

    # ------------------------ mean velocities & flow ---------------------------
    # voxel velocity (mm/s)
    v_vol = v_enc * phase_vol  # (T,H,W)

    # ROI mean velocity per frame (mm/s)
    v_roi_mean = (v_vol * mask).sum(axis=(1, 2)) / roi_pixels  # (T,)

    # Instantaneous uncorrected flow Q(t) = v_mean * Area (mm^3/s)
    a_pix = pixel_size ** 2  # mm^2
    A_roi = roi_pixels * a_pix  # mm^2
    flow = v_roi_mean * A_roi  # (T,)

    # ---------------------- background (baseline) correction -------------------
    flow_corr = flow
    if use_background_correction and ref_mask is not None:
        if ref_mask.shape != (H, W):
            raise ValueError("ref_mask must have shape (H, W)")
        ref_pixels = int(ref_mask.sum())
        if ref_pixels <= 0:
            raise ValueError("ref_mask is empty → cannot perform baseline correction.")
        v_ref_mean = (v_vol * ref_mask).sum(axis=(1, 2)) / ref_pixels  # (T,)
        flow_corr = (v_roi_mean - v_ref_mean) * A_roi  # (T,)

    # ------------------------- time axis & interpolation -----------------------
    period_s = trigger_delay_ms / 1e3  # s
    t = np.linspace(0.0, period_s, T, endpoint=False)  # native time (s)

    # Prefer periodic BC only when endpoints match to machine precision.
    # SciPy requires y[0]==y[-1] (within ~machine eps). Our earlier tolerance was too loose.
    y0, yN = float(flow_corr[0]), float(flow_corr[-1])
    y_max = np.nanmax(np.abs(flow_corr)) if np.isfinite(flow_corr).all() else 1.0
    periodic_eps = 100.0 * np.finfo(np.float64).eps * max(1.0, y_max)  # very tight
    use_periodic = (T > 2) and (abs(yN - y0) <= periodic_eps)

    bc = "periodic" if use_periodic else "natural"

    # If we choose periodic, snap the last sample to the first to satisfy SciPy's check.
    y_for_cs = flow_corr.copy()
    if use_periodic:
        y_for_cs[-1] = y_for_cs[0]

    try:
        cs = CubicSpline(t, y_for_cs, bc_type=bc)
    except ValueError:
        # If strict periodicity still fails (e.g., due to numerical quirks), fall back safely.
        cs = CubicSpline(t, flow_corr, bc_type="natural")

    t_interp = np.linspace(0.0, period_s, int(interpolate_n), endpoint=False)
    flow_interp = cs(t_interp)


    # ---------------------------- integrals (with dt) --------------------------
    pos = np.clip(flow_interp, 0.0, None)         # (mm^3/s)
    neg = np.clip(-flow_interp, 0.0, None)        # (mm^3/s)

    pos_area = float(np.trapz(pos, t_interp))     # V+ (mm^3)
    neg_area = float(np.trapz(neg, t_interp))     # V− magnitude (mm^3)

    # Article-consistent stroke volume:
    # SV = 0.5 * (V+ + |V−|)
    stroke_vol = 0.5 * (pos_area + neg_area)

    # Optional signed net transported volume (not stroke volume)
    net_vol = pos_area - neg_area

    vol_cum = cumulative_trapezoid(flow_interp, t_interp, initial=0.0)
    flow_range = float(flow_interp.max() - flow_interp.min())

    return {
        "t": t,
        "flow": flow,
        "flow_corr": flow_corr,
        "t_interp": t_interp,
        "flow_interp": flow_interp,
        "stroke_vol": stroke_vol,
        "net_vol": net_vol,
        "pos_area": pos_area,
        "neg_area": neg_area,
        "flow_range": flow_range,
        "vol_cum": vol_cum,
        "v_plus": pos_area,
        "v_minus": neg_area,
    }

# -----------------------------------------------------------------------------
# Metadata helper
# -----------------------------------------------------------------------------

def row_to_metadata(csv_path: str, sample_id: str) -> dict:
    """
    Load per-sample acquisition metadata from CSV and return a dict with:
      - v_enc (mm/s)
      - pixel_size (mm)
      - trigger_delay (ms)

    The CSV must have columns: sample, v_enc, pixel_size, delay_trigger.
    Robust to string-y numbers like "[645.0]" or "645,0".
    """
    import pandas as pd

    def _to_float(x) -> float:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        m = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
        if not m:
            raise ValueError(f"Cannot parse float from value: {x!r}")
        return float(m[0])

    df = pd.read_csv(csv_path)
    required_cols = {"sample", "v_enc", "pixel_size", "delay_trigger"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    row = df.loc[df["sample"] == sample_id]
    if row.empty:
        examples = ", ".join(map(str, df["sample"].head(8).tolist()))
        raise ValueError(f"Sample '{sample_id}' not in {csv_path}. Examples: {examples}")

    return {
        "v_enc": float(_to_float(row["v_enc"].iloc[0])),
        "pixel_size": float(_to_float(row["pixel_size"].iloc[0])),
        "trigger_delay": float(_to_float(row["delay_trigger"].iloc[0])),
    }


# -----------------------------------------------------------------------------
# Quick CLI sanity check (optional)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Example usage & quick visual sanity check (UPDATED)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import pathlib as _pl

    # ------------------ pick a subject you have on disk ------------------
    DATA_ROOT = _pl.Path("data/test")
    SUBJECT_ID = "Patient_2202021349"   # change if needed
    subj_dir = DATA_ROOT / SUBJECT_ID

    # ------------------ load arrays & metadata ---------------------------
    phase_full = np.load(subj_dir / "phase.npy") * 2 - 1  # scale → [-1,1]
    mag_full   = np.load(subj_dir / "mag.npy")
    mask_gt    = np.load(subj_dir / "mask.npy").astype(np.uint8)  # (240,240)
    metadata   = row_to_metadata("merged_metadata_file.csv", SUBJECT_ID)

    # Optional: prediction mask saved elsewhere (skip if missing)
    pred_path = _pl.Path("outputs/preds_single") / f"{SUBJECT_ID}_pred.npy"
    mask_pred = None
    if pred_path.exists():
        print(f"[INFO] Found prediction: {pred_path}")
        mask_pred = np.load(pred_path).astype(np.uint8)
    else:
        print(f"[WARN] No prediction found at {pred_path}; plotting GT only.")

    # ------------------ build one subject-level ref mask -----------------
    # Conservative exclusion from GT ROI (avoid over-exclusion)
    excl = build_exclusion_mask(mask_gt, radius_px=8)  # 1=exclude, 0=allow

    # Try default BC; if it fails, fall back to a looser setting
    try:
        ref_mask = compute_reference_mask(
            phase_full, mag_full,
            processed_mask=excl,
            n_ref_pixels=4000,
            min_component_size=50,
            verbose=True,
        )
        ref_mode = "default"
    except Exception as e:
        print(f"[WARN] default BC failed → {e}")
        try:
            ref_mask = compute_reference_mask(
                phase_full, mag_full,
                processed_mask=excl,
                n_ref_pixels=2000,
                pct_mag_low=0, pct_mag_high=99,
                pct_mean_vel=99, pct_fft=99,
                pct_std_low=5, pct_std_high=99,
                radial_center=None,
                min_component_size=10,
                verbose=True,
            )
            ref_mode = "loose"
        except Exception as e2:
            print(f"[WARN] loose BC failed → {e2}")
            ref_mask = None
            ref_mode = "none"

    # ------------------ compute flows (GT and Pred) ----------------------
    # GT with shared ref mask
    res_gt = compute_flow_and_stroke_volume(
        phase_vol=phase_full,
        mask=mask_gt,
        metadata=metadata,
        magnitude_vol=mag_full,
        ref_mask=ref_mask,                      # reuse same ref for fairness
        use_background_correction=(ref_mask is not None),
        interpolate_n=3201,
    )
    print(f"[INFO] GT SV (BC={ref_mode}): {res_gt['stroke_vol']:.2f} mm³")

    # Prediction with same ref (if available)
    res_pred = None
    if mask_pred is not None:
        res_pred = compute_flow_and_stroke_volume(
            phase_vol=phase_full,
            mask=mask_pred,
            metadata=metadata,
            magnitude_vol=mag_full,
            ref_mask=ref_mask,
            use_background_correction=(ref_mask is not None),
            interpolate_n=3201,
        )
        print(f"[INFO] Pred SV (BC={ref_mode}): {res_pred['stroke_vol']:.2f} mm³")

    # ------------------ (optional) uncorrected curves --------------------
    res_gt_unc = compute_flow_and_stroke_volume(
        phase_vol=phase_full, mask=mask_gt, metadata=metadata,
        magnitude_vol=mag_full, ref_mask=None, use_background_correction=False, interpolate_n=3201
    )
    if mask_pred is not None:
        res_pred_unc = compute_flow_and_stroke_volume(
            phase_vol=phase_full, mask=mask_pred, metadata=metadata,
            magnitude_vol=mag_full, ref_mask=None, use_background_correction=False, interpolate_n=3201
        )

    # ------------------ plot corrected & uncorrected ---------------------
    plt.figure(figsize=(10, 5))
    # corrected
    plt.plot(res_gt["t"], res_gt["flow_corr"], label=f"Manual (BC={ref_mode})", linewidth=2)
    if res_pred is not None:
        plt.plot(res_pred["t"], res_pred["flow_corr"], label=f"Prediction (BC={ref_mode})", linewidth=2, linestyle="--")
    # uncorrected
    plt.plot(res_gt_unc["t"], res_gt_unc["flow"], label="Manual (uncorrected)", alpha=0.6)
    if mask_pred is not None:
        plt.plot(res_pred_unc["t"], res_pred_unc["flow"], label="Prediction (uncorrected)", alpha=0.6, linestyle=":")

    plt.title(f"CSF Flow — {SUBJECT_ID}")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mm³/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
