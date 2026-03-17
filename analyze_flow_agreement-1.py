#!/usr/bin/env python3
"""
Analyze flow-function agreement across ALL evaluated models.

- Discovers runs in outputs/: unet2d_<mode>_c<crop>_b<base>_<loss>
- For each run and each TEST subject:
    * Predict mask from clean data (phase/mag), center crop = run crop size
    * Build ONE reference mask (from manual mask + clean images) per subject
    * Compute background-corrected flows (manual & predicted) using SAME ref
    * Align predicted flow to manual with circular lag ±MAX_LAG and sign flip
    * Compute metrics: rho (lag/sign-robust), NRMSE (z-score), SV/vol errors,
      peak/valley magnitude & timing error, zero-crossing timing error
- Outputs:
    * CSV: outputs/flow_agreement/flow_metrics_per_subject.csv
    * CSV: outputs/flow_agreement/flow_metrics_summary.csv  (median, IQR, n)
    * CSV: outputs/flow_agreement/stats_friedman_wilcoxon.csv
    * PNG: outputs/flow_agreement/rho_box.png
    * PNG: outputs/flow_agreement/sv_bland_altman.png
    * PNG: outputs/flow_agreement/overlay_examples.png

Usage:
    python analyze_flow_agreement.py \
        --config config.yaml \
        --runs_root outputs \
        --split test \
        --thresh 0.5 \
        --max_lag 4 \
        --interp_n 3201
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, friedmanchisquare, wilcoxon

# Project imports
from src.models.unet2d import UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.temporal_features import (
    temporal_std, temporal_tv, dft_bandpower_excl_dc, dft_magnitudes_bins
)
from csf_flow import (
    compute_flow_and_stroke_volume,
    compute_reference_mask,
    build_exclusion_mask,
    row_to_metadata,
)

# -------------------- run discovery --------------------

RUN_RE = re.compile(
    r"^unet2d_(?P<mode>tvt|std|dft_k123|pca|full|dft_power)_c(?P<crop>\d+)_b(?P<base>\d+)_(?P<loss>dice|tversky|focal_dice|flow_dice)$"
)

MODES_IN_CH = {"full": 64, "pca": 1, "dft_power": 1, "tvt": 1, "std": 1, "dft_k123": 3}

def parse_run_name(name: str) -> Tuple[str, int, int, str] | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    return (m.group("mode"), int(m.group("crop")), int(m.group("base")), m.group("loss"))

# -------------------- IEEE-TMI display mapping --------------------

MODE_DISPLAY = {
    "full": "Full",
    "pca": "PCA",
    "std": "STD",
    "tvt": "TVT",
    "dft_power": "DFT-Power",
    "dft_k123": "DFT(1–3)",  # en dash
}

LOSS_DISPLAY = {
    "dice": "Dice",
    "flow_dice": "Flow-Dice",
    "focal_dice": "Focal-Dice",
    "tversky": "Tversky",
}

def display_label_from_mode_loss(mode: str, loss: str, sep: str = " · ") -> str:
    """Return a single-line display label consistent with the IEEE TMI table."""
    mode_disp = MODE_DISPLAY.get(mode, mode)
    loss_disp = LOSS_DISPLAY.get(loss, loss)
    return f"{mode_disp}{sep}{loss_disp}"

def display_label_from_group_key(group_key: str, sep: str = " · ") -> str:
    """group_key comes as '<mode>+<loss>' (e.g., 'full+flow_dice')."""
    try:
        mode, loss = group_key.split("+", 1)
    except ValueError:
        return group_key
    return display_label_from_mode_loss(mode, loss, sep=sep)

# -------------------- data helpers --------------------

def center_crop(arr: np.ndarray, size: int) -> Tuple[np.ndarray, Tuple[int,int]]:
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top+size, left:left+size], (top, left)

def embed_crop_mask(mask_crop: np.ndarray, full_shape: Tuple[int,int], top_left: Tuple[int,int]) -> np.ndarray:
    H, W = full_shape
    top, left = top_left
    size = mask_crop.shape[-1]
    full = np.zeros((H, W), dtype=mask_crop.dtype)
    full[top:top+size, left:left+size] = mask_crop
    return full

def _first_pc(vol: np.ndarray) -> np.ndarray:
    C, H, W = vol.shape
    X = vol.reshape(C, H*W).astype(np.float32)
    X -= X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc = (S[0] * Vt[0]).reshape(H, W)
    pc = (pc - pc.mean()) / (pc.std() + 1e-8)
    return pc.astype(np.float32)

def build_input_for_mode(mode: str, phase_crop: np.ndarray, mag_crop: np.ndarray) -> np.ndarray:
    if mode == "full":
        return np.concatenate([phase_crop, mag_crop], axis=0).astype(np.float32)
    if mode == "pca":
        img = _first_pc(np.concatenate([phase_crop, mag_crop], axis=0))[None, ...]
        return img
    if mode == "dft_power":
        img = dft_bandpower_excl_dc(phase_crop)[None, ...]
        return img.astype(np.float32)
    if mode == "tvt":
        img = temporal_tv(phase_crop)[None, ...]
        return img.astype(np.float32)
    if mode == "std":
        img = temporal_std(phase_crop)[None, ...]
        return img.astype(np.float32)
    if mode == "dft_k123":
        img = dft_magnitudes_bins(phase_crop, bins=(1,2,3))
        return img.astype(np.float32)
    raise ValueError(f"Unsupported mode: {mode}")

# -------------------- model I/O --------------------

def load_model(run_dir: Path, in_ch: int, out_ch: int, base_ch: int, device) -> UNet2D:
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = load_ckpt(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model")
    if state is None:
        raise KeyError(f"Checkpoint has no 'state_dict'/'model' keys: {list(ckpt.keys())}")
    model = UNet2D(in_channels=in_ch, out_channels=1, base_channels=base_ch).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_mask_crop(model: UNet2D, inp_chw: np.ndarray, device, thresh: float = 0.5) -> np.ndarray:
    vol = (inp_chw - inp_chw.mean()) / (inp_chw.std() + 1e-8)
    import torch
    x = torch.from_numpy(vol[None, ...]).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
    return (prob >= thresh).astype(np.uint8)

# -------------------- alignment & metrics --------------------

def align_lag_sign(man: np.ndarray, pred: np.ndarray, max_lag: int) -> Tuple[float, int, int, np.ndarray]:
    """Return best (rho, lag, sign, pred_aligned) under circular lag ±max_lag and sign flip."""
    best_rho, best_lag, best_sign = -2.0, 0, 1
    for lag in range(-max_lag, max_lag + 1):
        p = np.roll(pred, lag)
        for s in (1, -1):
            r, _ = pearsonr(man, s * p)
            if r > best_rho:
                best_rho, best_lag, best_sign = r, lag, s
    pred_aligned = best_sign * np.roll(pred, best_lag)
    return best_rho, best_lag, best_sign, pred_aligned

def nrmse_z(a: np.ndarray, b: np.ndarray) -> float:
    az = (a - a.mean()) / (a.std() + 1e-8)
    bz = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.sqrt(np.mean((az - bz) ** 2)))

def timing_errors(t: np.ndarray, q_m: np.ndarray, q_p: np.ndarray) -> Dict[str, float]:
    # Peaks (max/min) and first zero-cross
    i_max_m, i_min_m = int(np.argmax(q_m)), int(np.argmin(q_m))
    i_max_p, i_min_p = int(np.argmax(q_p)), int(np.argmin(q_p))
    peak_mag_err = float(np.max(q_p) - np.max(q_m))
    valley_mag_err = float(np.min(q_p) - np.min(q_m))
    peak_t_err_ms = float((t[i_max_p] - t[i_max_m]) * 1000.0)
    valley_t_err_ms = float((t[i_min_p] - t[i_min_m]) * 1000.0)

    def first_zero_cross(x: np.ndarray) -> int:
        s = np.sign(x)
        idx = np.where(np.diff(s) != 0)[0]
        return int(idx[0] + 1) if len(idx) else 0

    zc_m, zc_p = first_zero_cross(q_m), first_zero_cross(q_p)
    zc_t_err_ms = float((t[zc_p] - t[zc_m]) * 1000.0)
    return {
        "peak_mag_err": peak_mag_err,
        "valley_mag_err": valley_mag_err,
        "peak_t_err_ms": peak_t_err_ms,
        "valley_t_err_ms": valley_t_err_ms,
        "zero_cross_t_err_ms": zc_t_err_ms,
    }

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs_root", default="outputs")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max_lag", type=int, default=4)
    ap.add_argument("--interp_n", type=int, default=3201)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_root = Path(cfg["data"]["root"]) / (cfg["data"]["test_dir"] if args.split == "test" else cfg["data"]["train_dir"])
    meta_csv = cfg["data"]["metadata_csv"]
    runs_root = Path(args.runs_root)

    out_root = Path("outputs/flow_agreement")
    fig_root = out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # discover runs
    discovered: List[Tuple[Path, str, int, int, str]] = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        parsed = parse_run_name(p.name)
        if parsed is None:
            continue
        mode, crop, base, loss = parsed
        ck = p / "checkpoints" / "best_model.pt"
        if not ck.exists():
            continue
        discovered.append((p, mode, crop, base, loss))

    if not discovered:
        print("No evaluated runs found under", runs_root)
        return

    print(f"[INFO] Discovered {len(discovered)} runs.")

    # subject list
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"[INFO] Found {len(subjects)} subjects in {data_root}")

    # Prepare per-subject rows
    rows: List[Dict] = []

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-load manual & reference per subject (once)
    subj_cache = {}
    for sid in subjects:
        sdir = data_root / sid
        phase = np.load(sdir / "phase.npy").astype(np.float32)
        mag   = np.load(sdir / "mag.npy").astype(np.float32)
        man_mask = np.load(sdir / "mask.npy").astype(np.uint8)
        md = row_to_metadata(meta_csv, sid)

        # Normalize phase to [-1,1] if needed
        if phase.max() <= 1.1 and phase.min() >= -1.1:
            phase_norm = phase.copy()
        else:
            phase_norm = phase * 2.0 - 1.0

        # ONE ref mask from clean images + manual ROI exclusion
        excl = build_exclusion_mask(man_mask, radius_px=8)
        ref_mask = compute_reference_mask(
            phase_full=phase_norm,
            mag_full=mag,
            processed_mask=excl,
            n_ref_pixels=4000,
            min_component_size=50,
            verbose=False,
        )

        # Manual flow once
        res_m = compute_flow_and_stroke_volume(
            phase_vol=phase_norm,
            mask=man_mask,
            metadata=md,
            magnitude_vol=mag,
            ref_mask=ref_mask,
            use_background_correction=True,
            interpolate_n=args.interp_n,
        )

        subj_cache[sid] = dict(
            phase=phase_norm, mag=mag, man_mask=man_mask, meta=md,
            ref_mask=ref_mask, man_flow=res_m
        )

    # Iterate runs and subjects
    for run_dir, mode, crop, base, loss in discovered:
        print(f"[RUN] {run_dir.name}")
        in_ch = MODES_IN_CH[mode]
        model = load_model(run_dir, in_ch=in_ch, out_ch=1, base_ch=base, device=device)

        for sid in subjects:
            data = subj_cache[sid]
            phase, mag, man_mask = data["phase"], data["mag"], data["man_mask"]
            ref_mask, meta = data["ref_mask"], data["meta"]
            man_flow = data["man_flow"]

            # Center-crop to the model crop size
            ph_crop, (top,left) = center_crop(phase, crop)
            mg_crop, _ = center_crop(mag,   crop)

            # Build input & predict
            inp = build_input_for_mode(mode, ph_crop, mg_crop)
            pred_crop = predict_mask_crop(model, inp, device, args.thresh)
            pred_full = embed_crop_mask(pred_crop, man_mask.shape, (top, left))

            # Predicted flow on CLEAN stacks with SAME ref
            res_p = compute_flow_and_stroke_volume(
                phase_vol=phase,
                mask=pred_full,
                metadata=meta,
                magnitude_vol=mag,
                ref_mask=ref_mask,
                use_background_correction=True,
                interpolate_n=args.interp_n,
            )

            # Align & compute metrics
            t = man_flow["t_interp"]
            q_m = man_flow["flow_interp"]
            q_p = res_p["flow_interp"]

            rho, lag, sgn, q_p_aligned = align_lag_sign(q_m, q_p, max_lag=args.max_lag)
            nrmse = nrmse_z(q_m, q_p_aligned)

            # volumes & stroke
            sv_m, sv_p = man_flow["stroke_vol"], res_p["stroke_vol"]
            vplus_err = abs(res_p["v_plus"] - man_flow["v_plus"])
            vminus_err = abs(res_p["v_minus"] - man_flow["v_minus"])
            sv_abs_err = abs(sv_p - sv_m)
            sv_signed_err = sv_p - sv_m

            # timing & peak/valley errors
            te = timing_errors(t, q_m, q_p_aligned)

            rows.append({
                "subject": sid,
                "run": run_dir.name,
                "mode": mode,
                "loss": loss,
                "crop": crop,
                "base": base,
                "rho": rho,
                "best_lag": lag,
                "best_sign": sgn,
                "nrmse_z": nrmse,
                "sv_manual": sv_m,
                "sv_pred": sv_p,
                "sv_abs_err": sv_abs_err,
                "sv_signed_err": sv_signed_err,
                "vplus_abs_err": vplus_err,
                "vminus_abs_err": vminus_err,
                "peak_mag_err": te["peak_mag_err"],
                "valley_mag_err": te["valley_mag_err"],
                "peak_t_err_ms": te["peak_t_err_ms"],
                "valley_t_err_ms": te["valley_t_err_ms"],
                "zero_cross_t_err_ms": te["zero_cross_t_err_ms"],
            })

    # ---------------- Save per-subject CSV ----------------
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_subj_csv = out_root / "flow_metrics_per_subject.csv"
    df.to_csv(per_subj_csv, index=False)
    print(f"[OK] {per_subj_csv}")

    # ---------------- Per-model summary (median [IQR]) ----------------
    def iqr(x: pd.Series) -> float:
        return float(np.percentile(x.to_numpy(), 75) - np.percentile(x.to_numpy(), 25))

    agg = {
        "rho": ["median", iqr, "count"],
        "nrmse_z": ["median", iqr],
        "sv_abs_err": ["median", iqr],
        "vplus_abs_err": ["median", iqr],
        "vminus_abs_err": ["median", iqr],
        "peak_t_err_ms": ["median", iqr],
        "valley_t_err_ms": ["median", iqr],
        "zero_cross_t_err_ms": ["median", iqr],
    }
    summary = df.groupby(["mode","loss"]).agg(agg)
    summary.columns = ["_".join([c for c in col if isinstance(c, str)]) for col in summary.columns.values]
    summary_csv = out_root / "flow_metrics_summary.csv"
    summary.to_csv(summary_csv)
    print(f"[OK] {summary_csv}")

    # ---------------- Stats: Friedman + pairwise Wilcoxon (Holm) ----------------
    # Only over subjects present in ALL runs (paired design)
    runs_all = sorted(df["run"].unique().tolist())
    # sets of subjects per run
    subj_sets = [set(df.loc[df["run"] == r, "subject"].unique().tolist()) for r in runs_all]
    common_subjects = set.intersection(*subj_sets) if subj_sets else set()
    common_subjects = sorted(common_subjects)

    stats_rows = []
    # Only compute paired stats if we have a decent intersection
    if len(common_subjects) >= 5 and len(runs_all) >= 3:
        # Friedman on rho
        mat_rho = []
        for r in runs_all:
            svals = (
                df.loc[(df["run"] == r) & (df["subject"].isin(common_subjects))]
                  .sort_values("subject")["rho"]
                  .to_numpy()
            )
            mat_rho.append(svals)
        try:
            stat_f, p_f = friedmanchisquare(*mat_rho)
            stats_rows.append({
                "metric": "rho", "test": "Friedman",
                "stat": stat_f, "p": p_f, "N": len(common_subjects), "k": len(runs_all)
            })
        except Exception as e:
            print("[WARN] Friedman failed:", e)

        # Pairwise Wilcoxon with Holm correction
        pairs = []
        for i in range(len(runs_all)):
            for j in range(i + 1, len(runs_all)):
                ri, rj = runs_all[i], runs_all[j]
                di = (
                    df.loc[(df["run"] == ri) & (df["subject"].isin(common_subjects))]
                      .sort_values("subject")["rho"].to_numpy()
                )
                dj = (
                    df.loc[(df["run"] == rj) & (df["subject"].isin(common_subjects))]
                      .sort_values("subject")["rho"].to_numpy()
                )
                if len(di) == len(dj) and len(di) >= 5:
                    try:
                        stat, p = wilcoxon(di, dj, zero_method="zsplit", alternative="two-sided")
                        pairs.append((ri, rj, p))
                    except Exception:
                        pass

        # Holm step-down
        pairs_sorted = sorted(pairs, key=lambda x: x[2])
        m = len(pairs_sorted)
        for idx, (ri, rj, p) in enumerate(pairs_sorted, start=1):
            p_holm = min(1.0, p * (m - idx + 1))
            stats_rows.append({
                "metric": "rho", "test": "Wilcoxon-Holm",
                "run_i": ri, "run_j": rj, "p_raw": p, "p_holm": p_holm
            })

    stats_csv = out_root / "stats_friedman_wilcoxon.csv"
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    print(f"[OK] {stats_csv}")

    # ---------------- Figures ----------------
    # 1) Rho boxplot by run (mode+loss) with IEEE-TMI display names
    grp_key = df["mode"] + "+" + df["loss"]
    df_plot = df.copy()
    df_plot["group"] = grp_key
    med_order = (
        df_plot.groupby("group")["rho"]
              .median()
              .sort_values(ascending=False)
              .index.tolist()
    )

    # Build data and labels with display mapping
    data = [df_plot.loc[df_plot["group"] == g, "rho"].to_numpy() for g in med_order]
    labels = [display_label_from_group_key(g, sep=" · ") for g in med_order]

    plt.figure(figsize=(max(8, 0.6 * len(med_order)), 5))
    plt.boxplot(data, labels=labels, whis=1.5, showfliers=True)
    plt.ylabel("Lag/sign-robust Pearson ρ")
    plt.title("Flow-function agreement (manual vs. predicted)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "rho_box.png", dpi=180)
    plt.close()

    # 2) Bland–Altman of SV — overlay all runs (light alpha)
    plt.figure(figsize=(6.8, 6))
    for rname, grp in df.groupby("run"):
        mean_sv = (grp["sv_manual"] + grp["sv_pred"]) / 2.0
        diff_sv = grp["sv_pred"] - grp["sv_manual"]
        plt.scatter(mean_sv, diff_sv, alpha=0.45, label=rname)
    # global bias/LOA over all points
    all_diff = (df["sv_pred"] - df["sv_manual"]).to_numpy()
    bias = all_diff.mean()
    sd = all_diff.std(ddof=1) if len(all_diff) > 1 else 0.0
    plt.axhline(bias, linewidth=1.2)
    plt.axhline(bias + 1.96*sd, linestyle="--")
    plt.axhline(bias - 1.96*sd, linestyle="--")
    plt.xlabel("Mean SV (mm³)")
    plt.ylabel("SV difference (Pred − Manual) (mm³)")
    plt.title("Bland–Altman of Stroke Volume (all runs)")
    # Legend omitted intentionally to preserve readability.
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_root / "sv_bland_altman.png", dpi=180)
    plt.close()

    # 3) Overlay examples (best/median/worst by rho) for the TOP median-ρ run
    med_by_run = df.groupby("run")["rho"].median().sort_values(ascending=False)
    top_run = med_by_run.index[0]
    top_pairs = df[df["run"]==top_run].sort_values("rho")
    if len(top_pairs) >= 3:
        pick_idx = [0, len(top_pairs)//2, len(top_pairs)-1]
        picks = top_pairs.iloc[pick_idx]
        run_dir_name = top_run
        # Need to rebuild flows for plotting
        run_parsed = parse_run_name(Path(top_run).name)
        if run_parsed:
            mode, crop, base, loss = run_parsed
            in_ch = MODES_IN_CH[mode]
            model = load_model(Path(args.runs_root)/top_run, in_ch=in_ch, out_ch=1, base_ch=base, device=device)

            fig, axes = plt.subplots(1, len(picks), figsize=(4.2*len(picks), 3.6))
            if len(picks)==1:
                axes = [axes]
            for ax, (_, row) in zip(axes, picks.iterrows()):
                sid = row["subject"]
                data = subj_cache[sid]
                phase, mag, man_mask = data["phase"], data["mag"], data["man_mask"]
                ref_mask, meta = data["ref_mask"], data["meta"]
                man_flow = data["man_flow"]
                ph_crop, (top,left) = center_crop(phase, crop)
                mg_crop, _ = center_crop(mag, crop)
                inp = build_input_for_mode(mode, ph_crop, mg_crop)
                pred_crop = predict_mask_crop(model, inp, device, args.thresh)
                pred_full = embed_crop_mask(pred_crop, man_mask.shape, (top, left))
                res_p = compute_flow_and_stroke_volume(phase, pred_full, meta, magnitude_vol=mag,
                                                       ref_mask=ref_mask, use_background_correction=True,
                                                       interpolate_n=args.interp_n)
                t = man_flow["t_interp"]
                q_m = man_flow["flow_interp"]
                q_p = res_p["flow_interp"]
                rho, lag, sgn, q_p_aligned = align_lag_sign(q_m, q_p, max_lag=args.max_lag)
                ax.plot(t, q_m, linewidth=1.8, label="Manual")
                ax.plot(t, q_p_aligned, linewidth=1.2, linestyle="--", label="Pred.")
                anonymized_sid = "Pathological participant " if sid[0] == "P" else "Healthy participant "
                ax.set_title(f"{anonymized_sid}\nρ={rho:.3f}, lag={lag}, sgn={sgn}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Flow (mm³/s)")
                ax.grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)
            # Use IEEE-TMI display names in the figure title
            run_label = display_label_from_mode_loss(mode, loss, sep=" · ")
            plt.suptitle(f"Waveform overlays — {run_label}")
            plt.tight_layout(rect=[0,0,1,0.92])
            plt.savefig(fig_root / "overlay_examples.png", dpi=180)
            plt.close()

    print("[DONE] Outputs in:", out_root)

if __name__ == "__main__":
    main()
