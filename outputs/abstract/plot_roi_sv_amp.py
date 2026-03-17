#!/usr/bin/env python3
"""
plot_roi_sv_amp.py  (revised)

Changes vs previous version:
1) Plot BOTH novice (Eder) segmentations (1 and 2) as separate points using the same symbol,
   instead of plotting their mean.
2) Add a third figure: SV vs Amplitude (experts still shown as mean + 2D std ellipse).

Figures:
- roi_vs_sv.png       : ROI (x) vs unsigned SV (y)
- roi_vs_amp.png      : ROI (x) vs flow amplitude (y)
- sv_vs_amp.png       : unsigned SV (x) vs flow amplitude (y)

Conventions expected in the CSV:
- Columns: ["sample", "roi_area_mm2", "flow_amp_mm3_per_s", "stroke_vol_unsigned_mm3"]
- "sample" naming: "model-PATIENT", "Eder-1-PATIENT", "Eder-2-PATIENT",
  "Kimi-1-PATIENT", "Leo-2-PATIENT", "Olivier-1-PATIENT", etc.
- Patients are the last token in "sample" split by '-'.

Visualization:
- Three user markers: model=■, novice (Eder)=▲, experts (Kimi/Leo/Olivier)=●
- Ten patient colors (tab10). Each patient's points share a color across users.
- Experts are aggregated across all experts and both repeats to a single mean point,
  with a 2D σ ellipse (std on the corresponding axes).

Usage:
    python plot_roi_sv_amp.py --csv /path/to/abstract_biomarkers.csv --outdir ./out --ellipse_sigma 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser(description="Plot ROI/SV/Amplitude with experts 2D std ellipses and both novice repeats.")
    p.add_argument("--csv", required=True, help="Path to CSV (expects columns: sample, roi_area_mm2, flow_amp_mm3_per_s, stroke_vol_unsigned_mm3)")
    p.add_argument("--outdir", default=".", help="Output directory for figures and aggregated CSV")
    p.add_argument("--ellipse_sigma", type=float, default=1.0, help="How many sigmas for expert ellipse (default: 1.0)")
    return p.parse_args()


def parse_sample(sample: str):
    """Return (user, repeat:int|None, patient) from a sample string."""
    parts = sample.split("-")
    if parts[0] == "model":
        user = "model"
        repeat = None
        patient = parts[1] if len(parts) > 1 else "UNKNOWN"
    else:
        user = parts[0]
        repeat = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
        patient = parts[-1] if len(parts) >= 2 else "UNKNOWN"
    return user, repeat, patient


def add_std_ellipse(ax, x_mean, y_mean, x_std, y_std, color, sigma=1.0, lw=1.5, alpha=1.0):
    """Draw an axis-aligned ellipse centered at (x_mean, y_mean) with width=2*sigma*x_std, height=2*sigma*y_std."""
    if any(np.isnan([x_mean, y_mean, x_std, y_std])):
        return
    width = 2.0 * sigma * float(x_std) if np.isfinite(x_std) else 0.0
    height = 2.0 * sigma * float(y_std) if np.isfinite(y_std) else 0.0
    if width <= 0.0 and height <= 0.0:
        return
    ell = Ellipse((float(x_mean), float(y_mean)), width=width, height=height,
                  facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(ell)


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with per-patient aggregates for model and experts (means and stds)."""
    # Parse sample
    parsed = df["sample"].apply(parse_sample)
    df[["user", "repeat", "patient"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    expert_names = {"Kimi", "Leo", "Olivier"}
    df["category"] = np.where(df["user"] == "model", "model",
                       np.where(df["user"] == "Eder", "novice",
                                np.where(df["user"].isin(expert_names), "expert", "other")))

    patients = sorted(df["patient"].unique().tolist())
    rows = []
    for p in patients:
        sub = df[df["patient"] == p]
        out = {"patient": p}

        # Model
        m = sub[sub["category"] == "model"]
        out["model_roi"] = float(m["roi_area_mm2"].mean()) if len(m) else np.nan
        out["model_sv"] = float(m["stroke_vol_unsigned_mm3"].mean()) if len(m) else np.nan
        out["model_amp"] = float(m["flow_amp_mm3_per_s"].mean()) if len(m) else np.nan

        # Experts pooled across names and repeats
        e = sub[sub["category"] == "expert"]
        if len(e):
            out["experts_roi_mean"] = float(e["roi_area_mm2"].mean())
            out["experts_sv_mean"] = float(e["stroke_vol_unsigned_mm3"].mean())
            out["experts_amp_mean"] = float(e["flow_amp_mm3_per_s"].mean())
            out["experts_roi_std"] = float(e["roi_area_mm2"].std(ddof=1)) if len(e) > 1 else 0.0
            out["experts_sv_std"] = float(e["stroke_vol_unsigned_mm3"].std(ddof=1)) if len(e) > 1 else 0.0
            out["experts_amp_std"] = float(e["flow_amp_mm3_per_s"].std(ddof=1)) if len(e) > 1 else 0.0
        else:
            out["experts_roi_mean"] = np.nan
            out["experts_sv_mean"] = np.nan
            out["experts_amp_mean"] = np.nan
            out["experts_roi_std"] = np.nan
            out["experts_sv_std"] = np.nan
            out["experts_amp_std"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def plot_scatter_with_expert_ellipse(ax, color, marker_model, marker_novice_filled, marker_novice_empty, marker_expert,
                                     model_xy, novice_xy_list, expert_mean_xy, expert_std_xy, sigma):
    # Model
    if model_xy is not None and not any(np.isnan(model_xy)):
        ax.scatter(model_xy[0], model_xy[1], marker=marker_model, s=80, color=color)

    # Novice repeats: each entry in novice_xy_list is (x, y, repeat)
    for xy in novice_xy_list:
        if xy is None:
            continue
        x, y, repeat = xy
        if any(np.isnan([x, y])):
            continue
        if repeat == 2:
            # empty triangle for second repeat
            ax.scatter(x, y, marker=marker_novice_empty, s=80, facecolors="none", edgecolors=color, linewidths=1.5)
        else:
            # filled triangle for first repeat or unknown
            ax.scatter(x, y, marker=marker_novice_filled, s=80, color=color)

    # Experts mean + 2D std ellipse
    if expert_mean_xy is not None and not any(np.isnan(expert_mean_xy)):
        ax.scatter(expert_mean_xy[0], expert_mean_xy[1], marker=marker_expert, s=80, color=color)
        if expert_std_xy is not None and not any(np.isnan(expert_std_xy)):
            add_std_ellipse(ax, expert_mean_xy[0], expert_mean_xy[1],
                            expert_std_xy[0], expert_std_xy[1], color=color, sigma=sigma, lw=1.5)


def make_figure(df: pd.DataFrame, agg_df: pd.DataFrame, patients: list[str], color_map: dict,
                x_key: str, y_key: str, x_label: str, y_label: str, title: str,
                outpath: Path, sigma: float):
    """Generic figure builder reading novice individual rows from df and expert/model aggregates from agg_df."""
    marker_model = "s"   # ■
    marker_novice_filled = "^"  # ▲ filled (repeat 1)
    marker_novice_empty = "^"   # ▲ empty (repeat 2) -> use facecolors='none'
    marker_expert = "o"  # ●

    fig, ax = plt.subplots(figsize=(8, 6))

    for p in patients:
        color = color_map[p]
        # Aggregates for model & experts
        row = agg_df[agg_df["patient"] == p].iloc[0]

        # Model tuple
        model_xy = None
        if not (np.isnan(row.get(f"model_{x_key}", np.nan)) or np.isnan(row.get(f"model_{y_key}", np.nan))):
            model_xy = (row[f"model_{x_key}"], row[f"model_{y_key}"])

        # Novice BOTH repeats -> get from df directly (include repeat)
        novice_xy_list = []
        novice_rows = df[(df["patient"] == p) & (df["user"] == "Eder")]
        for _, r in novice_rows.iterrows():
            x_val = r["roi_area_mm2"] if x_key == "roi" else (
                    r["stroke_vol_unsigned_mm3"] if x_key == "sv" else r["flow_amp_mm3_per_s"])
            y_val = r["roi_area_mm2"] if y_key == "roi" else (
                    r["stroke_vol_unsigned_mm3"] if y_key == "sv" else r["flow_amp_mm3_per_s"])
            repeat = int(r["repeat"]) if not pd.isna(r.get("repeat")) else None
            novice_xy_list.append((x_val, y_val, repeat))

        # Experts
        expert_mean_xy = None
        expert_std_xy = None
        mean_x = row.get(f"experts_{x_key}_mean", np.nan)
        mean_y = row.get(f"experts_{y_key}_mean", np.nan)
        if not (np.isnan(mean_x) or np.isnan(mean_y)):
            expert_mean_xy = (mean_x, mean_y)
        std_x = row.get(f"experts_{x_key}_std", np.nan)
        std_y = row.get(f"experts_{y_key}_std", np.nan)
        if not (np.isnan(std_x) or np.isnan(std_y)):
            expert_std_xy = (std_x, std_y)

        plot_scatter_with_expert_ellipse(ax, color, marker_model, marker_novice_filled, marker_novice_empty, marker_expert,
                                         model_xy, novice_xy_list, expert_mean_xy, expert_std_xy, sigma)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Legends
    user_handles = [
        Line2D([0], [0], marker=marker_model, linestyle="None", markersize=8, color="black", label="Model (DL)"),
        Line2D([0], [0], marker=marker_novice_filled, linestyle="None", markersize=8, color="black", label="Novice (Eder) — repeat 1 (filled)"),
        Line2D([0], [0], marker=marker_novice_empty, linestyle="None", markersize=8, markerfacecolor='none', markeredgecolor='black', color="black", label="Novice (Eder) — repeat 2 (empty)"),
        Line2D([0], [0], marker=marker_expert, linestyle="None", markersize=8, color="black", label=f"Experts (mean) + {sigma}σ ellipse"),
    ]
    ax.legend(handles=user_handles, loc="best", title="Users")

    patient_handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=8, color=color_map[p], label=p) for p in patients]
    leg2 = ax.legend(handles=patient_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Patients")
    ax.add_artist(ax.get_legend())

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    needed = {"sample", "roi_area_mm2", "flow_amp_mm3_per_s", "stroke_vol_unsigned_mm3"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Parse sample once here for novice extraction
    parsed = df["sample"].apply(parse_sample)
    df[["user", "repeat", "patient"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # Aggregate for model & experts
    agg_df = aggregate_by_patient(df)
    agg_df.to_csv(outdir / "aggregated_by_patient.csv", index=False)

    # patients & colors
    patients = sorted(agg_df["patient"].dropna().astype(str).tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(patients)}

    sigma = args.ellipse_sigma

    # 1) ROI (x) vs SV (y)
    make_figure(
        df=df,
        agg_df=agg_df,
        patients=patients,
        color_map=color_map,
        x_key="roi", y_key="sv",
        x_label="ROI area (mm²)",
        y_label="Unsigned stroke volume (mm³)",
        title=f"ROI vs Stroke Volume \n model (■), novice (▲×2), experts mean±{sigma}σ (● + ellipse)",
        outpath=outdir / "roi_vs_sv.png",
        sigma=sigma,
    )

    # 2) ROI (x) vs Amplitude (y)
    make_figure(
        df=df,
        agg_df=agg_df,
        patients=patients,
        color_map=color_map,
        x_key="roi", y_key="amp",
        x_label="ROI area (mm²)",
        y_label="Flow amplitude (mm³/s)",
        title=f"ROI vs Flow Amplitude \n model (■), novice (▲×2), experts mean±{sigma}σ (● + ellipse)",
        outpath=outdir / "roi_vs_amp.png",
        sigma=sigma,
    )

    # 3) SV (x) vs Amplitude (y)
    make_figure(
        df=df,
        agg_df=agg_df,
        patients=patients,
        color_map=color_map,
        x_key="sv", y_key="amp",
        x_label="Unsigned stroke volume (mm³)",
        y_label="Flow amplitude (mm³/s)",
        title=f"Stroke Volume vs Flow Amplitude \n model (■), novice (▲×2), experts mean±{sigma}σ (● + ellipse)",
        outpath=outdir / "sv_vs_amp.png",
        sigma=sigma,
    )

    print("Wrote:")
    print(" -", outdir / "roi_vs_sv.png")
    print(" -", outdir / "roi_vs_amp.png")
    print(" -", outdir / "sv_vs_amp.png")
    print(" -", outdir / "aggregated_by_patient.csv")


if __name__ == "__main__":
    main()
