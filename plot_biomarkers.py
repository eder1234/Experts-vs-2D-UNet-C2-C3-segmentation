#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import matplotlib.ticker as mtick

import matplotlib.ticker as mtick

# ... keep the rest ...

def _scale_and_units(metric: str):
    """
    Return (scale_fn, unit_label) for plotting.
    - stroke_vol: mm^3 -> mL   (÷1000)
    - flow_range: mm^3/s -> mL/s (÷1000)
    """
    if metric == "stroke_vol":
        return (lambda x: x / 1000.0, "mL")
    if metric == "flow_range":
        return (lambda x: x / 1000.0, "mL/s")
    return (lambda x: x, "")

def _tight_limits(x, y, pad_ratio=0.06):
    mn = float(min(np.min(x), np.min(y)))
    mx = float(max(np.max(x), np.max(y)))
    if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
        return None
    span = mx - mn
    pad = max(1e-6, pad_ratio * span)
    lo = max(0.0, mn - pad)
    hi = mx + pad
    return lo, hi

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def make_scatter(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=28, alpha=0.9)
    lims = _tight_limits(x, y, pad_ratio=0.08)
    if lims:
        ax.plot([lims[0], lims[1]], [lims[0], lims[1]])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    # regression
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.plot(xs, m*xs + c, linestyle="--")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{title}\n$r$={r:.3f}, slope={m:.3f}")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    # pretty ticks
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(6))

def make_bland_altman(ax, a, b, title, ylabel):
    mean_ab = (a + b) / 2.0
    diff = b - a
    bias, lo, hi = bland_altman(a, b)
    ax.scatter(mean_ab, diff, s=28, alpha=0.9)
    ax.axhline(bias, color='k')
    ax.axhline(lo, color='k', linestyle='--')
    ax.axhline(hi, color='k', linestyle='--')
    # y-lims around LOA with padding
    span = hi - lo
    pad = 0.1 * span if span > 0 else 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{title}\nBias={bias:.2f}, LOA=[{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("Mean (GT, Pred)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(6))

def bland_altman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (bias, loa_lo, loa_hi): bias = mean(b - a)
    LOA = bias ± 1.96 * SD of differences
    """
    diff = b - a
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd
    return bias, loa_lo, loa_hi

def summary_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # per model summary: mean±SD of GT, Pred, and error
    out = []
    for model, g in df.groupby("model"):
        gt = g[f"gt_{metric}"].to_numpy()
        pr = g[f"pr_{metric}"].to_numpy()
        err = pr - gt
        out.append({
            "model": model,
            f"{metric}_GT_mean": np.mean(gt),
            f"{metric}_GT_sd":   np.std(gt, ddof=1),
            f"{metric}_PR_mean": np.mean(pr),
            f"{metric}_PR_sd":   np.std(pr, ddof=1),
            f"{metric}_bias":    np.mean(err),
            f"{metric}_loa_lo":  np.mean(err) - 1.96*np.std(err, ddof=1),
            f"{metric}_loa_hi":  np.mean(err) + 1.96*np.std(err, ddof=1),
            f"{metric}_pearson_r": np.corrcoef(gt, pr)[0,1] if len(gt) > 1 else np.nan,
            f"{metric}_n": len(gt),
        })
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser("Plots & tables for biomarker accuracy")
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="One or more CSVs from eval_biomarkers.py (different models)")
    ap.add_argument("--out_dir", default="outputs/biomarkers/figures")
    ap.add_argument("--n_example_curves", type=int, default=3,
                    help="Number of subjects to plot GT vs Pred flow example curves")
    ap.add_argument("--data_root", default="data/test", help="To reload per-subject flow curves if needed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(p) for p in args.csvs]
    df = pd.concat(dfs, ignore_index=True)

    # --- SCATTER & BLAND-ALTMAN for Stroke Volume ---
    for metric in [("stroke_vol"), ("flow_range")]:
        scale_fn, unit = _scale_and_units(metric)

        # scale columns to plotting units
        for col in [f"gt_{metric}", f"pr_{metric}"]:
            df[col] = scale_fn(df[col].to_numpy())

        # SCATTER overlay
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        xs_all = []; ys_all = []
        for model, g in df.groupby("model"):
            x = g[f"gt_{metric}"].to_numpy()
            y = g[f"pr_{metric}"].to_numpy()
            ax2.scatter(x, y, s=28, alpha=0.9, label=model)
            xs_all.append(x); ys_all.append(y)
        xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all)
        lims = _tight_limits(xs_all, ys_all, pad_ratio=0.08)
        if lims:
            ax2.plot([lims[0], lims[1]], [lims[0], lims[1]])
            ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel(f"GT {metric.replace('_',' ').title()} ({unit})")
        ax2.set_ylabel(f"Pred {metric.replace('_',' ').title()} ({unit})")
        ax2.set_title(f"{metric.replace('_',' ').title()} — GT vs Pred")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(mtick.MaxNLocator(6))
        ax2.yaxis.set_major_locator(mtick.MaxNLocator(6))
        _ensure_dir(out_dir / f"scatter_{metric}_overlay.png")
        fig2.tight_layout(); fig2.savefig(out_dir / f"scatter_{metric}_overlay.png", dpi=300)
        plt.close(fig2)

        # BLAND–ALTMAN per model (in plotting units)
        fig3, axes = plt.subplots(1, len(df["model"].unique()), figsize=(5*len(df['model'].unique()), 4))
        if not isinstance(axes, np.ndarray): axes = np.array([axes])
        for ax, (model, g) in zip(axes, df.groupby("model")):
            a = g[f"gt_{metric}"].to_numpy()
            b = g[f"pr_{metric}"].to_numpy()
            make_bland_altman(ax, a, b,
                              title=f"{metric.replace('_',' ').title()} — {model}",
                              ylabel=f"Pred − GT ({unit})")
        fig3.tight_layout(); fig3.savefig(out_dir / f"bland_altman_{metric}.png", dpi=300)
        plt.close(fig3)
    # Optional: per-subject flow curves (example subset)
    # We’ll pick up to n subjects per model with median absolute error on stroke_vol
    for model, g in df.groupby("model"):
        g = g.copy()
        g["sv_abs_err"] = (g["pr_stroke_vol"] - g["gt_stroke_vol"]).abs()
        g = g.sort_values("sv_abs_err")
        pick = g.head(max(1, args.n_example_curves))

        for _, row in pick.iterrows():
            sid = row["subject"]
            # To plot curves, we reload masked flows quickly using csf_flow if needed:
            # Here we just annotate bars: figure left to the main paper. Optional to skip.

    print(f"[OK] Figures → {out_dir}")

if __name__ == "__main__":
    main()
