#!/usr/bin/env python3
"""
phase_shuffle_plot.py

Generate figures from precomputed phase shuffle CSVs without re-running inference.

Outputs
-------
CSV
  - outputs/phase_shuffle/summary_by_run.csv
      run,mode,loss,crop,base,baseline_dice,nAUC_shift,k50,nAUC_shuffle,s50

FIGURES (PNG + PDF)
  - outputs/figures/dice_vs_shift.png|.pdf      (mean±SEM across subjects)
  - outputs/figures/dice_vs_shuffle.png|.pdf    (mean±SEM across subjects)
  - outputs/figures/shift_shuffle_boxplots.png|.pdf (nAUC boxplots)

Usage
-----
python phase_shuffle_plot.py --outputs_root outputs --split test
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import re
from itertools import cycle

# ------------------------- constants -------------------------
NT = 32  # frames per sequence
SHIFT_GRID = list(range(-NT//2, NT//2 + 1))  # [-16..+16]
SHUFFLE_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]  # 0..1 by 0.1

# Line styles to cycle through for differentiation
LINE_STYLES = cycle(['-', '--', '-.', ':'])

# ------------------------- summaries -------------------------
def _smooth_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # lightweight smoothing for monotone-ish half-decay detection (moving average)
    if y.size < 3: return y
    k = 3
    pad = k // 2
    ypad = np.pad(y, (pad,pad), mode="edge")
    out = np.convolve(ypad, np.ones(k)/k, mode="valid")
    return out

def _half_decay_point(x: np.ndarray, y: np.ndarray) -> float | None:
    """
    Return smallest |x| (for shift) or smallest x (for shuffle) where y <= 0.5*y0.
    Assumes x is symmetric around 0 for shift, nondecreasing for shuffle.
    """
    y0 = float(y[0])
    thr = 0.5 * y0
    if x.ndim != 1: x = np.asarray(x).ravel()
    y_smooth = _smooth_y(x, y)
    # Find first crossing
    for i in range(1, len(x)):
        if y_smooth[i] <= thr:
            # linear interpolate x[i-1]..x[i]
            x0, x1 = x[i-1], x[i]
            y0i, y1 = y_smooth[i-1], y_smooth[i]
            if y1 == y0i: return float(x1)
            t = (thr - y0i) / (y1 - y0i)
            t = float(np.clip(t, 0.0, 1.0))
            return float(x0 + t*(x1-x0))
    return None

def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mean of y/y[0] across the grid (discrete nAUC in [0,1…]).
    Requires y[0]>0.
    """
    y0 = max(float(y[0]), 1e-8)
    return float(np.mean(y / y0))

# ------------------------- main routine ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--split", default="test", choices=["val","test"])
    args = ap.parse_args()

    phase_dir = Path(args.outputs_root) / "phase_shuffle"
    fig_dir = Path(args.outputs_root) / "figures"
    phase_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    shift_csv = phase_dir / "shift_per_subject.csv"
    shuffle_csv = phase_dir / "shuffle_per_subject.csv"
    summary_csv = phase_dir / "summary_by_run.csv"

    if not shift_csv.exists() or not shuffle_csv.exists():
        raise FileNotFoundError("Missing CSVs; run phase_shuffle_eval.py first.")

    # Read shift data
    shift_data = defaultdict(lambda: defaultdict(list))
    shift_meta = {}
    with open(shift_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            if run not in shift_meta:
                shift_meta[run] = {
                    "mode": row["mode"],
                    "loss": row["loss"],
                    "crop": int(row["crop"]),
                    "base": int(row["base"]),
                }
            subj = row["subject"]
            k = int(row["k"])
            dice = float(row["dice"])
            shift_data[run][subj].append((k, dice))

    # Read shuffle data
    shuffle_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    shuffle_meta = {}
    with open(shuffle_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            if run not in shuffle_meta:
                shuffle_meta[run] = {
                    "mode": row["mode"],
                    "loss": row["loss"],
                    "crop": int(row["crop"]),
                    "base": int(row["base"]),
                }
            subj = row["subject"]
            s = float(row["s"])
            rep = int(row["rep"])
            dice = float(row["dice"])
            shuffle_data[run][subj][s].append(dice)

    # Aggregate per run
    per_run_shift = {}
    per_run_shuffle = {}
    runs = sorted(set(shift_data.keys()) | set(shuffle_data.keys()))
    for run in runs:
        if run not in shift_meta or run not in shuffle_meta:
            continue
        meta = shift_meta[run]
        mode, loss, crop, base = meta["mode"], meta["loss"], meta["crop"], meta["base"]
        key = f"{mode}-{loss}-c{crop}-b{base}"

        # Shift curves
        subj_shift_curves = []
        for subj, lst in shift_data[run].items():
            sorted_lst = sorted(lst)
            ks = np.array([t[0] for t in sorted_lst])
            ds = np.array([t[1] for t in sorted_lst])
            if not np.allclose(ks, SHIFT_GRID):
                print(f"Warning: k grid mismatch for {run}-{subj}")
                continue
            subj_shift_curves.append(ds)
        if not subj_shift_curves:
            continue
        shift_arr = np.array(subj_shift_curves)
        mean_shift = shift_arr.mean(axis=0)
        sem_shift = shift_arr.std(axis=0, ddof=1) / np.sqrt(shift_arr.shape[0])
        per_run_shift[key] = {
            "x": SHIFT_GRID,
            "mean": mean_shift.tolist(),
            "sem": sem_shift.tolist(),
        }

        # Shuffle curves
        subj_shuffle_curves = []
        for subj in shift_data[run]:  # Assume same subjects
            dice_vs_s = []
            for s in SHUFFLE_GRID:
                reps = shuffle_data[run][subj][s]
                if not reps:
                    print(f"Warning: no reps for {run}-{subj}-s{s}")
                    dice_vs_s.append(0.0)
                    continue
                mean_rep = np.mean(reps)
                dice_vs_s.append(mean_rep)
            subj_shuffle_curves.append(np.array(dice_vs_s))
        shuffle_arr = np.array(subj_shuffle_curves)
        mean_shuffle = shuffle_arr.mean(axis=0)
        sem_shuffle = shuffle_arr.std(axis=0, ddof=1) / np.sqrt(shuffle_arr.shape[0])
        per_run_shuffle[key] = {
            "x": SHUFFLE_GRID,
            "mean": mean_shuffle.tolist(),
            "sem": sem_shuffle.tolist(),
        }

    if not per_run_shift:
        print("No data found.")
        return

    # ---------------- Summaries & plots ----------------
    # Baseline dice per run (k=0; s=0)
    run_order = sorted(per_run_shift.keys())

    # Collect summary rows
    summary_rows = []
    for key in run_order:
        xk = np.array(per_run_shift[key]["x"], dtype=float)
        yk = np.array(per_run_shift[key]["mean"], dtype=float)
        xs = np.array(per_run_shuffle[key]["x"], dtype=float)
        ys = np.array(per_run_shuffle[key]["mean"], dtype=float)

        # parse run meta
        m = re.match(r"(?P<mode>.+)-(?P<loss>.+)-c(?P<crop>\d+)-b(?P<base>\d+)", key)
        if not m:
            continue
        mode, loss, crop, base = m.group("mode"), m.group("loss"), int(m.group("crop")), int(m.group("base"))

        baseline_dice = float(yk[np.where(xk==0)[0][0]]) if 0 in xk else float(yk[0])
        nAUC_shift = normalized_auc(xk, yk)
        nAUC_shuffle = normalized_auc(xs, ys)
        k50 = _half_decay_point(xk, yk)  # symmetric grid; distance from 0
        if k50 is not None: k50 = abs(float(k50))
        s50 = _half_decay_point(xs, ys)

        summary_rows.append({
            "run": f"unet2d_{mode}_c{crop}_b{base}_{loss}",
            "mode": mode, "loss": loss, "crop": crop, "base": base,
            "baseline_dice": f"{baseline_dice:.6f}",
            "nAUC_shift": f"{nAUC_shift:.6f}",
            "k50": "" if k50 is None else f"{k50:.6f}",
            "nAUC_shuffle": f"{nAUC_shuffle:.6f}",
            "s50": "" if s50 is None else f"{s50:.6f}",
        })

    # Write summary CSV
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run","mode","loss","crop","base","baseline_dice","nAUC_shift","k50","nAUC_shuffle","s50"])
        w.writeheader()
        for r in summary_rows: w.writerow(r)

    # --------- Plots: Dice vs shift ---------
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['#FF6B6B'])
    plt.figure(figsize=(10, 6))
    line_styles = cycle(['-', '--', '-.', ':'])
    for key in run_order:
        x = np.array(per_run_shift[key]["x"], dtype=float)
        y = np.array(per_run_shift[key]["mean"], dtype=float)
        e = np.array(per_run_shift[key]["sem"], dtype=float)
        ls = next(line_styles)
        plt.plot(x, y, linestyle=ls, alpha=0.9, label=key)
        plt.fill_between(x, y - e, y + e, alpha=0.15)
    plt.xlabel("Circular shift k (frames, N_t=32)")
    plt.ylabel("Dice")
    plt.title(f"Dice vs Temporal Phase Shift — {args.split.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(fig_dir / "dice_vs_shift.png", dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / "dice_vs_shift.pdf", bbox_inches='tight')
    plt.close()

    # --------- Plots: Dice vs shuffle ---------
    plt.figure(figsize=(10, 6))
    line_styles = cycle(['-', '--', '-.', ':'])
    for key in run_order:
        x = np.array(per_run_shuffle[key]["x"], dtype=float)
        y = np.array(per_run_shuffle[key]["mean"], dtype=float)
        e = np.array(per_run_shuffle[key]["sem"], dtype=float)
        ls = next(line_styles)
        plt.plot(x, y, linestyle=ls, alpha=0.9, label=key)
        plt.fill_between(x, y - e, y + e, alpha=0.15)
    plt.xlabel("Shuffle rate s (fraction of frames)")
    plt.ylabel("Dice")
    plt.title(f"Dice vs Frame Shuffling — {args.split.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(fig_dir / "dice_vs_shuffle.png", dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / "dice_vs_shuffle.pdf", bbox_inches='tight')
    plt.close()

    # --------- Boxplots: nAUC metrics ---------
    nAUC_shift_vals = [float(r["nAUC_shift"]) for r in summary_rows]
    nAUC_shuffle_vals = [float(r["nAUC_shuffle"]) for r in summary_rows]
    labels = [r["mode"] + "-" + r["loss"] for r in summary_rows]

    plt.figure(figsize=(max(10, 0.5*len(labels)), 5))
    pos = np.arange(len(labels))
    plt.boxplot([nAUC_shift_vals, nAUC_shuffle_vals], tick_labels=["nAUC_shift","nAUC_shuffle"], showmeans=True)
    plt.title(f"Phase/Shuffle Robustness (nAUC) — {args.split.upper()}")
    plt.grid(True, axis="y", alpha=0.3)
    # Removed legend here, as there are no colored lines to label
    plt.tight_layout()
    plt.savefig(fig_dir / "shift_shuffle_boxplots.png", dpi=150)
    plt.savefig(fig_dir / "shift_shuffle_boxplots.pdf")
    plt.close()

    print("[OK] CSVs in outputs/phase_shuffle; figures in outputs/figures")

if __name__ == "__main__":
    main()