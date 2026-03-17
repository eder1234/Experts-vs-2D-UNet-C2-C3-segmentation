import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
shift_csv = "outputs/phase_shuffle/shift_per_subject.csv"
shuffle_csv = "outputs/phase_shuffle/shuffle_per_subject.csv"
out_dir = Path("outputs/phase_shuffle/")
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
shift_df = pd.read_csv(shift_csv)
shuffle_df = pd.read_csv(shuffle_csv)

# --------- Helper functions ---------
def mean_sem_norm_curve_shift(df: pd.DataFrame, run_name: str):
    """Return x (k), mean(y), sem(y) for Dice normalized per subject at k=0."""
    d = df[df["run"] == run_name].copy()
    # average over any replicates per (subject,k)
    d = d.groupby(["subject","k"], as_index=False)["dice"].mean()
    base = d[d["k"] == 0.0][["subject","dice"]].rename(columns={"dice":"base"})
    d = d.merge(base, on="subject", how="left")
    d["norm"] = d["dice"] / d["base"].replace(0, np.nan)
    # pivot to subjects x k grid
    piv = d.pivot_table(index="subject", columns="k", values="norm")
    x = np.array(sorted(piv.columns.values, key=lambda v: float(v)))
    y = piv[x].mean(axis=0).values.astype(float)
    sem = piv[x].std(axis=0, ddof=1).values.astype(float) / max(1, np.sqrt(piv.shape[0]))
    return x, y, sem, piv.shape[0]

def mean_sem_norm_curve_shuffle(df: pd.DataFrame, run_name: str):
    """Return x (s), mean(y), sem(y) for Dice normalized per subject at s=0, averaging over reps."""
    d = df[df["run"] == run_name].copy()
    # average over reps per (subject,s)
    d = d.groupby(["subject","s"], as_index=False)["dice"].mean()
    base = d[d["s"] == 0.0][["subject","dice"]].rename(columns={"dice":"base"})
    d = d.merge(base, on="subject", how="left")
    d["norm"] = d["dice"] / d["base"].replace(0, np.nan)
    piv = d.pivot_table(index="subject", columns="s", values="norm")
    x = np.array(sorted(piv.columns.values, key=lambda v: float(v)))
    y = piv[x].mean(axis=0).values.astype(float)
    sem = piv[x].std(axis=0, ddof=1).values.astype(float) / max(1, np.sqrt(piv.shape[0]))
    return x, y, sem, piv.shape[0]

# --------- Figure 1: Shift exemplars ---------
ex_shift = [
    ("unet2d_dft_k123_c80_b32_flow_dice", "DFT(1-3) Dice (invariant)", {"linestyle":"-", "marker":None}),
    ("unet2d_full_c80_b32_flow_dice", "Full Flow-Dice (sensitive)", {"linestyle":"--", "marker":None})
]

plt.figure(figsize=(8,5))
for run, label, style in ex_shift:
    x, y, sem, nsub = mean_sem_norm_curve_shift(shift_df, run)
    plt.plot(x, y, label=f"{label}", linestyle=style.get("linestyle","-"))
    plt.fill_between(x, y - sem, y + sem, alpha=0.2)
plt.xlabel("Circular shift $k$ (frames, $N_t=32$)")
plt.ylabel("Dice (normalized to $k{=}0$)")
plt.title("Sensitivity to Temporal Phase Shift (mean ± SEM)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best", frameon=True)
plt.tight_layout()
shift_pdf = out_dir / "shift_example.pdf"
shift_png = out_dir / "shift_example.png"
plt.savefig(shift_pdf)
plt.savefig(shift_png, dpi=200)
plt.close()

# --------- Figure 2: Shuffle exemplars ---------
ex_shuffle = [
    ("unet2d_std_c80_b32_dice", "STD Dice (variant)", {"linestyle":"--"}),
    ("unet2d_pca_c80_b32_dice", "PCA Dice (mildly invariant)", {"linestyle":"-"}),
    ("unet2d_full_c80_b32_flow_dice", "Full Flow-Dice (degraded)", {"linestyle":"-."})
]

plt.figure(figsize=(8,5))
for run, label, style in ex_shuffle:
    x, y, sem, nsub = mean_sem_norm_curve_shuffle(shuffle_df, run)
    plt.plot(x, y, label=f"{label}", linestyle=style.get("linestyle","-"))
    plt.fill_between(x, y - sem, y + sem, alpha=0.2)
plt.xlabel("Shuffle rate $s$ (fraction of frames)")
plt.ylabel("Dice (normalized to $s{=}0$)")
plt.title("Sensitivity to Frame Shuffling (mean ± SEM)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best", frameon=True)
plt.tight_layout()
shuffle_pdf = out_dir / "shuffle_example.pdf"
shuffle_png = out_dir / "shuffle_example.png"
plt.savefig(shuffle_pdf)
plt.savefig(shuffle_png, dpi=200)
plt.close()

print("Saved:", shift_pdf, shift_png, shuffle_pdf, shuffle_png)
