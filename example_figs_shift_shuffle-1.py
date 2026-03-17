import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
shift_csv = "outputs/phase_shuffle/shift_per_subject.csv"
shuffle_csv = "outputs/phase_shuffle/shuffle_per_subject.csv"
out_dir = Path("outputs/phase_shuffle/")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- Helper functions ----------
def mean_sem_rdf_curve_shift(df: pd.DataFrame, run_name: str):
    """Compute mean ± SEM of Retained Dice Fraction (RDF) across subjects for phase shifts."""
    d = df[df["run"] == run_name].copy()
    d = d.groupby(["subject", "k"], as_index=False)["dice"].mean()  # avg replicates
    base = d[d["k"] == 0.0][["subject", "dice"]].rename(columns={"dice": "base"})
    d = d.merge(base, on="subject", how="left")
    d["rdf"] = np.clip(d["dice"] / d["base"].replace(0, np.nan), 0, 1)
    piv = d.pivot_table(index="subject", columns="k", values="rdf")
    x = np.array(sorted(piv.columns.values, key=lambda v: float(v)))
    y = piv[x].mean(axis=0).values.astype(float)
    sem = piv[x].std(axis=0, ddof=1).values.astype(float) / max(1, np.sqrt(piv.shape[0]))
    return x, y, sem, piv.shape[0]

def mean_sem_rdf_curve_shuffle(df: pd.DataFrame, run_name: str):
    """Compute mean ± SEM of Retained Dice Fraction (RDF) across subjects for frame shuffling."""
    d = df[df["run"] == run_name].copy()
    d = d.groupby(["subject", "s"], as_index=False)["dice"].mean()
    base = d[d["s"] == 0.0][["subject", "dice"]].rename(columns={"dice": "base"})
    d = d.merge(base, on="subject", how="left")
    d["rdf"] = np.clip(d["dice"] / d["base"].replace(0, np.nan), 0, 1)
    piv = d.pivot_table(index="subject", columns="s", values="rdf")
    x = np.array(sorted(piv.columns.values, key=lambda v: float(v)))
    y = piv[x].mean(axis=0).values.astype(float)
    sem = piv[x].std(axis=0, ddof=1).values.astype(float) / max(1, np.sqrt(piv.shape[0]))
    return x, y, sem, piv.shape[0]

# ---------- Figure 1: Phase shift (RDF) ----------
ex_shift = [
    ("unet2d_dft_k123_c80_b32_flow_dice", "DFT(1–3) Dice (invariant)", {"linestyle": "-"}),
    ("unet2d_full_c80_b32_flow_dice", "Full Flow-Dice (sensitive)", {"linestyle": "--"}),
]

plt.figure(figsize=(8, 5), edgecolor='none', facecolor='none')  # Remove figure border
ax = plt.gca()
for run, label, style in ex_shift:
    x, y, sem, nsub = mean_sem_rdf_curve_shift(pd.read_csv(shift_csv), run)
    plt.plot(x, y, label=f"{label}", linestyle=style.get("linestyle", "-"))
    plt.fill_between(x, y - sem, y + sem, alpha=0.2, edgecolor='none')
plt.xlabel("Circular shift $k$ (frames, $N_t=32$)")
plt.ylabel("Retained Dice fraction (RDF)")
plt.title("Sensitivity to Circular Phase Shift (mean ± SEM)")
plt.ylim(0.4, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(loc="best", frameon=False)  # Remove legend border
plt.tight_layout()

# Remove axes borders but keep grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

shift_pdf = out_dir / "shift_example_rdf.pdf"
shift_png = out_dir / "shift_example_rdf.png"
plt.savefig(shift_pdf, bbox_inches='tight', pad_inches=0.1, edgecolor='none', facecolor='none')
plt.savefig(shift_png, dpi=200, bbox_inches='tight', pad_inches=0.1, edgecolor='none', facecolor='none')
plt.close()

# ---------- Figure 2: Frame shuffle (same scale, RDF for consistency) ----------
ex_shuffle = [
    ("unet2d_std_c80_b32_dice", "STD Dice (variant)", {"linestyle": "--"}),
    ("unet2d_pca_c80_b32_dice", "PCA Dice (mildly invariant)", {"linestyle": "-"}),
    ("unet2d_full_c80_b32_flow_dice", "Full Flow-Dice (degraded)", {"linestyle": "-."}),
]

plt.figure(figsize=(8, 5), edgecolor='none', facecolor='none')  # Remove figure border
ax = plt.gca()
for run, label, style in ex_shuffle:
    x, y, sem, nsub = mean_sem_rdf_curve_shuffle(pd.read_csv(shuffle_csv), run)
    plt.plot(x, y, label=f"{label}", linestyle=style.get("linestyle", "-"))
    plt.fill_between(x, y - sem, y + sem, alpha=0.2, edgecolor='none')
plt.xlabel("Shuffle rate $s$ (fraction of frames)")
plt.ylabel("Retained Dice fraction (RDF)")
plt.title("Sensitivity to Frame Shuffling (mean ± SEM)")
plt.ylim(0.4, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(loc="best", frameon=False)  # Remove legend border
plt.tight_layout()

# Remove axes borders but keep grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

shuffle_pdf = out_dir / "shuffle_example_rdf.pdf"
shuffle_png = out_dir / "shuffle_example_rdf.png"
plt.savefig(shuffle_pdf, bbox_inches='tight', pad_inches=0.1, edgecolor='none', facecolor='none')
plt.savefig(shuffle_png, dpi=200, bbox_inches='tight', pad_inches=0.1, edgecolor='none', facecolor='none')
plt.close()

print("Saved:", shift_pdf, shift_png, shuffle_pdf, shuffle_png)