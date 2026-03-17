#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bland_altman_with_patient_colors.py

Patient color & rater marker mapping (same policy as your scatter figures):
- Patient colors: deterministic mapping using matplotlib 'tab10' by sorted patient name:
    color_map = {patient_i : tab10[i % 10]} with patients = sorted(unique patients)
  NOTE: The exact (patient → RGBA) mapping used at runtime is saved to:
        outputs/abstract/patient_color_mapping.txt
        and printed in the console. This mirrors the reference code behavior.

- Rater markers (by role/session), consistent with your reference:
    DL model         → ■ (square), solid
    Novice (Eder-1)  → ▲ (triangle), filled
    Novice (Eder-2)  → ▲ (triangle), empty (facecolors='none', edgecolors=color)

Tweaks for visibility on Bland–Altman plots:
- To ensure EVERY data sample is visible on the y-axis (differences), we apply a very small,
  deterministic vertical jitter per point (based on patient name hash). This prevents points
  with identical (mean, diff) from sitting exactly on top of each other while preserving the
  overall Bland–Altman interpretation. You can disable jitter by setting JITTER_Y=0.0.

Outputs:
- CSVs: bland_altman_summary.csv, icc_summary.csv, comparisons_long.csv
- Plots: one BA plot per biomarker × role (Novice, DL), with patient-color & rater-markers
- A bar plot for ICC(3,1) test–retest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import hashlib

# ----------------------- Config -----------------------
IN_CSV = Path("outputs/abstract/abstract_biomarkers.csv")
OUT_DIR = Path("outputs/abstract")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Vertical jitter amplitude as a fraction of data range (set 0 to disable)
JITTER_Y = 0.01  # 1% of (max(diff)-min(diff)); deterministic per patient

# ----------------- Load & parse dataframe -----------------
df_raw = pd.read_csv(IN_CSV)

def parse_sample(s):
    # split from the rightmost '-' to separate patient
    if isinstance(s, str) and '-' in s:
        rater_token, patient = s.rsplit('-', 1)
        return rater_token, patient
    return s, None

df = df_raw.copy()
df[["rater_token","patient_id"]] = df["sample"].apply(lambda s: pd.Series(parse_sample(s)))

def role_and_session(rater_token):
    if rater_token == "model":
        return "DL", "model", 1
    if "-" in rater_token:
        name, sess = rater_token.split("-", 1)
        try:
            session = int(sess)
        except:
            session = None
        role = "Novice" if name.lower() == "eder" else "Expert"
        return role, name, session
    return "Expert", rater_token, None

roles, names, sessions = [], [], []
for tok in df["rater_token"]:
    r, n, s = role_and_session(tok)
    roles.append(r); names.append(n); sessions.append(s)

df["role"] = roles
df["rater_name"] = names
df["session"] = sessions

# Harmonize biomarker columns
df = df.rename(columns={
    "roi_area_mm2": "roi_mm2",
    "flow_amp_mm3_per_s": "flow_amp_mm3_s",
    "stroke_vol_unsigned_mm3": "sv_mm3"
})
df["flow_amp_ml_s"] = df["flow_amp_mm3_s"] / 1000.0
df["sv_ml"] = df["sv_mm3"] / 1000.0

# ----------------- Reference (experts mean per patient) -----------------
experts = df[df["role"]=="Expert"]
ref = experts.groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean().add_prefix("ref_")
cmp_df = df[df["role"].isin(["Novice","DL"])].merge(ref, on="patient_id", how="left")

# ----------------- Patient color mapping (tab10, deterministic) -----------------
patients_sorted = sorted(df["patient_id"].dropna().astype(str).unique().tolist())
cmap = plt.get_cmap("tab10")
patient_color = {p: cmap(i % 10) for i, p in enumerate(patients_sorted)}

# Save and print the mapping actually used
mapping_txt = OUT_DIR / "patient_color_mapping.txt"
with open(mapping_txt, "w", encoding="utf-8") as f:
    for p in patients_sorted:
        rgba = patient_color[p]
        f.write(f"{p}: {rgba}\n")
print(f"[INFO] Saved patient→color mapping to: {mapping_txt}")

# ----------------- Bland–Altman & metrics tables -----------------
def bland_altman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    diff = x - y
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = bias - 1.96*sd
    loa_high = bias + 1.96*sd
    mae = float(np.mean(np.abs(diff)))
    return bias, sd, loa_low, loa_high, mae

def ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = np.mean(x), np.mean(y)
    vx = np.var(x, ddof=1) if len(x)>1 else 0.0
    vy = np.var(y, ddof=1) if len(y)>1 else 0.0
    sxy = np.cov(x, y, ddof=1)[0,1] if len(x)>1 else 0.0
    denom = vx + vy + (mx - my)**2
    return float((2*sxy) / denom) if denom != 0 else 1.0

ba_rows = []
for role in ["Novice","DL"]:
    for metric, ref_metric, units in [
        ("roi_mm2", "ref_roi_mm2", "mm^2"),
        ("flow_amp_mm3_s", "ref_flow_amp_mm3_s", "mm^3/s"),
        ("sv_mm3", "ref_sv_mm3", "mm^3")
    ]:
        sub = cmp_df[cmp_df["role"]==role]
        x = sub[metric].values
        y = sub[ref_metric].values
        bias, sd, lo, hi, mae = bland_altman(x, y)
        ccc_val = ccc(x, y)
        ba_rows.append({
            "role": role,
            "biomarker": metric,
            "units": units,
            "n": int(np.sum(~np.isnan(x) & ~np.isnan(y))),
            "mean_ref": float(np.mean(y)),
            "mean_rater": float(np.mean(x)),
            "bias": bias,
            "sd": sd,
            "loa_low_95": lo,
            "loa_high_95": hi,
            "loa_range_95": hi - lo,
            "MAE": mae,
            "CCC": ccc_val
        })

ba_table = pd.DataFrame(ba_rows)

# ----------------- ICC(3,1) test–retest -----------------
def icc_3_1(x1, x2):
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    n = len(x1)
    if n < 2:
        return np.nan
    grand_mean = np.mean(np.concatenate([x1, x2]))
    msb = 2*np.sum(((x1+x2)/2 - grand_mean)**2)/(n-1)
    msw = (np.sum((x1 - (x1+x2)/2)**2 + (x2 - (x1+x2)/2)**2))/(n-1)
    denom = msb + msw
    return float((msb - msw)/denom) if denom != 0 else 1.0

icc_rows = []
exp_s1 = (experts[experts["session"]==1].groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean())
exp_s2 = (experts[experts["session"]==2].groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean())
for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    pts = sorted(set(exp_s1.index).intersection(set(exp_s2.index)))
    icc_val = icc_3_1(exp_s1.loc[pts, biom].values, exp_s2.loc[pts, biom].values) if pts else np.nan
    icc_rows.append({"group":"Experts (pooled)", "biomarker": biom, "ICC_3_1": icc_val})

nov = df[(df["role"]=="Novice")]
nov_s1 = nov[nov["session"]==1].set_index("patient_id")
nov_s2 = nov[nov["session"]==2].set_index("patient_id")
for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    pts = sorted(set(nov_s1.index).intersection(set(nov_s2.index)))
    icc_val = icc_3_1(nov_s1.loc[pts, biom].values, nov_s2.loc[pts, biom].values) if pts else np.nan
    icc_rows.append({"group":"Novice (Eder)", "biomarker": biom, "ICC_3_1": icc_val})

for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    icc_rows.append({"group":"DL (deterministic)", "biomarker": biom, "ICC_3_1": 1.0})

icc_table = pd.DataFrame(icc_rows)

# ----------------- Save tables -----------------
ba_csv = OUT_DIR / "bland_altman_summary.csv"
icc_csv = OUT_DIR / "icc_summary.csv"
cmp_df_csv = OUT_DIR / "comparisons_long.csv"
ba_table.to_csv(ba_csv, index=False)
icc_table.to_csv(icc_csv, index=False)
cmp_df.to_csv(cmp_df_csv, index=False)

# ----------------- Plot helpers -----------------
def _deterministic_patient_jitter(patient_id, base_amp):
    """Deterministic tiny jitter based on patient_id string hash."""
    if base_amp == 0.0:
        return 0.0
    h = hashlib.sha1(str(patient_id).encode("utf-8")).hexdigest()
    # map hex -> [-0.5, 0.5]
    v = (int(h[:8], 16) / 0xFFFFFFFF) - 0.5
    return float(v) * base_amp

def bland_altman_plot_by_patient(ax, sub_df, metric, ref_metric, pretty, units, role):
    """
    Per-role BA plot using patient colors and rater markers:
      - DL: square
      - Novice session 1: filled triangle
      - Novice session 2: empty triangle
    Adds small deterministic vertical jitter so all points are visible.
    """
    # Compute BA stats
    x = sub_df[metric].values
    y = sub_df[ref_metric].values
    mean_vals = (x + y) / 2.0
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    loa_low, loa_high = bias - 1.96*sd, bias + 1.96*sd

    # Base jitter amplitude from y-range
    y_min, y_max = (float(np.min(diff)), float(np.max(diff))) if len(diff) else (0.0, 0.0)
    y_range = max(y_max - y_min, 1e-9)
    jitter_amp = JITTER_Y * y_range

    # Scatter by row to apply patient colors + role markers + session fill
    for i, row in sub_df.iterrows():
        p = row["patient_id"]
        color = patient_color.get(p, "gray")
        m = row[metric]; r = row[ref_metric]
        if pd.isna(m) or pd.isna(r):
            continue
        mean_xy = (m + r) / 2.0
        dval = (m - r)

        # apply deterministic vertical jitter so all samples are visible
        dval_j = dval + _deterministic_patient_jitter(p, jitter_amp)

        if role == "DL":
            # square
            ax.scatter(mean_xy, dval_j, marker="s", s=60, color=color, edgecolors="black", linewidths=0.5, zorder=3)
        else:
            # Novice: session 1 filled, session 2 empty
            sess = row.get("session", None)
            if sess == 2:
                ax.scatter(mean_xy, dval_j, marker="^", s=60, facecolors="none",
                           edgecolors=color, linewidths=1.2, zorder=3)
            else:
                ax.scatter(mean_xy, dval_j, marker="^", s=60, color=color,
                           edgecolors="black", linewidths=0.5, zorder=3)

    # Lines
    ax.axhline(bias, linestyle='--', color="black", linewidth=1.0, label="Bias")
    ax.axhline(loa_low, linestyle=':', color="black", linewidth=1.0, label="95% LoA")
    ax.axhline(loa_high, linestyle=':', color="black", linewidth=1.0)

    # Labels
    ax.set_title(f"{pretty} — {role} vs Reference")
    ax.set_xlabel(f"Mean of methods ({units})")
    ax.set_ylabel(f"Difference (Method − Ref) ({units})")

    # Legends (user markers + patient colors)
    if role == "DL":
        user_handles = [
            Line2D([0],[0], marker="s", linestyle="None", markersize=8, color="black", label="Model (DL)"),
        ]
    else:
        user_handles = [
            Line2D([0],[0], marker="^", linestyle="None", markersize=8, color="black",
                   label="Novice (Eder) — session 1 (filled)"),
            Line2D([0],[0], marker="^", linestyle="None", markersize=8, markerfacecolor='none',
                   markeredgecolor='black', color="black", label="Novice (Eder) — session 2 (empty)"),
        ]
    ax.legend(handles=user_handles, loc="best", title="Users")

    # Patient legend on the right
    patient_handles = [Line2D([0],[0], marker="o", linestyle="None", markersize=8,
                              color=patient_color[p], label=p) for p in patients_sorted]
    leg2 = ax.legend(handles=patient_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Patients")
    ax.add_artist(ax.get_legend())

# ----------------- Generate BA plots -----------------
plot_files = []
biom_meta = [
    ("roi_mm2", "ref_roi_mm2", "ROI area", "mm^2"),
    ("flow_amp_mm3_s", "ref_flow_amp_mm3_s", "Flow amplitude", "mm^3/s"),
    ("sv_mm3", "ref_sv_mm3", "Stroke volume", "mm^3")
]

for role in ["Novice","DL"]:
    role_df = cmp_df[cmp_df["role"]==role].copy()
    # Ensure patient_id is string for hashing stability
    role_df["patient_id"] = role_df["patient_id"].astype(str)
    for metric, ref_metric, pretty, units in biom_meta:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        bland_altman_plot_by_patient(ax, role_df, metric, ref_metric, pretty, units, role)
        fname = OUT_DIR / f"BA_{metric}_{role}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(str(fname))

# ----------------- ICC bar plot -----------------
fig, ax = plt.subplots(figsize=(6.4, 4.8))
plot_icc = icc_table.pivot(index="biomarker", columns="group", values="ICC_3_1")
plot_icc.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Test–retest ICC(3,1)")
ax.set_ylabel("ICC")
icc_plot_path = OUT_DIR / "ICC_bar.png"
fig.tight_layout()
fig.savefig(icc_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)
