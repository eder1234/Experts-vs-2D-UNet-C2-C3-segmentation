import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Re-load just in case
in_path = Path("outputs/abstract/abstract_biomarkers.csv")
df_raw = pd.read_csv(in_path)

# --------- Parse identifiers ---------
# sample format: "<rater_token>-<PATIENT_NAME>", e.g., "Eder-1-GRONIER", "model-GRONIER"
def parse_sample(s):
    # split from the rightmost '-' to separate patient
    if isinstance(s, str) and '-' in s:
        rater_token, patient = s.rsplit('-', 1)
        return rater_token, patient
    return s, None

df = df_raw.copy()
df[["rater_token","patient_id"]] = df["sample"].apply(lambda s: pd.Series(parse_sample(s)))

# derive role and session
def role_and_session(rater_token):
    if rater_token == "model":
        return "DL", "model", 1
    # match "<name>-<session>"
    if "-" in rater_token:
        name, sess = rater_token.split("-", 1)
        # session as int when possible
        try:
            session = int(sess)
        except:
            session = None
        # novice = Eder
        role = "Novice" if name.lower() == "eder" else "Expert"
        return role, name, session
    # fallback
    return "Expert", rater_token, None

roles, names, sessions = [], [], []
for tok in df["rater_token"]:
    r, n, s = role_and_session(tok)
    roles.append(r); names.append(n); sessions.append(s)

df["role"] = roles
df["rater_name"] = names
df["session"] = sessions

# --------- Harmonize biomarker columns ---------
# Real CSV has:
#   roi_area_mm2
#   flow_amp_mm3_per_s
#   stroke_vol_unsigned_mm3
# We will also compute ml-based for reporting convenience (1 ml = 1000 mm3)
df = df.rename(columns={
    "roi_area_mm2": "roi_mm2",
    "flow_amp_mm3_per_s": "flow_amp_mm3_s",
    "stroke_vol_unsigned_mm3": "sv_mm3"
})
df["flow_amp_ml_s"] = df["flow_amp_mm3_s"] / 1000.0
df["sv_ml"] = df["sv_mm3"] / 1000.0

# --------- Build expert reference (mean across experts and BOTH sessions per patient) ---------
experts = df[df["role"]=="Expert"]
ref = experts.groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean().add_prefix("ref_")

# Merge ref back for comparisons (Novice & DL)
cmp_df = df[df["role"].isin(["Novice","DL"])].merge(ref, on="patient_id", how="left")

# --------- Bland-Altman & metrics ---------
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
    vx, vy = np.var(x, ddof=1) if len(x)>1 else 0.0, np.var(y, ddof=1) if len(y)>1 else 0.0
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

# --------- ICC(3,1) test–retest ---------
def icc_3_1(x1, x2):
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    n = len(x1)
    if n < 2: 
        return np.nan
    msb = 2*np.sum(((x1+x2)/2 - np.mean(np.concatenate([x1,x2])))**2)/(n-1)
    msw = (np.sum((x1 - (x1+x2)/2)**2 + (x2 - (x1+x2)/2)**2))/(n-1)
    denom = msb + msw
    return float((msb - msw)/denom) if denom != 0 else 1.0

icc_rows = []

# Experts pooled: mean per patient for session 1 vs session 2
exp_s1 = (experts[experts["session"]==1].groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean())
exp_s2 = (experts[experts["session"]==2].groupby("patient_id")[["roi_mm2","flow_amp_mm3_s","sv_mm3"]].mean())
for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    # only keep patients present in both sessions
    pts = sorted(set(exp_s1.index).intersection(set(exp_s2.index)))
    icc_val = icc_3_1(exp_s1.loc[pts, biom].values, exp_s2.loc[pts, biom].values) if pts else np.nan
    icc_rows.append({"group":"Experts (pooled)", "biomarker": biom, "ICC_3_1": icc_val})

# Novice: Eder session 1 vs 2
nov = df[(df["role"]=="Novice")]
nov_s1 = nov[nov["session"]==1].set_index("patient_id")
nov_s2 = nov[nov["session"]==2].set_index("patient_id")
for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    pts = sorted(set(nov_s1.index).intersection(set(nov_s2.index)))
    icc_val = icc_3_1(nov_s1.loc[pts, biom].values, nov_s2.loc[pts, biom].values) if pts else np.nan
    icc_rows.append({"group":"Novice (Eder)", "biomarker": biom, "ICC_3_1": icc_val})

# DL deterministic
for biom in ["roi_mm2","flow_amp_mm3_s","sv_mm3"]:
    icc_rows.append({"group":"DL (deterministic)", "biomarker": biom, "ICC_3_1": 1.0})

icc_table = pd.DataFrame(icc_rows)

# --------- Save outputs ----------
out_dir = Path("outputs/abstract")
out_dir.mkdir(parents=True, exist_ok=True)
ba_csv = out_dir / "bland_altman_summary.csv"
icc_csv = out_dir / "icc_summary.csv"
cmp_df_csv = out_dir / "comparisons_long.csv"

ba_table.to_csv(ba_csv, index=False)
icc_table.to_csv(icc_csv, index=False)
cmp_df.to_csv(cmp_df_csv, index=False)

# --------- Plots: Bland-Altman for each biomarker & role ---------
def bland_altman_plot(ax, x, y, title, units):
    mean_vals = (x + y)/2.0
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1) if len(diff)>1 else 0.0
    loa_low, loa_high = bias - 1.96*sd, bias + 1.96*sd
    ax.scatter(mean_vals, diff)
    ax.axhline(bias, linestyle='--')
    ax.axhline(loa_low, linestyle=':')
    ax.axhline(loa_high, linestyle=':')
    ax.set_title(title)
    ax.set_xlabel(f"Mean of methods ({units})")
    ax.set_ylabel(f"Difference (Method - Ref) ({units})")

plot_files = []

biom_meta = [
    ("roi_mm2", "ref_roi_mm2", "ROI area", "mm^2"),
    ("flow_amp_mm3_s", "ref_flow_amp_mm3_s", "Flow amplitude", "mm^3/s"),
    ("sv_mm3", "ref_sv_mm3", "Stroke volume", "mm^3")
]

for role in ["Novice","DL"]:
    sub = cmp_df[cmp_df["role"]==role]
    for metric, ref_metric, pretty, units in biom_meta:
        x = sub[metric].values
        y = sub[ref_metric].values
        fig, ax = plt.subplots(figsize=(6,4))
        bland_altman_plot(ax, x, y, f"{pretty} — {role} vs Reference", units)
        fname = out_dir / f"BA_{metric}_{role}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(str(fname))

# ICC bar chart
fig, ax = plt.subplots(figsize=(6,4))
plot_icc = icc_table.pivot(index="biomarker", columns="group", values="ICC_3_1")
plot_icc.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Test–retest ICC(3,1)")
ax.set_ylabel("ICC")
icc_plot_path = out_dir / "ICC_bar.png"
fig.tight_layout()
fig.savefig(icc_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)
plot_files.append(str(icc_plot_path))

# Provide the user with links
outputs = {
    "ba_csv": str(ba_csv),
    "icc_csv": str(icc_csv),
    "cmp_df_csv": str(cmp_df_csv),
    "plots": plot_files
}
