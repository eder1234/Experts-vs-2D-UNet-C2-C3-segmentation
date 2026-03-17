#!/usr/bin/env python3
"""
Compute biomarker summary table (percent bias/LoA and ICC(1,1)) from abstract_biomarkers.csv

Inputs
------
- CSV with columns:
    sample, roi_area_mm2, flow_amp_mm3_per_s, stroke_vol_unsigned_mm3

Conventions
-----------
- sample encodes rater, session, and patient:
    - Experts: "Leo-1-PATIENT", "Kimi-2-PATIENT", "Olivier-1-PATIENT"
    - Novice:  "Eder-1-PATIENT" / "Eder-2-PATIENT"
    - Model:   "model-PATIENT"  (no session)
- Reference per patient (for percent stats) is the multi-expert mean of **session 1**:
    mean(Leo-1, Kimi-1, Olivier-1).

Outputs
-------
- biomarker_summary_table.csv: one row per biomarker × rater with:
    Biomarker, Rater, Mean ± SD (%), Bias vs. Expert Mean (%), LoA (±%), ICC(1,1)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_sample(s: str):
    parts = s.split("-")
    if parts[0].lower() == "model":
        rater = "Model"
        session = None
        patient = "-".join(parts[1:])
    else:
        rater = parts[0].capitalize()
        try:
            session = int(parts[1])
            patient = "-".join(parts[2:])
        except:
            session = None
            patient = "-".join(parts[1:])
    return rater, session, patient

def icc_1_1(values: np.ndarray) -> float:
    """
    ICC(1,1): one-way random effects, absolute agreement, single measurement.
    Shrout & Fleiss (1979). values shape: (n_subjects, k=2 sessions).
    """
    X = np.asarray(values, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan
    m_i = X.mean(axis=1, keepdims=True)
    m = X.mean()
    ss_between = k * np.sum((m_i - m) ** 2)
    ss_within = np.sum((X - m_i) ** 2)
    df_between = n - 1
    df_within = n * (k - 1)
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    denom = ms_between + (k - 1) * ms_within
    return (ms_between - ms_within) / denom if denom != 0 else np.nan

def main(in_csv: Path, out_csv: Path):
    df = pd.read_csv(in_csv)
    # parse sample
    parsed = df["sample"].apply(parse_sample)
    df[["rater","session","patient"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    biomarkers = {
        "roi_area_mm2": "ROI Surface",
        "flow_amp_mm3_per_s": "Peak Flow Amplitude",
        "stroke_vol_unsigned_mm3": "Stroke Volume (SV)",
    }

    # build expert reference (session 1 mean per patient)
    experts = {"Leo","Kimi","Olivier"}
    ref = (
        df[df["rater"].isin(experts) & (df["session"] == 1)]
        .groupby("patient")[list(biomarkers.keys())]
        .mean()
        .rename(columns={k: f"ref_{k}" for k in biomarkers})
    )
    df_ref = df.merge(ref, left_on="patient", right_index=True, how="left")

    # restrict to session 1 for humans + model (no session)
    is_session1_or_model = (df_ref["session"].eq(1)) | (df_ref["rater"].eq("Model"))
    df_s1 = df_ref[is_session1_or_model].copy()

    # percent differences and levels
    for col in biomarkers:
        df_s1[f"{col}_pct_diff"]  = 100.0 * (df_s1[col] - df_s1[f"ref_{col}"]) / df_s1[f"ref_{col}"]
        df_s1[f"{col}_pct_level"] = 100.0 * df_s1[col] / df_s1[f"ref_{col}"]

    def summarize(sub_df: pd.DataFrame, biomarker_col: str):
        diffs = sub_df[f"{biomarker_col}_pct_diff"].dropna()
        levels = sub_df[f"{biomarker_col}_pct_level"].dropna()
        mean_level = levels.mean()
        sd_level = levels.std(ddof=1)
        bias = diffs.mean()
        loa = 1.96 * diffs.std(ddof=1)
        return mean_level, sd_level, bias, loa

    rows = []
    raters_order = ["Leo","Kimi","Olivier","Eder","Model"]
    for b_col, b_name in biomarkers.items():
        for r in raters_order:
            sub = df_s1[df_s1["rater"] == r]
            if sub.empty:
                continue
            mean_level, sd_level, bias, loa = summarize(sub, b_col)
            rows.append({
                "Biomarker": b_name,
                "Rater": "Eder (novice)" if r=="Eder" else r,
                "Mean ± SD (%)": f"{mean_level:.1f} ± {sd_level:.1f}",
                "Bias vs. Expert Mean (%)": f"{bias:+.1f}",
                "LoA (±%)": f"±{loa:.1f}",
                "ICC(1,1)": None,  # filled below
            })

    summary_df = pd.DataFrame(rows)

    # ICC(1,1) for humans (session 1 vs 2)
    icc_rows = []
    humans = ["Leo","Kimi","Olivier","Eder"]
    for b_col, b_name in biomarkers.items():
        for r in humans:
            d1 = df[(df["rater"]==r) & (df["session"]==1)][["patient", b_col]].rename(columns={b_col:"s1"})
            d2 = df[(df["rater"]==r) & (df["session"]==2)][["patient", b_col]].rename(columns={b_col:"s2"})
            merged = d1.merge(d2, on="patient", how="inner")
            icc_val = np.nan if merged.empty else icc_1_1(merged[["s1","s2"]].to_numpy())
            icc_rows.append((b_name, "Eder (novice)" if r=="Eder" else r, icc_val))

    # fill ICCs
    for b_name, r_name, icc_val in icc_rows:
        mask = (summary_df["Biomarker"]==b_name) & (summary_df["Rater"]==r_name)
        summary_df.loc[mask, "ICC(1,1)"] = f"{icc_val:.3f}" if pd.notna(icc_val) else "—"
    summary_df.loc[summary_df["Rater"]=="Model", "ICC(1,1)"] = "—"

    # sort rows
    summary_df["Rater_order"] = pd.Categorical(summary_df["Rater"], ["Leo","Kimi","Olivier","Eder (novice)","Model"], ordered=True)
    summary_df = summary_df.sort_values(["Biomarker","Rater_order"]).drop(columns=["Rater_order"]).reset_index(drop=True)

    summary_df.to_csv(out_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, default=Path("abstract_biomarkers.csv"))
    ap.add_argument("--out_csv", type=Path, default=Path("biomarker_summary_table.csv"))
    args = ap.parse_args()
    main(args.in_csv, args.out_csv)
