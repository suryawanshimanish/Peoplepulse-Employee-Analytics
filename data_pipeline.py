"""
Data Validation, Normalization, Index Construction, and Diagnostic Metrics Pipeline
for Palo Alto Networks Employee Engagement, Satisfaction, and Burnout Analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "uploads" / "Palo Alto Networks.csv"
OUTPUT_DIR = ROOT / "outputs"

ENGAGEMENT_COLS = [
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]
ORDINAL_1_4_COLS = ENGAGEMENT_COLS + ["WorkLifeBalance"]

TRAVEL_SCORE_MAP = {
    "Non-Travel": 0.0,
    "Travel_Rarely": 50.0,
    "Travel_Frequently": 100.0,
}

EDUCATION_MAP = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor",
}


def validate_and_clean(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate ordinal scales (1-4), missing values, duplicates, and ranges."""
    audit: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isnull().sum().sum()),
        "missing_by_column": {},
        "ordinal_validation": {},
    }

    for col in ORDINAL_1_4_COLS:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            # Median imputation fallback if future datasets contain missing values
            df[col] = df[col].fillna(df[col].median())
        audit["missing_by_column"][col] = missing
        out_of_range = int(((df[col] < 1) | (df[col] > 4)).sum())
        df[col] = df[col].clip(1, 4)
        audit["ordinal_validation"][col] = {
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "out_of_range_fixed": out_of_range,
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "distribution": {
                int(k): int(v) for k, v in df[col].value_counts().sort_index().items()
            },
        }

    return audit


def compute_diagnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Engagement Index, Burnout Risk, Stability, Workload Stress, and Segments."""
    df = df.copy()

    # 1. Normalize 1-4 ordinal scales to 0-100 for comparability
    for col in ORDINAL_1_4_COLS:
        df[f"{col}_Norm"] = (df[col] - 1.0) / 3.0 * 100.0

    # 2. Engagement Index Construction
    df["Engagement_Raw"] = df[ENGAGEMENT_COLS].mean(axis=1)
    df["Engagement_Index"] = (df["Engagement_Raw"] - 1.0) / 3.0 * 100.0
    eng_std = df["Engagement_Raw"].std(ddof=1)
    if pd.notna(eng_std) and eng_std > 0:
        df["Engagement_Z"] = (df["Engagement_Raw"] - df["Engagement_Raw"].mean()) / eng_std
    else:
        df["Engagement_Z"] = 0.0

    df["Engagement_Band"] = pd.cut(
        df["Engagement_Index"],
        bins=[-0.01, 41.67, 58.33, 75.0, 100.01],
        labels=[
            "Low (≤41.7)",
            "Moderate (41.8–58.3)",
            "High (58.4–75.0)",
            "Very High (>75.0)",
        ],
    )

    # 3. Satisfaction Stability Score (0-100; 100 = identical scores across all 4 dimensions)
    # Maximum population standard deviation on [1, 4] is 1.5 (two 1s and two 4s)
    df["Satisfaction_Dispersion"] = df[ENGAGEMENT_COLS].std(axis=1, ddof=0)
    df["Satisfaction_Stability_Score"] = (
        1.0 - df["Satisfaction_Dispersion"] / 1.5
    ) * 100.0

    # 4. Burnout Risk Identification & Score
    df["OverTime_Flag"] = (df["OverTime"] == "Yes").astype(int)
    df["Low_WLB_Flag"] = (df["WorkLifeBalance"] <= 2).astype(int)

    df["Burnout_Risk_Level"] = np.where(
        (df["OverTime_Flag"] == 1) & (df["Low_WLB_Flag"] == 1),
        "High",
        np.where(
            (df["OverTime_Flag"] == 1) | (df["Low_WLB_Flag"] == 1),
            "Medium",
            "Low",
        ),
    )

    # Burnout Risk Score (0-100): 50% OverTime indicator + 50% Work-Life Imbalance strain
    df["WLB_Strain_Score"] = (4.0 - df["WorkLifeBalance"]) / 3.0 * 100.0
    df["Burnout_Risk_Score"] = (
        0.5 * (df["OverTime_Flag"] * 100.0) + 0.5 * df["WLB_Strain_Score"]
    )

    # 5. Workload Stress Indicator (0-100): 50% OverTime + 50% Business Travel intensity
    df["Travel_Intensity_Score"] = df["BusinessTravel"].map(TRAVEL_SCORE_MAP)
    df["Workload_Stress_Indicator"] = (
        0.5 * (df["OverTime_Flag"] * 100.0) + 0.5 * df["Travel_Intensity_Score"]
    )

    # 6. Descriptive Bands for Workload & Career-Stage Diagnostics
    df["Commute_Band"] = pd.cut(
        df["DistanceFromHome"],
        bins=[0, 5, 15, 100],
        labels=["Short (1–5 mi)", "Moderate (6–15 mi)", "Long (16–29 mi)"],
    )

    df["Tenure_Band"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, 100],
        labels=["0–1 yrs", "2–3 yrs", "4–5 yrs", "6–10 yrs", "11+ yrs"],
    )

    df["Role_Tenure_Band"] = pd.cut(
        df["YearsInCurrentRole"],
        bins=[-1, 1, 3, 6, 100],
        labels=["0–1 yrs", "2–3 yrs", "4–6 yrs", "7+ yrs"],
    )

    df["Promo_Gap_Band"] = pd.cut(
        df["YearsSinceLastPromotion"],
        bins=[-1, 0, 2, 5, 100],
        labels=["0 yrs (Recent)", "1–2 yrs", "3–5 yrs", "6+ yrs (Stagnant)"],
    )

    df["Manager_Tenure_Band"] = pd.cut(
        df["YearsWithCurrManager"],
        bins=[-1, 0, 2, 5, 100],
        labels=["0 yrs (New Mgr)", "1–2 yrs", "3–5 yrs", "6+ yrs"],
    )

    df["Age_Band"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 100],
        labels=["18–25", "26–35", "36–45", "46–60"],
    )

    df["Education_Label"] = df["Education"].map(EDUCATION_MAP)
    df["Attrition_Label"] = np.where(df["Attrition"] == 1, "Left", "Stayed")

    # Priority Alert Category
    def alert_tier(row: pd.Series) -> str:
        low_eng = row["Engagement_Index"] <= 41.67
        high_burn = row["Burnout_Risk_Level"] == "High"
        med_ot = (row["Burnout_Risk_Level"] == "Medium") and (row["OverTime_Flag"] == 1)
        if low_eng and high_burn:
            return "Dual Crisis (Low Eng + High Burnout)"
        if high_burn:
            return "Silent Burnout Risk (High Burnout)"
        if low_eng:
            return "Disengaged Alert (Low Engagement)"
        if med_ot:
            return "Overload Watch (OverTime)"
        return "Stable / Monitor"

    df["Priority_Alert_Tier"] = df.apply(alert_tier, axis=1)
    return df


REQUIRED_UPLOAD_COLS = [
    "Department",
    "JobRole",
    "OverTime",
    "Attrition",
    "WorkLifeBalance",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
]

OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "BusinessTravel": "Non-Travel",
    "DistanceFromHome": 5,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
    "JobLevel": 2,
    "MonthlyIncome": 5000,
    "Age": 32,
    "Education": 3,
}


def _normalize_uploaded_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate required schema and normalize boolean/string encodings for uploaded CSVs."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_UPLOAD_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Uploaded CSV is missing required diagnostic column(s): "
            + ", ".join(missing_cols)
        )
    if df.empty:
        raise ValueError("Uploaded CSV contains 0 rows.")

    for col, default_val in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val

    # Normalize Attrition to 0 / 1 integers
    if df["Attrition"].dtype == object or str(df["Attrition"].dtype) == "string":
        att_norm = df["Attrition"].astype(str).str.strip().str.lower()
        df["Attrition"] = np.where(att_norm.isin(["1", "yes", "true", "left", "y"]), 1, 0)
    else:
        df["Attrition"] = (pd.to_numeric(df["Attrition"], errors="coerce").fillna(0) > 0).astype(int)

    # Normalize OverTime to "Yes" / "No"
    if df["OverTime"].dtype == object or str(df["OverTime"].dtype) == "string":
        ot_norm = df["OverTime"].astype(str).str.strip().str.lower()
        df["OverTime"] = np.where(ot_norm.isin(["1", "yes", "true", "y"]), "Yes", "No")
    else:
        df["OverTime"] = np.where(pd.to_numeric(df["OverTime"], errors="coerce").fillna(0) > 0, "Yes", "No")

    # Normalize ordinal and numeric columns safely
    for col in ORDINAL_1_4_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for num_col in ["DistanceFromHome", "YearsAtCompany", "YearsInCurrentRole",
                    "YearsSinceLastPromotion", "YearsWithCurrManager", "JobLevel",
                    "MonthlyIncome", "Age", "Education"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(OPTIONAL_DEFAULTS.get(num_col, 0))

    df["Department"] = df["Department"].fillna("Unspecified").astype(str).str.strip()
    df["JobRole"] = df["JobRole"].fillna("Unspecified").astype(str).str.strip()
    df["BusinessTravel"] = (
        df["BusinessTravel"]
        .fillna("Non-Travel")
        .astype(str)
        .str.strip()
        .replace({"Travel Rarely": "Travel_Rarely", "Travel Frequently": "Travel_Frequently", "Non Travel": "Non-Travel"})
    )
    return df


def process_uploaded_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate, clean, and enrich an uploaded DataFrame purely in memory."""
    normalized = _normalize_uploaded_frame(raw)
    audit = validate_and_clean(normalized)
    enriched = compute_diagnostic_features(normalized)
    enriched["Travel_Intensity_Score"] = enriched["Travel_Intensity_Score"].fillna(0.0)
    enriched["Workload_Stress_Indicator"] = (
        0.5 * (enriched["OverTime_Flag"] * 100.0) + 0.5 * enriched["Travel_Intensity_Score"]
    )
    return enriched, audit


def load_diagnostic_data(csv_path: Path = DATA_PATH) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset, run validation, compute all diagnostic features."""
    raw = pd.read_csv(csv_path)
    audit = validate_and_clean(raw)
    enriched = compute_diagnostic_features(raw)
    return enriched, audit


def export_summary_artifacts() -> None:
    """Compute and persist enriched CSV and comprehensive JSON summary statistics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, audit = load_diagnostic_data()
    df.to_csv(OUTPUT_DIR / "palo_alto_enriched.csv", index=False)

    stayed = df[df["Attrition"] == 0]
    left = df[df["Attrition"] == 1]

    t_eng, p_eng = stats.ttest_ind(stayed["Engagement_Index"], left["Engagement_Index"])
    u_eng, pu_eng = stats.mannwhitneyu(stayed["Engagement_Index"], left["Engagement_Index"])
    d_eng = (left["Engagement_Index"].mean() - stayed["Engagement_Index"].mean()) / np.sqrt(
        (left["Engagement_Index"].std() ** 2 + stayed["Engagement_Index"].std() ** 2) / 2.0
    )

    t_burn, p_burn = stats.ttest_ind(stayed["Burnout_Risk_Score"], left["Burnout_Risk_Score"])
    d_burn = (left["Burnout_Risk_Score"].mean() - stayed["Burnout_Risk_Score"].mean()) / np.sqrt(
        (left["Burnout_Risk_Score"].std() ** 2 + stayed["Burnout_Risk_Score"].std() ** 2) / 2.0
    )

    ct_burn = pd.crosstab(df["Burnout_Risk_Level"], df["Attrition"])
    chi2_burn, p_chi_burn, _, _ = stats.chi2_contingency(ct_burn)

    summary = {
        "audit": audit,
        "kpis": {
            "total_employees": int(len(df)),
            "engagement_index_mean": round(float(df["Engagement_Index"].mean()), 2),
            "engagement_index_median": round(float(df["Engagement_Index"].median()), 2),
            "engagement_index_std": round(float(df["Engagement_Index"].std()), 2),
            "engagement_raw_mean": round(float(df["Engagement_Raw"].mean()), 3),
            "burnout_risk_score_mean": round(float(df["Burnout_Risk_Score"].mean()), 2),
            "work_life_balance_index_1_4": round(float(df["WorkLifeBalance"].mean()), 3),
            "work_life_balance_index_0_100": round(float(df["WorkLifeBalance_Norm"].mean()), 2),
            "satisfaction_stability_mean": round(float(df["Satisfaction_Stability_Score"].mean()), 2),
            "workload_stress_indicator_mean": round(float(df["Workload_Stress_Indicator"].mean()), 2),
            "overtime_rate_pct": round(float(df["OverTime_Flag"].mean() * 100), 2),
            "low_wlb_rate_pct": round(float(df["Low_WLB_Flag"].mean() * 100), 2),
            "high_burnout_count": int((df["Burnout_Risk_Level"] == "High").sum()),
            "high_burnout_pct": round(float((df["Burnout_Risk_Level"] == "High").mean() * 100), 2),
            "medium_burnout_count": int((df["Burnout_Risk_Level"] == "Medium").sum()),
            "medium_burnout_pct": round(float((df["Burnout_Risk_Level"] == "Medium").mean() * 100), 2),
            "low_burnout_count": int((df["Burnout_Risk_Level"] == "Low").sum()),
            "low_burnout_pct": round(float((df["Burnout_Risk_Level"] == "Low").mean() * 100), 2),
            "attrition_count": int(df["Attrition"].sum()),
            "attrition_rate_pct": round(float(df["Attrition"].mean() * 100), 2),
        },
        "contextual_attrition": {
            "stayed_engagement_mean": round(float(stayed["Engagement_Index"].mean()), 2),
            "left_engagement_mean": round(float(left["Engagement_Index"].mean()), 2),
            "engagement_diff": round(float(left["Engagement_Index"].mean() - stayed["Engagement_Index"].mean()), 2),
            "engagement_cohens_d": round(float(d_eng), 3),
            "engagement_t": round(float(t_eng), 2),
            "engagement_p": float(p_eng),
            "engagement_mw_p": float(pu_eng),
            "stayed_burnout_score": round(float(stayed["Burnout_Risk_Score"].mean()), 2),
            "left_burnout_score": round(float(left["Burnout_Risk_Score"].mean()), 2),
            "burnout_cohens_d": round(float(d_burn), 3),
            "burnout_t": round(float(t_burn), 2),
            "burnout_p": float(p_burn),
            "burnout_chi2": round(float(chi2_burn), 2),
            "burnout_chi_p": float(p_chi_burn),
        },
    }

    with open(OUTPUT_DIR / "diagnostic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Exported palo_alto_enriched.csv and diagnostic_metrics.json successfully.")


if __name__ == "__main__":
    export_summary_artifacts()
