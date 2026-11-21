"""
Builds publication-grade PDF deliverables via WeasyPrint:
1. Research_Paper_Palo_Alto_Networks.pdf (Comprehensive EDA, Methodology, Insights, Recommendations)
2. Executive_Summary_Government_Stakeholders.pdf (Policy & Governance Brief for Government Stakeholders)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import weasyprint
from scipy import stats

from data_pipeline import load_diagnostic_data, ENGAGEMENT_COLS, ORDINAL_1_4_COLS

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "figures"


def img_to_b64(filename: str) -> str:
    data = (FIG_DIR / filename).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def compute_report_context(df: pd.DataFrame) -> Dict:
    stayed = df[df["Attrition"] == 0]
    left = df[df["Attrition"] == 1]

    # Dimension stats
    dim_rows = []
    dim_labels = {
        "JobInvolvement": "Job Involvement",
        "JobSatisfaction": "Job Satisfaction",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "WorkLifeBalance": "Work-Life Balance",
    }
    for col in ORDINAL_1_4_COLS:
        t_val, p_val = stats.ttest_ind(stayed[col], left[col])
        d_val = (left[col].mean() - stayed[col].mean()) / np.sqrt(
            (left[col].std() ** 2 + stayed[col].std() ** 2) / 2.0
        )
        dim_rows.append({
            "name": dim_labels[col],
            "mean": df[col].mean(),
            "norm": df[f"{col}_Norm"].mean(),
            "std": df[col].std(),
            "low_pct": (df[col] <= 2).mean() * 100,
            "stayed": stayed[col].mean(),
            "left": left[col].mean(),
            "d": d_val,
            "p": p_val,
        })

    # Burnout tier stats
    burn_rows = []
    for lvl, desc in [
        ("Low", "OverTime = No & WLB ≥ 3"),
        ("Medium", "OverTime = Yes OR WLB ≤ 2"),
        ("High", "OverTime = Yes & WLB ≤ 2"),
    ]:
        sub = df[df["Burnout_Risk_Level"] == lvl]
        burn_rows.append({
            "level": lvl,
            "criteria": desc,
            "n": len(sub),
            "wf_pct": len(sub) / len(df) * 100,
            "eng": sub["Engagement_Index"].mean(),
            "wlb": sub["WorkLifeBalance"].mean(),
            "score": sub["Burnout_Risk_Score"].mean(),
            "attr_n": int(sub["Attrition"].sum()),
            "attr_rate": sub["Attrition"].mean() * 100,
        })

    # Job Role stats
    role_df = (
        df.groupby(["Department", "JobRole"])
        .agg(
            n=("Attrition", "count"),
            eng=("Engagement_Index", "mean"),
            wlb=("WorkLifeBalance", "mean"),
            stab=("Satisfaction_Stability_Score", "mean"),
            ot_pct=("OverTime_Flag", lambda s: s.mean() * 100),
            high_burn=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
            low_eng=("Engagement_Index", lambda s: (s <= 41.67).mean() * 100),
            burn_score=("Burnout_Risk_Score", "mean"),
            income=("MonthlyIncome", "mean"),
            attr=("Attrition", lambda s: s.mean() * 100),
        )
        .reset_index()
        .sort_values("attr", ascending=False)
    )

    # Job Level stats
    jl_df = (
        df.groupby("JobLevel")
        .agg(
            n=("Attrition", "count"),
            eng=("Engagement_Index", "mean"),
            wlb=("WorkLifeBalance", "mean"),
            ot_pct=("OverTime_Flag", lambda s: s.mean() * 100),
            high_burn=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
            income=("MonthlyIncome", "mean"),
            attr=("Attrition", lambda s: s.mean() * 100),
            attr_ot=("Attrition", lambda s: s[df.loc[s.index, "OverTime"] == "Yes"].mean() * 100),
        )
        .reset_index()
    )

    # Priority Tiers
    tier_order = [
        "Dual Crisis (Low Eng + High Burnout)",
        "Silent Burnout Risk (High Burnout)",
        "Disengaged Alert (Low Engagement)",
        "Overload Watch (OverTime)",
        "Stable / Monitor",
    ]
    tier_df = (
        df.groupby("Priority_Alert_Tier")
        .agg(
            n=("Attrition", "count"),
            attr_n=("Attrition", "sum"),
            attr_rate=("Attrition", lambda s: s.mean() * 100),
            eng=("Engagement_Index", "mean"),
            burn=("Burnout_Risk_Score", "mean"),
        )
        .reindex(tier_order)
        .reset_index()
    )
    tier_df["wf_pct"] = tier_df["n"] / len(df) * 100
    tier_df["attr_share"] = tier_df["attr_n"] / df["Attrition"].sum() * 100

    # Multivariate Logit
    X = pd.DataFrame({
        "const": 1.0,
        "OverTime = Yes": df["OverTime_Flag"],
        "Travel Frequently": (df["BusinessTravel"] == "Travel_Frequently").astype(int),
        "DistanceFromHome (+10 mi)": df["DistanceFromHome"] / 10.0,
        "YearsSinceLastPromotion (+1 yr)": df["YearsSinceLastPromotion"],
        "YearsAtCompany (+1 yr)": df["YearsAtCompany"],
        "WorkLifeBalance (+1 pt)": df["WorkLifeBalance"],
        "Age (+10 yrs)": df["Age"] / 10.0,
        "JobLevel (+1 lvl)": df["JobLevel"],
        "Engagement Index (+10 pts)": df["Engagement_Index"] / 10.0,
    })
    logit = sm.Logit(df["Attrition"], X).fit(disp=0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(df["Attrition"], logit.predict(X))

    logit_rows = []
    for name in X.columns:
        if name == "const":
            continue
        coef = logit.params[name]
        or_val = np.exp(coef)
        ci_l, ci_u = np.exp(logit.conf_int().loc[name])
        z_val = logit.tvalues[name]
        p_val = logit.pvalues[name]
        logit_rows.append({
            "feature": name,
            "coef": coef,
            "or": or_val,
            "ci_l": ci_l,
            "ci_u": ci_u,
            "z": z_val,
            "p": p_val,
        })

    return {
        "n": len(df),
        "eng_mean": df["Engagement_Index"].mean(),
        "eng_median": df["Engagement_Index"].median(),
        "eng_std": df["Engagement_Index"].std(),
        "eng_raw": df["Engagement_Raw"].mean(),
        "burn_mean": df["Burnout_Risk_Score"].mean(),
        "wlb_14": df["WorkLifeBalance"].mean(),
        "wlb_100": df["WorkLifeBalance_Norm"].mean(),
        "stab_mean": df["Satisfaction_Stability_Score"].mean(),
        "ws_mean": df["Workload_Stress_Indicator"].mean(),
        "ot_pct": df["OverTime_Flag"].mean() * 100,
        "ot_n": int(df["OverTime_Flag"].sum()),
        "low_wlb_pct": df["Low_WLB_Flag"].mean() * 100,
        "low_wlb_n": int(df["Low_WLB_Flag"].sum()),
        "attr_pct": df["Attrition"].mean() * 100,
        "attr_n": int(df["Attrition"].sum()),
        "stayed_n": len(stayed),
        "stayed_eng": stayed["Engagement_Index"].mean(),
        "left_eng": left["Engagement_Index"].mean(),
        "dim_rows": dim_rows,
        "burn_rows": burn_rows,
        "role_df": role_df,
        "jl_df": jl_df,
        "tier_df": tier_df,
        "logit_rows": logit_rows,
        "logit_auc": auc,
        "logit_r2": logit.prsquared,
    }


COMMON_CSS = """
@page {
    size: letter;
    margin: 0.72in 0.72in 0.78in 0.72in;
    @top-left {
        content: "PALO ALTO NETWORKS — PREVENTIVE WORKFORCE DIAGNOSTICS";
        font-family: 'DejaVu Sans', sans-serif;
        font-size: 7.5pt;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    @top-right {
        content: "HR ANALYTICS & BURNOUT STUDY";
        font-family: 'DejaVu Sans', sans-serif;
        font-size: 7.5pt;
        color: #64748B;
        letter-spacing: 0.5px;
    }
    @bottom-left {
        content: "Confidential Analytical Deliverable • Unified Mentor × Palo Alto Networks";
        font-family: 'DejaVu Sans', sans-serif;
        font-size: 7.5pt;
        color: #64748B;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'DejaVu Sans', sans-serif;
        font-size: 7.5pt;
        font-weight: bold;
        color: #0F172A;
    }
}
@page :first {
    @top-left { content: none; }
    @top-right { content: none; }
}
body {
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 9.6pt;
    line-height: 1.48;
    color: #1E293B;
    margin: 0;
}
.cover-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 70%, #0D9488 100%);
    color: #FFFFFF;
    padding: 26px 28px;
    border-radius: 8px;
    margin-bottom: 18px;
    border-bottom: 5px solid #EA580C;
}
.cover-tag {
    font-size: 8pt;
    font-weight: bold;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #5EEAD4;
    margin-bottom: 8px;
}
.cover-title {
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.22;
    margin: 0 0 8px 0;
}
.cover-subtitle {
    font-size: 10.5pt;
    color: #CBD5E1;
    margin: 0 0 12px 0;
}
.cover-meta {
    font-size: 8.5pt;
    color: #E2E8F0;
    border-top: 1px solid rgba(255,255,255,0.2);
    padding-top: 9px;
}
h1 {
    font-size: 13.5pt;
    color: #0F172A;
    border-bottom: 2px solid #EA580C;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 9px;
    page-break-after: avoid;
    break-after: avoid;
}
h2 {
    font-size: 11pt;
    color: #0D9488;
    margin-top: 13px;
    margin-bottom: 6px;
    page-break-after: avoid;
    break-after: avoid;
}
h3 {
    font-size: 9.8pt;
    color: #0F172A;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
}
p {
    margin: 0 0 8px 0;
    text-align: justify;
}
ul, ol {
    margin: 4px 0 9px 0;
    padding-left: 18px;
}
li {
    margin-bottom: 3.5px;
}
.kpi-grid {
    display: table;
    width: 100%;
    table-layout: fixed;
    margin: 12px 0 14px 0;
    border-spacing: 6px 0;
    break-inside: avoid;
}
.kpi-card {
    display: table-cell;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #2563EB;
    border-radius: 5px;
    padding: 9px 8px;
    text-align: center;
    vertical-align: top;
}
.kpi-card.coral { border-top-color: #EA580C; }
.kpi-card.teal { border-top-color: #0D9488; }
.kpi-card.amber { border-top-color: #D97706; }
.kpi-card.navy { border-top-color: #0F172A; }
.kpi-val {
    font-size: 12.8pt;
    font-weight: bold;
    color: #0F172A;
    line-height: 1.15;
    white-space: nowrap;
}
.kpi-lbl {
    font-size: 7.5pt;
    font-weight: bold;
    color: #475569;
    text-transform: uppercase;
    margin-top: 3px;
}
.kpi-sub {
    font-size: 7.2pt;
    color: #64748B;
    margin-top: 2px;
}
.callout {
    background: #F0FDFA;
    border-left: 4px solid #0D9488;
    padding: 10px 13px;
    border-radius: 4px;
    margin: 10px 0 12px 0;
    break-inside: avoid;
}
.callout.warn {
    background: #FEF2F2;
    border-left-color: #DC2626;
}
.callout.info {
    background: #EFF6FF;
    border-left-color: #2563EB;
}
.callout-title {
    font-weight: bold;
    font-size: 9.5pt;
    color: #0F172A;
    margin-bottom: 3px;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 7.9pt;
    break-inside: avoid;
}
th {
    background-color: #0F172A;
    color: #FFFFFF;
    text-align: left;
    padding: 5.5px 5.5px;
    font-weight: bold;
    border: 1px solid #0F172A;
}
td {
    padding: 4.5px 5.5px;
    border-bottom: 1px solid #E2E8F0;
    border-left: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
}
tr:nth-child(even) td {
    background-color: #F8FAFC;
}
.num { text-align: right; font-variant-numeric: tabular-nums; }
.tbl-cap { font-size: 8.5pt; font-weight: bold; color: #0F172A; margin: 10px 0 3px 0; page-break-after: avoid; break-after: avoid; }
.badge-high { color: #DC2626; font-weight: bold; }
.badge-med { color: #D97706; font-weight: bold; }
.badge-low { color: #0D9488; font-weight: bold; }
.fig-box {
    margin: 10px 0 13px 0;
    padding: 8px;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    background: #FFFFFF;
    break-inside: avoid;
}
.fig-box img {
    width: 100%;
    height: auto;
    display: block;
}
.fig-cap {
    font-size: 8pt;
    color: #475569;
    margin-top: 5px;
    line-height: 1.35;
}
.abstract-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 12px 15px;
    margin-bottom: 12px;
    break-inside: avoid;
}
.two-col {
    display: table;
    width: 100%;
    table-layout: fixed;
    border-spacing: 10px 0;
    break-inside: avoid;
}
.col-half {
    display: table-cell;
    vertical-align: top;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 5px;
    padding: 10px 12px;
}
.page-break {
    page-break-before: always;
    break-before: page;
}
"""


def build_research_paper(ctx: Dict) -> Path:
    figs = {f"fig{i}": img_to_b64(fn) for i, fn in [
        (1, "fig1_engagement_overview.png"),
        (2, "fig2_burnout_risk_matrix.png"),
        (3, "fig3_workload_stress_drivers.png"),
        (4, "fig4_role_department_diagnostics.png"),
        (5, "fig5_career_stage_stagnation.png"),
        (6, "fig6_contextual_attrition_drivers.png"),
        (7, "fig7_engaged_burnout_paradox.png"),
        (8, "fig8_priority_intervention_segments.png"),
    ]}

    dim_table_rows = ""
    for r in ctx["dim_rows"]:
        p_str = f"{r['p']:.2e}" if r["p"] < 0.001 else f"{r['p']:.4f}"
        dim_table_rows += f"""
        <tr>
            <td><strong>{r['name']}</strong></td>
            <td class="num">{r['mean']:.2f}</td>
            <td class="num">{r['norm']:.1f}</td>
            <td class="num">{r['std']:.2f}</td>
            <td class="num">{r['low_pct']:.1f}%</td>
            <td class="num">{r['stayed']:.2f}</td>
            <td class="num">{r['left']:.2f}</td>
            <td class="num">{r['d']:+.3f}</td>
            <td class="num">{p_str}</td>
        </tr>"""

    burn_table_rows = ""
    for r in ctx["burn_rows"]:
        cls = "badge-high" if r["level"] == "High" else ("badge-med" if r["level"] == "Medium" else "badge-low")
        burn_table_rows += f"""
        <tr>
            <td><span class="{cls}">{r['level']} Risk</span></td>
            <td>{r['criteria']}</td>
            <td class="num">{r['n']}</td>
            <td class="num">{r['wf_pct']:.1f}%</td>
            <td class="num">{r['score']:.1f}</td>
            <td class="num">{r['eng']:.1f}</td>
            <td class="num">{r['wlb']:.2f}</td>
            <td class="num">{r['attr_n']}</td>
            <td class="num"><strong>{r['attr_rate']:.1f}%</strong></td>
        </tr>"""

    role_table_rows = ""
    for _, r in ctx["role_df"].iterrows():
        role_table_rows += f"""
        <tr>
            <td><strong>{r['JobRole']}</strong></td>
            <td>{r['Department']}</td>
            <td class="num">{int(r['n'])}</td>
            <td class="num">{r['eng']:.1f}</td>
            <td class="num">{r['wlb']:.2f}</td>
            <td class="num">{r['ot_pct']:.1f}%</td>
            <td class="num">{r['high_burn']:.1f}%</td>
            <td class="num">{r['low_eng']:.1f}%</td>
            <td class="num">${r['income']:,.0f}</td>
            <td class="num"><strong>{r['attr']:.1f}%</strong></td>
        </tr>"""

    jl_table_rows = ""
    for _, r in ctx["jl_df"].iterrows():
        jl_table_rows += f"""
        <tr>
            <td><strong>Level {int(r['JobLevel'])}</strong></td>
            <td class="num">{int(r['n'])}</td>
            <td class="num">{r['eng']:.1f}</td>
            <td class="num">{r['wlb']:.2f}</td>
            <td class="num">{r['ot_pct']:.1f}%</td>
            <td class="num">{r['high_burn']:.1f}%</td>
            <td class="num">${r['income']:,.0f}</td>
            <td class="num">{r['attr']:.1f}%</td>
            <td class="num"><strong>{r['attr_ot']:.1f}%</strong></td>
        </tr>"""

    logit_table_rows = ""
    for r in ctx["logit_rows"]:
        p_str = "&lt;&nbsp;0.0001" if r["p"] < 0.0001 else f"{r['p']:.4f}"
        dir_cls = "badge-high" if r["or"] > 1.0 else "badge-low"
        logit_table_rows += f"""
        <tr>
            <td><strong>{r['feature']}</strong></td>
            <td class="num">{r['coef']:+.4f}</td>
            <td class="num"><span class="{dir_cls}">{r['or']:.3f}×</span></td>
            <td class="num">[{r['ci_l']:.2f}, {r['ci_u']:.2f}]</td>
            <td class="num">{r['z']:+.2f}</td>
            <td class="num">{p_str}</td>
        </tr>"""

    tier_table_rows = ""
    for _, r in ctx["tier_df"].iterrows():
        tier_table_rows += f"""
        <tr>
            <td><strong>{r['Priority_Alert_Tier']}</strong></td>
            <td class="num">{int(r['n'])}</td>
            <td class="num">{r['wf_pct']:.1f}%</td>
            <td class="num">{r['eng']:.1f}</td>
            <td class="num">{r['burn']:.1f}</td>
            <td class="num">{int(r['attr_n'])}</td>
            <td class="num"><strong>{r['attr_rate']:.1f}%</strong></td>
            <td class="num">{r['attr_share']:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Research Paper — Palo Alto Networks Engagement & Burnout Diagnostics</title>
<style>{COMMON_CSS}</style>
</head>
<body>

<div class="cover-banner">
    <div class="cover-tag">Preventive People Analytics • Empirical Diagnostic Research Paper</div>
    <div class="cover-title">Employee Engagement, Satisfaction, and Burnout Diagnostic Analysis at Palo Alto Networks</div>
    <div class="cover-subtitle">Moving HR Diagnostics from Reactive Attrition Reporting to Early-Warning Well-Being &amp; Workforce Risk Intelligence</div>
    <div class="cover-meta">
        <strong>Partner Context:</strong> Palo Alto Networks &amp; Unified Mentor &nbsp;|&nbsp;
        <strong>Sample Scope:</strong> N = {ctx['n']:,} Employees across 3 Departments &amp; 9 Job Roles &nbsp;|&nbsp;
        <strong>Date:</strong> November 2026
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi-card teal">
        <div class="kpi-val">{ctx['eng_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Engagement Index</div>
        <div class="kpi-sub">Raw μ = {ctx['eng_raw']:.2f} / 4.00 (SD 16.8)</div>
    </div>
    <div class="kpi-card coral">
        <div class="kpi-val">{ctx['burn_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Burnout Risk Score</div>
        <div class="kpi-sub">8.6% High Risk (33.3% Attr.)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-val">{ctx['wlb_14']:.2f} / 4.00</div>
        <div class="kpi-lbl">Work-Life Balance</div>
        <div class="kpi-sub">{ctx['wlb_100']:.1f}/100 Norm • {ctx['low_wlb_pct']:.1f}% Low (≤2)</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-val">{ctx['stab_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Satisfaction Stability</div>
        <div class="kpi-sub">Cross-facet consistency index</div>
    </div>
    <div class="kpi-card navy">
        <div class="kpi-val">{ctx['ws_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Workload Stress</div>
        <div class="kpi-sub">{ctx['ot_pct']:.1f}% Overtime • 18.8% Freq. Travel</div>
    </div>
</div>

<div class="abstract-box">
    <h2 style="margin-top:0; color:#0F172A;">Abstract</h2>
    <p style="margin-bottom:5px;">
        In high-velocity cybersecurity enterprises such as Palo Alto Networks, reactive exit interviews and post-hoc attrition tracking fail to prevent costly engineering, research, and field-sales turnover. Using organizational workforce data across <strong>N = {ctx['n']:,} employees</strong>, this diagnostic study establishes a unified, preventive employee-experience measurement framework combining five standardized Key Performance Indicators (KPIs): the <em>Composite Engagement Index</em>, <em>Burnout Risk Score</em>, <em>Work-Life Balance Index</em>, <em>Satisfaction Stability Score</em>, and <em>Workload Stress Indicator</em>.
    </p>
    <p style="margin-bottom:0;">
        Empirical results validate that engagement erosion and workload-driven burnout operate as <strong>distinct, complementary early-warning pathways</strong>. Employees who separated exhibited an <strong>8.36-point Engagement Index deficit</strong> (50.4 vs. 58.8, Cohen's <em>d</em> = −0.481, <em>p</em> = 1.49×10⁻¹²) and a <strong>17.12-point Burnout Risk elevation</strong> (49.2 vs. 32.0, <em>d</em> = +0.653, <em>p</em> = 1.54×10⁻²¹). Crucially, we uncover an <strong>"Engaged Burnout Paradox"</strong>: overtime employees report significantly <em>higher</em> engagement (59.5 vs. 56.6, <em>p</em> = 0.0035) yet depart at nearly triple the rate (30.5% vs. 10.4%). Even among Very High Engagement staff (&gt;75/100), High Burnout Risk elevates attrition 11-fold (from 3.0% to 33.3%). A multivariate logistic regression model achieves <strong>AUC = 0.803</strong>, proving independent risk multipliers from Overtime (OR = 5.02), Frequent Travel (OR = 2.40), Commute Distance (OR = 1.37 per 10 mi), and Promotion Stagnation (OR = 1.14 per year without promotion), alongside protective effects from Engagement (OR = 0.68 per +10 pts) and Work-Life Balance (OR = 0.78). Targeted manager interventions across four priority tiers—representing 31.0% of the workforce—capture <strong>55.3% of all organizational attrition</strong>.
    </p>
</div>

<h1>1. Introduction &amp; Strategic Problem Context</h1>
<p>
    Cybersecurity enterprises operate under relentless operational tempo: continuous threat-intelligence research, mission-critical incident response, rapid product release cycles, and high-stakes enterprise sales. At Palo Alto Networks, human capital is the primary engine of technological differentiation. However, traditional Human Resources analytics historically emphasizes lagging indicators—counting exits after resignations are signed rather than diagnosing upstream deterioration in engagement, role satisfaction, and recovery capacity.
</p>
<p>
    This study directly addresses four diagnostic blind spots identified in the project mandate: (1) the absence of a unified, comparable organization-wide <strong>Engagement Health Index</strong>; (2) insufficient visibility into <strong>burnout-prone roles and tenure cohorts</strong>; (3) limited understanding of how <strong>overtime, work-life imbalance, travel, and satisfaction interact</strong>; and (4) the lack of <strong>preventive manager alerts</strong> prior to voluntary separation.
</p>

<h1>2. Data Validation, Normalization &amp; KPI Methodology</h1>
<h2>2.1 Dataset Integrity &amp; Ordinal Scale Audit</h2>
<p>
    The analytical corpus comprises <strong>1,470 complete employee records</strong> across <strong>31 demographic, compensation, workload, tenure, and psychometric attributes</strong> spanning three departments: Research &amp; Development (<em>n</em> = 961, 65.4%), Sales (<em>n</em> = 446, 30.3%), and Human Resources (<em>n</em> = 63, 4.3%). Automated pipeline validation verified <strong>zero missing values (0/45,570 cells)</strong>, <strong>zero duplicate records</strong>, and strict adherence to valid [1, 4] integer bounds across all five psychometric ordinal scales (Table 1).
</p>

<div class="tbl-cap">Table 1. Psychometric Ordinal Scale Validation and Contextual Separation (N = 1,470)</div>
<table>
    <thead>
        <tr>
            <th>Diagnostic Dimension (1–4 Scale)</th>
            <th class="num">Org Mean</th>
            <th class="num">0–100 Norm</th>
            <th class="num">Std Dev</th>
            <th class="num">Low Share (≤2)</th>
            <th class="num">Stayed Mean</th>
            <th class="num">Left Mean</th>
            <th class="num">Cohen's <em>d</em></th>
            <th class="num"><em>p</em>-value</th>
        </tr>
    </thead>
    <tbody>{dim_table_rows}</tbody>
</table>

<h2>2.2 Normalization &amp; Composite KPI Definitions</h2>
<p>
    To establish direct comparability across indices and support intuitive HR governance, all 1–4 ordinal ratings were min-max normalized onto a <strong>0–100 scale</strong> via <em>X<sub>norm</sub> = ((X − 1) / 3) × 100</em> alongside standardized Z-scores. Five diagnostic KPIs were constructed:
</p>
<ul>
    <li><strong>Composite Engagement Index (0–100):</strong> Equal-weighted mean of normalized <em>JobInvolvement</em>, <em>JobSatisfaction</em>, <em>EnvironmentSatisfaction</em>, and <em>RelationshipSatisfaction</em> (Org μ = <strong>57.44</strong>, median = 58.33, SD = 16.80; raw 1–4 μ = 2.723, SD = 0.504). Standardized Z-scores (<em>Engagement_Z</em>) correlate at <em>r</em> = 0.987 with the mean of dimension-specific Z-scores.</li>
    <li><strong>Satisfaction Stability Score (0–100):</strong> Measures within-employee consistency across the four engagement dimensions: <em>100 × (1 − σ<sub>within</sub> / 1.5)</em>, where 1.5 is the maximum population standard deviation on [1, 4]. Mean stability is <strong>45.25</strong>, revealing substantial intra-individual variance.</li>
    <li><strong>Burnout Risk Level &amp; Score (0–100):</strong> Flags employees exhibiting chronic workload strain: <strong>High Risk</strong> (<em>OverTime = Yes</em> AND <em>WorkLifeBalance ≤ 2</em>; <em>n</em> = 126, 8.6%), <strong>Medium Risk</strong> (either <em>OverTime = Yes</em> OR <em>WorkLifeBalance ≤ 2</em>; <em>n</em> = 588, 40.0%), and <strong>Low Risk</strong> (<em>OverTime = No</em> AND <em>WorkLifeBalance ≥ 3</em>; <em>n</em> = 756, 51.4%). The continuous Burnout Risk Score combines OverTime (50% weight) and inverted Work-Life Balance strain <em>((4 − WLB)/3 × 100)</em> (50% weight), yielding an organizational mean of <strong>34.80</strong>.</li>
    <li><strong>Work-Life Balance Index:</strong> Direct mean of <em>WorkLifeBalance</em> across the enterprise (<strong>2.761 / 4.00</strong>; 58.71 / 100), with <strong>28.8%</strong> (<em>n</em> = 424) scoring ≤ 2 (Poor or Fair).</li>
    <li><strong>Workload Stress Indicator (0–100):</strong> Equally weights <em>OverTime</em> (0 or 100) and <em>BusinessTravel</em> intensity (Non-Travel = 0, Travel_Rarely = 50, Travel_Frequently = 100), yielding an enterprise mean of <strong>41.31</strong>.</li>
</ul>

<div class="callout info">
    <div class="callout-title">Methodological Finding — Orthogonal Engagement Facets Require Stability Monitoring</div>
    Pairwise Pearson correlations among the four engagement inputs lie strictly between <em>r</em> = −0.022 and <em>r</em> = +0.034. Because job satisfaction, work environment, peer relationships, and job involvement vary independently, an employee can register a moderate overall score while suffering acute single-facet failure (e.g., toxic environment = 1). Consequently, the <strong>Satisfaction Stability Score</strong> (<em>r</em> = +0.285 with Engagement Index) prevents single-dimension distress from being masked by composite averages.
</div>

<h1>3. Engagement Health Overview &amp; Distribution</h1>
<p>
    Organization-wide engagement exhibits mild negative skew (−0.200), clustering around a central band of 50.0–75.0 while exposing a critical lower tail. Exactly <strong>353 employees (24.0%)</strong> fall into the <strong>Low Engagement Band (≤ 41.7)</strong>, while 267 (18.2%) are Moderate (41.8–58.3), 702 (47.8%) are High (58.4–75.0), and 148 (10.1%) achieve Very High Engagement (&gt; 75.0). Across individual facets (Figure 1B), nearly <strong>39% of employees</strong> rate Job Satisfaction (38.7%), Environment Satisfaction (38.8%), and Relationship Satisfaction (39.4%) at 1 or 2, whereas Job Involvement (31.2% ≤ 2) and Work-Life Balance (28.8% ≤ 2) concentrate heavily in category 3 (~60%).
</p>

<div class="fig-box">
    <img src="{figs['fig1']}" alt="Engagement Index Distribution and Ordinal Facet Breakdown">
    <div class="fig-cap"><strong>Figure 1.</strong> (A) Normalized Composite Engagement Index density comparison between retained and departed staff, showing an 8.4-point leftward shift among leavers. (B) Workforce share across 1–4 ordinal ratings for all five psychometric metrics.</div>
</div>

<h1>4. Burnout Risk Diagnostics &amp; The Engaged Burnout Paradox</h1>
<p>
    Cross-tabulating overtime exposure with work-life balance reveals a sharp, statistically significant escalation in workforce vulnerability (Table 2, Figure 2; Pearson χ² = 67.20, <em>p</em> = 2.56×10⁻¹⁵). While slightly over half of Palo Alto Networks employees (51.4%) operate in the Low Burnout zone (9.1% attrition), <strong>40.0% experience Medium Risk</strong> (21.4% attrition) and <strong>8.6% (126 employees) sit in the High Burnout zone</strong>, where attrition reaches <strong>33.3%</strong>—and spikes to <strong>45.5%</strong> among overtime staff reporting the lowest work-life balance (WLB = 1).
</p>

<div class="tbl-cap">Table 2. Burnout Risk Classification Tiers, Well-Being Profiles, and Contextual Attrition</div>
<table>
    <thead>
        <tr>
            <th>Burnout Tier</th>
            <th>Diagnostic Rule</th>
            <th class="num">Headcount</th>
            <th class="num">Workforce %</th>
            <th class="num">Burnout Score</th>
            <th class="num">Eng. Index</th>
            <th class="num">Mean WLB</th>
            <th class="num">Exits</th>
            <th class="num">Attrition %</th>
        </tr>
    </thead>
    <tbody>{burn_table_rows}</tbody>
</table>

<div class="fig-box">
    <img src="{figs['fig2']}" alt="Burnout Risk Tiers and OverTime x Work-Life Balance Matrix">
    <div class="fig-cap"><strong>Figure 2.</strong> (A) Workforce distribution and contextual attrition across the three Burnout Risk Tiers. (B) Full 2×4 OverTime × Work-Life Balance matrix demonstrating that overtime shifts attrition from 8.5%–12.8% (WLB 2–4, No OT) up to 28.7%–45.5%.</div>
</div>

<h2>4.1 The Engaged Burnout Paradox</h2>
<p>
    A pivotal counter-intuitive finding emerges when comparing overtime cohorts: employees working overtime (<em>n</em> = 416, 28.3%) actually exhibit a <strong>statistically higher mean Engagement Index</strong> than non-overtime peers (<strong>59.48 vs. 56.63</strong>, <em>t</em> = 2.93, <em>p</em> = 0.0035), yet experience nearly <strong>triple the attrition rate (30.5% vs. 10.4%)</strong>. Even more strikingly, Figure 3 demonstrates how High Burnout dismantles the protective effect of engagement:
</p>
<ul>
    <li>Among <strong>Very High Engagement (&gt;75.0)</strong> employees with Low Burnout Risk, attrition is just <strong>3.0%</strong> (2/66). Under High Burnout Risk, attrition leaps <strong>11-fold to 33.3%</strong> (6/18).</li>
    <li>Among <strong>Moderate Engagement (41.8–58.3)</strong> employees, High Burnout produces a <strong>54.2% departure rate</strong> (13/24)—over 10× higher than Low Burnout peers in the same band (5.3%).</li>
    <li>Thus, conventional engagement surveys alone are blind to <strong>"Silent Burnout"</strong> among committed, high-output staff who remain dedicated to their work until exhaustion precipitates abrupt resignation.</li>
</ul>

<div class="fig-box">
    <img src="{figs['fig7']}" alt="Engaged Burnout Paradox">
    <div class="fig-cap"><strong>Figure 3.</strong> Interaction between Composite Engagement Bands and Burnout Risk Tiers. High Burnout systematically overrides engagement protection across every tier, elevating departures among highly engaged staff by up to 11×.</div>
</div>

<h1>5. Workload, Business Travel &amp; Commute Stress Analysis</h1>
<p>
    Analyzing workload drivers alongside geographic commute burden confirms that <strong>external energy depletion operates independently of intrinsic job enjoyment</strong> (Figure 4):
</p>
<ul>
    <li><strong>OverTime &amp; Frequent Travel Compounding:</strong> While Business Travel alone shows flat Engagement Index averages (57.2–58.3, ANOVA <em>F</em> = 0.56, <em>p</em> = 0.57), attrition climbs monotonically from <strong>8.0% (Non-Travel)</strong> to <strong>15.0% (Travel_Rarely)</strong> to <strong>24.9% (Travel_Frequently)</strong> (χ² = 24.18, <em>p</em> &lt; 0.0001). When Frequent Travel combines with OverTime (Workload Stress Indicator = 100, <em>n</em> = 86), employees post the <em>highest</em> engagement (60.95) yet suffer <strong>41.9% attrition</strong>—nearly <strong>10×</strong> the 4.3% baseline of Non-Travel/No-OverTime peers.</li>
    <li><strong>Commute Distance Burden:</strong> Longer commutes show near-zero linear correlation with Engagement (<em>r</em> = −0.004, <em>p</em> = 0.875) but significantly erode retention and well-being: attrition rises from <strong>13.8%</strong> (Short 1–5 mi, <em>n</em> = 632) to <strong>16.1%</strong> (Moderate 6–15 mi, <em>n</em> = 509) and <strong>20.7%</strong> (Long 16–29 mi, <em>n</em> = 329; <em>r</em> = 0.078, <em>p</em> = 0.0028), alongside the highest High Burnout concentration (<strong>10.0%</strong>) and lowest Work-Life Balance (2.72).</li>
</ul>

<div class="fig-box">
    <img src="{figs['fig3']}" alt="Workload Stress Drivers">
    <div class="fig-cap"><strong>Figure 4.</strong> (A) Compounding Workload Stress across OverTime × BusinessTravel combinations overlaid with Engagement Index. (B) Commute distance gradient showing elevated attrition and burnout among long-distance commuters (16–29 miles).</div>
</div>

<h1>6. Role, Department &amp; Career-Stage Stagnation Diagnostics</h1>
<h2>6.1 Departmental &amp; Job-Role Risk Segmentation</h2>
<p>
    At the departmental level, <strong>Sales</strong> displays the lowest Engagement Index (56.89), highest Low-Engagement share (26.5%), and highest overall attrition (20.6%), while <strong>Research &amp; Development</strong> carries the highest Burnout Risk Score (35.35) and High Burnout concentration (8.84%, 85 employees). <strong>Human Resources</strong> exhibits high Satisfaction Stability (48.05) and high Work-Life Balance (2.92) yet elevated attrition (19.0%), concentrated among non-managerial HR generalists (23.1%).
</p>
<p>
    Granular job-role diagnostics (Table 3, Figure 5) expose sharply differentiated risk profiles requiring tailored interventions rather than uniform HR programs:
</p>

<div class="tbl-cap">Table 3. Job-Role Engagement, Burnout Risk Exposure, Compensation, and Attrition Profile</div>
<table>
    <thead>
        <tr>
            <th>Job Role</th>
            <th>Department</th>
            <th class="num"><em>n</em></th>
            <th class="num">Eng. Index</th>
            <th class="num">WLB</th>
            <th class="num">OT %</th>
            <th class="num">High Burn %</th>
            <th class="num">Low Eng %</th>
            <th class="num">Avg Income</th>
            <th class="num">Attrition %</th>
        </tr>
    </thead>
    <tbody>{role_table_rows}</tbody>
</table>

<div class="fig-box">
    <img src="{figs['fig4']}" alt="Job Role Risk Landscape">
    <div class="fig-cap"><strong>Figure 5.</strong> (A) Two-dimensional Job Role Diagnostic Landscape plotting Engagement Index against Burnout Risk Score (bubble area = headcount, color = attrition). (B) Role prevalence of Low Engagement (≤41.7) and High Burnout Risk.</div>
</div>

<h2>6.2 Career-Stage Curves, Early-Tenure Shock &amp; Stagnation</h2>
<p>
    Examining engagement and turnover across job levels, company tenure, manager tenure, and promotion intervals uncovers four critical structural vulnerabilities (Table 4, Figure 6):
</p>
<ol>
    <li><strong>Entry-Level Overtime Crisis (Job Level 1):</strong> Level 1 staff (<em>n</em> = 543) experience 26.3% attrition overall, but when assigned overtime (<em>n</em> = 156), attrition surges to an unsustainable <strong>52.6%</strong> (vs. 15.8% without overtime).</li>
    <li><strong>Mid-Career Engagement Trough (Job Level 3):</strong> Level 3 professionals (<em>n</em> = 218) record the <strong>lowest Engagement Index of any tier (55.81)</strong>—dipping notably below both junior staff (57.52) and Level 4 leaders (59.43), reflecting mid-career compression and role plateauing.</li>
    <li><strong>Onboarding &amp; Manager-Transition Shock:</strong> Employees with <strong>0–1 years at the company</strong> (<em>n</em> = 215) exhibit the lowest Engagement Index across tenure bands (<strong>56.12</strong>), highest overtime assignment (<strong>32.1%</strong>), 10.7% High Burnout, and <strong>34.9% attrition</strong>. Similarly, employees with <strong>0 years under their current manager</strong> (<em>n</em> = 263) drop to 55.96 Engagement and 32.3% attrition (vs. 11.0% at 6+ years with the same manager).</li>
    <li><strong>Promotion Stagnation &amp; Training Deficit:</strong> While bivariate promotion-gap tables are partially masked by senior survivorship, employees waiting <strong>6+ years since last promotion</strong> (<em>n</em> = 215) suffer a distinct <strong>Engagement drop (56.5 vs. 59.4 at 3–5 yrs)</strong>, the <strong>lowest Job Satisfaction of any cohort (2.63 / 4.00, 54.4/100)</strong>, and an attrition rebound to <strong>16.3%</strong> (vs. 10.1% at 3–5 yrs). Furthermore, employees receiving <strong>zero training in the prior year</strong> (<em>n</em> = 54) show 11.1% High Burnout and 27.8% attrition (vs. 1.5% High Burnout and 9.2% attrition at 6 trainings), and top-rated performers (PerformanceRating = 4, <em>n</em> = 226) post lower engagement (55.8 vs. 57.7) and higher burnout (10.6% vs. 8.2%) than peers.</li>
</ol>

<div class="tbl-cap">Table 4. Career-Stage Diagnostics by Job Level (Levels 1–5)</div>
<table>
    <thead>
        <tr>
            <th>Job Level</th>
            <th class="num">Headcount</th>
            <th class="num">Eng. Index</th>
            <th class="num">Mean WLB</th>
            <th class="num">OverTime %</th>
            <th class="num">High Burnout %</th>
            <th class="num">Mean Income</th>
            <th class="num">Overall Attr. %</th>
            <th class="num">Attr. w/ OverTime %</th>
        </tr>
    </thead>
    <tbody>{jl_table_rows}</tbody>
</table>

<div class="fig-box">
    <img src="{figs['fig5']}" alt="Career Stage and Promotion Stagnation">
    <div class="fig-cap"><strong>Figure 6.</strong> (A) Job Level 1–5 attrition split by OverTime alongside the Level 3 Engagement trough. (B) Promotion gap trajectory showing engagement and job satisfaction peaking at 3–5 years before deteriorating sharply at 6+ years without promotion.</div>
</div>

<h1>7. Contextual Validation: Engagement vs. Attrition &amp; Multivariate Drivers</h1>
<p>
    Consistent with preventive diagnostic theory, departed employees (<em>n</em> = 237, 16.1%) show marked upstream experience degradation relative to retained employees (<em>n</em> = 1,233): a <strong>−8.36-point Engagement Index gap</strong> (50.42 vs. 58.79, <em>t</em> = 7.14, <em>p</em> = 1.49×10⁻¹², Cohen's <em>d</em> = −0.481) and a <strong>+17.12-point Burnout Risk Score elevation</strong> (49.16 vs. 32.04, <em>t</em> = −9.68, <em>p</em> = 1.54×10⁻²¹, <em>d</em> = +0.653). Attrition drops steadily across Engagement Bands from <strong>27.2%</strong> (Low ≤41.7) to <strong>15.7%</strong> (Moderate), <strong>11.8%</strong> (High), and <strong>10.8%</strong> (Very High).
</p>
<p>
    To isolate independent causal risk factors from demographic confounders, we estimated a <strong>multivariate logistic regression</strong> predicting attrition (Pseudo-<em>R</em>² = {ctx['logit_r2']:.3f}, <strong>ROC-AUC = {ctx['logit_auc']:.3f}</strong>; Table 5, Figure 7). Crucially, holding tenure, age, and job level constant resolves the survivorship bias in promotion intervals: <strong>each additional year without promotion independently raises attrition odds by +13.5% (OR = 1.135, <em>p</em> = 0.0005)</strong>, confirming stagnation as a true structural exit driver alongside OverTime (OR = 5.015), Frequent Travel (OR = 2.400), Commute Distance (OR = 1.368 per +10 mi), Engagement Index (OR = 0.681 per +10 pts), and Work-Life Balance (OR = 0.777 per +1 pt).
</p>

<div class="tbl-cap">Table 5. Multivariate Logistic Regression Predicting Employee Attrition (ROC-AUC = 0.803)</div>
<table>
    <thead>
        <tr>
            <th>Diagnostic Predictor</th>
            <th class="num">Log-Odds Coef (β)</th>
            <th class="num">Adjusted Odds Ratio</th>
            <th class="num">95% Confidence Interval</th>
            <th class="num"><em>z</em>-statistic</th>
            <th class="num"><em>p</em>-value</th>
        </tr>
    </thead>
    <tbody>{logit_table_rows}</tbody>
</table>

<div class="fig-box">
    <img src="{figs['fig6']}" alt="Contextual Attrition Drivers and Multivariate Logistic Odds Ratios">
    <div class="fig-cap"><strong>Figure 7.</strong> (A) Stayed vs. Left profile across all normalized indices. (B) Multivariate logistic regression forest plot confirming independent effects of workload stressors, promotion stagnation, work-life balance, and composite engagement (AUC = 0.803).</div>
</div>

<h1>8. Preventive Action Framework &amp; Strategic Recommendations</h1>
<p>
    Operationalizing these insights into HR workflow requires segmenting the workforce into five actionable <strong>Manager Alert Tiers</strong> (Table 6, Figure 8). Flagging employees who meet either Low Engagement (≤ 41.7) or High Burnout Risk identifies <strong>456 priority employees (31.0% of staff) who account for 131 of all 237 attrition events (55.3%)</strong>. Including the Overload Watch tier (233 overtime staff with ≥ 2 WLB) expands diagnostic coverage to <strong>76.4% of all organizational turnover</strong>.
</p>

<div class="tbl-cap">Table 6. Five-Tier Preventive Manager Action Framework and Attrition Capture</div>
<table>
    <thead>
        <tr>
            <th>Manager Action Tier</th>
            <th class="num">Headcount</th>
            <th class="num">Workforce %</th>
            <th class="num">Eng. Index</th>
            <th class="num">Burnout Score</th>
            <th class="num">Exits</th>
            <th class="num">Attrition %</th>
            <th class="num">Share of All Exits</th>
        </tr>
    </thead>
    <tbody>{tier_table_rows}</tbody>
</table>

<div class="fig-box">
    <img src="{figs['fig8']}" alt="Manager Action Tiers and Department Exposure">
    <div class="fig-cap"><strong>Figure 8.</strong> (A) Manager Action Tiers comparing headcount share against attrition capture efficiency. (B) Department-level prevalence of low engagement, high burnout risk, and attrition.</div>
</div>

<h2>8.1 Prioritized Preventive Recommendations</h2>
<div class="two-col">
    <div class="col-half">
        <h3 style="margin-top:0; color:#EA580C;">1. Overtime &amp; Travel Caps for High-Risk Roles</h3>
        <p style="font-size:8.6pt; margin-bottom:6px;">
            Enforce mandatory manager workload audits whenever an employee triggers <strong>OverTime = Yes AND WLB ≤ 2</strong> (<em>n</em> = 126, 33.3% attrition) or <strong>OverTime + Frequent Travel</strong> (<em>n</em> = 86, 41.9% attrition). Restrict overtime assignment for Job Level 1 staff (52.6% OT attrition) and redistribute incident-response/sprint loads among Research Scientists (33.2% OT, 11.0% High Burnout).
        </p>
        <h3 style="color:#2563EB;">2. Early-Tenure &amp; Manager-Change Safeguards</h3>
        <p style="font-size:8.6pt; margin-bottom:0;">
            Institute structured 30-60-90-day workload onboarding limits (no overtime in first 6 months) and dedicated mentorship for 0–1 yr hires (34.9% attrition) and employees under a new manager (0 yrs with manager: 32.3% attrition). Guarantee at least 2–3 annual training/upskilling sessions (0-training cohort suffers 27.8% attrition).
        </p>
    </div>
    <div class="col-half">
        <h3 style="margin-top:0; color:#0D9488;">3. Mid-Career &amp; Promotion Stagnation Pathing</h3>
        <p style="font-size:8.6pt; margin-bottom:6px;">
            Mandate formal career-mobility and dual-track technical promotion reviews at Year 3 and Year 5 to eliminate the 6+ year stagnation cliff (where Job Satisfaction falls to 2.63 and each unpromoted year raises departure odds +13.5%). Address Job Level 3 compression (lowest Engagement at 55.8) via scope expansion, technical leadership badges, and equity refreshes.
        </p>
        <h3 style="color:#0F172A;">4. Targeted Compensation &amp; Hybrid Commute Relief</h3>
        <p style="font-size:8.6pt; margin-bottom:0;">
            Audit base/variable pay bands and stock option equity (Level 0 equity = 24.4% attrition vs. 7.6%–9.4% for Levels 1–2) for high-churn Sales Representatives (39.8% attrition) and Lab Technicians (23.9%). Deploy structured hybrid/remote flexibility for long-distance commuters (16–29 mi = 20.7% attrition, 10.0% High Burnout).
        </p>
    </div>
</div>

<h1>9. Conclusion &amp; Methodological Limitations</h1>
<p>
    This diagnostic investigation demonstrates that employee turnover at Palo Alto Networks is predictable well before resignation—but only when HR looks beyond aggregate satisfaction averages to measure <strong>composite engagement, cross-dimension stability, overtime-induced burnout, travel strain, and career stagnation simultaneously</strong>. Deploying the companion live <strong>Streamlit Diagnostic Dashboard</strong> empowers HR business partners and engineering/sales directors to run real-time cohort filtering, spot Silent Burnout among top performers, and trigger preventive stay conversations before institutional cybersecurity talent is lost.
</p>
<p style="font-size:8.5pt; color:#475569;">
    <strong>Limitations:</strong> Because the dataset reflects a single cross-sectional organizational snapshot rather than multi-wave longitudinal panel surveys, causal ordering is inferred through multivariate statistical controls rather than time-lagged observation. Future iterations should ingest quarterly pulse-survey time series, badge/calendar overtime telemetry, and exit-interview codes to calibrate real-time early-warning thresholds.
</p>

</body>
</html>
"""
    out_path = ROOT / "Research_Paper_Palo_Alto_Networks.pdf"
    weasyprint.HTML(string=html).write_pdf(str(out_path))
    return out_path


def build_executive_summary(ctx: Dict) -> Path:
    fig7_b64 = img_to_b64("fig7_engaged_burnout_paradox.png")
    fig8_b64 = img_to_b64("fig8_priority_intervention_segments.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Executive Summary for Government Stakeholders — Palo Alto Networks</title>
<style>
{COMMON_CSS}
@page {{
    @top-left {{ content: "EXECUTIVE BRIEF — GOVERNMENT & WORKFORCE POLICY STAKEHOLDERS"; }}
    @top-right {{ content: "PALO ALTO NETWORKS DIAGNOSTIC STUDY"; }}
}}
</style>
</head>
<body>

<div class="cover-banner">
    <div class="cover-tag">Executive Policy &amp; Workforce Resilience Brief • Government Stakeholders</div>
    <div class="cover-title">Preventive Cybersecurity Workforce Diagnostics: Burnout Mitigation, Engagement Health &amp; Talent Retention</div>
    <div class="cover-subtitle">Diagnostic Evidence and Governance Recommendations from Palo Alto Networks Workforce Analytics (N = {ctx['n']:,})</div>
    <div class="cover-meta">
        <strong>Prepared For:</strong> Public-Sector Workforce, Labor Standards &amp; Cybersecurity Resilience Stakeholders &nbsp;|&nbsp;
        <strong>Date:</strong> November 2026
    </div>
</div>

<h1>1. Strategic Mandate: Why Cybersecurity Workforce Health Is a Public-Interest Priority</h1>
<p>
    Critical infrastructure defense, federal cybersecurity modernization, and national cyber readiness depend directly on the stability and institutional continuity of specialized engineering, threat-research, and technical support teams at leading cybersecurity enterprises such as Palo Alto Networks. When experienced cybersecurity researchers, lab technicians, and frontline specialists exit prematurely due to unmonitored burnout or career stagnation, organizations face severe operational disruption, knowledge loss, and elevated contractor/onboarding burdens.
</p>
<p>
    Traditional workforce reporting tracks <strong>lagging attrition</strong>—recording separations only after institutional knowledge has departed. Commissioned in alignment with preventive human-capital governance principles, this diagnostic study establishes an <strong>early-warning workforce health framework</strong> across <strong>{ctx['n']:,} employee records</strong>, quantifying engagement vitality, psychosocial workload stress, overtime-induced burnout, and retention equity.
</p>

<div class="kpi-grid">
    <div class="kpi-card teal">
        <div class="kpi-val">{ctx['eng_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Engagement Index</div>
        <div class="kpi-sub">24.0% in Low Band (≤41.7)</div>
    </div>
    <div class="kpi-card coral">
        <div class="kpi-val">{ctx['burn_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Burnout Risk Score</div>
        <div class="kpi-sub">48.6% Med/High Risk</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-val">{ctx['wlb_14']:.2f} / 4.00</div>
        <div class="kpi-lbl">Work-Life Balance</div>
        <div class="kpi-sub">28.8% Report Poor/Fair WLB</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-val">{ctx['stab_mean']:.1f} / 100</div>
        <div class="kpi-lbl">Satisfaction Stability</div>
        <div class="kpi-sub">High cross-facet dispersion</div>
    </div>
    <div class="kpi-card navy">
        <div class="kpi-val">AUC 0.803</div>
        <div class="kpi-lbl">Diagnostic Model</div>
        <div class="kpi-sub">16.1% Org Attrition Context</div>
    </div>
</div>

<h1>2. Key Empirical Findings for Workforce Policy &amp; Oversight</h1>

<div class="callout warn">
    <div class="callout-title">Critical Diagnostic Finding — The "Engaged Burnout Paradox"</div>
    Standard engagement surveys alone cannot safeguard critical technical talent. Employees working chronic overtime (<strong>28.3% of workforce</strong>) score <em>higher</em> on average engagement (<strong>59.5 vs. 56.6</strong>, <em>p</em> = 0.0035) yet separate at nearly <strong>3× the rate (30.5% vs. 10.4%)</strong>. Even among the most highly engaged personnel (&gt;75/100), High Burnout Risk (OverTime = Yes &amp; Work-Life Balance ≤ 2) elevates turnover <strong>11-fold—from 3.0% to 33.3%</strong> (Figure 1). Proactive labor governance must track <strong>workload intensity and recovery balance</strong> alongside engagement sentiment.
</div>

<div class="fig-box">
    <img src="{fig7_b64}" alt="Engaged Burnout Paradox">
    <div class="fig-cap"><strong>Figure 1.</strong> Contextual attrition across Engagement Bands stratified by Burnout Risk Tier. Workload exhaustion systematically overrides high engagement, pushing Very High Engagement staff to 33.3% attrition and Moderate Engagement staff to 54.2% attrition.</div>
</div>

<table>
    <thead>
        <tr>
            <th>Diagnostic Risk Signal</th>
            <th>Exposed Workforce Segment</th>
            <th class="num">Cohort Size</th>
            <th class="num">Attrition / Risk Impact</th>
            <th>Statistical Evidence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Overtime &amp; WLB Strain</strong></td>
            <td>High Burnout Tier (OT=Yes &amp; WLB ≤ 2)</td>
            <td class="num">126 (8.6%)</td>
            <td class="num"><span class="badge-high">33.3% Attr.</span> (45.5% at WLB=1)</td>
            <td>χ² = 67.2, <em>p</em> &lt; 10⁻¹⁴; OT OR = 5.02×</td>
        </tr>
        <tr>
            <td><strong>Compounding Travel Load</strong></td>
            <td>OverTime = Yes + Frequent Travel</td>
            <td class="num">86 (5.9%)</td>
            <td class="num"><span class="badge-high">41.9% Attr.</span> (vs. 4.3% baseline)</td>
            <td>Travel Freq. Adjusted OR = 2.40× (<em>p</em> &lt; 0.001)</td>
        </tr>
        <tr>
            <td><strong>Early-Career Vulnerability</strong></td>
            <td>Job Level 1 w/ OT &amp; 0–1 Yr Tenure</td>
            <td class="num">156 / 215</td>
            <td class="num"><span class="badge-high">52.6% / 34.9% Attr.</span></td>
            <td>Lowest Tenure Engagement (56.1); 32.1% OT</td>
        </tr>
        <tr>
            <td><strong>R&amp;D Technical Strain</strong></td>
            <td>Research Scientists &amp; Lab Technicians</td>
            <td class="num">292 / 259</td>
            <td class="num"><span class="badge-med">11.0% / 9.3% High Burnout</span></td>
            <td>Highest Role Burnout Score (38.6) &amp; 23.9% Lab Attr.</td>
        </tr>
        <tr>
            <td><strong>Promotion Stagnation</strong></td>
            <td>6+ Years Without Promotion</td>
            <td class="num">215 (14.6%)</td>
            <td class="num"><span class="badge-med">16.3% Attr.</span> (JobSat = 2.63)</td>
            <td>+13.5% Attrition Odds / Yr Unpromoted (<em>p</em> = 0.0005)</td>
        </tr>
        <tr>
            <td><strong>Commute Distance Burden</strong></td>
            <td>Long Commute (16–29 miles)</td>
            <td class="num">329 (22.4%)</td>
            <td class="num"><span class="badge-med">20.7% Attr.</span> (10.0% High Burn)</td>
            <td>Adjusted OR = 1.37× per +10 mi (<em>p</em> = 0.0009)</td>
        </tr>
    </tbody>
</table>

<div class="fig-box">
    <img src="{fig8_b64}" alt="Manager Action Tiers and Department Exposure">
    <div class="fig-cap"><strong>Figure 2.</strong> (A) Five-tier Preventive Action Segmentation: the top three priority cohorts represent 31.0% of staff yet capture 55.3% of all organizational attrition. (B) Departmental risk profile across R&amp;D, Sales, and HR.</div>
</div>

<h1>3. Governance, Labor Standards &amp; Workforce Resilience Recommendations</h1>
<p>
    To translate these diagnostic findings into durable, auditable workforce governance suitable for public-sector oversight, STEM/cyber workforce boards, and enterprise compliance committees, we recommend five structured policy actions:
</p>

<div class="two-col">
    <div class="col-half">
        <h3 style="margin-top:0; color:#EA580C;">1. Overtime Governance &amp; Fatigue Guardrails</h3>
        <p style="font-size:8.6pt; margin-bottom:6px;">
            Establish a formal threshold cap prohibiting sustained overtime for employees scoring ≤ 2 on Work-Life Balance or classified in Frequent Travel roles. Mandate rotation schedules and compensatory recovery leave for Research Scientists (33.2% overtime prevalence) and incident-facing technical staff, targeting a <strong>≥ 35% reduction in High Burnout headcount</strong> within 12 months.
        </p>
        <h3 style="color:#2563EB;">2. Early-Career &amp; Apprenticeship Stabilization</h3>
        <p style="font-size:8.6pt; margin-bottom:0;">
            Protect early-career talent (Job Level 1 overtime staff face 52.6% attrition; 0–1 yr hires face 34.9% attrition) through onboarding overtime exemptions, structured technical mentorship, and guaranteed minimum annual training allocation (≥ 2–3 programs/year, cutting high burnout from 11.1% down toward the 1.5% benchmark seen in high-training cohorts).
        </p>
    </div>
    <div class="col-half">
        <h3 style="margin-top:0; color:#0D9488;">3. Transparent Career Progression &amp; Equity Audits</h3>
        <p style="font-size:8.6pt; margin-bottom:6px;">
            Implement mandatory 36-month promotion/progression reviews and technical dual-ladder pathways to eliminate the 6-year stagnation cliff (+13.5% exit odds per stagnant year) and mid-career Job Level 3 engagement compression (55.8/100). Pair progression audits with entry-level compensation and equity parity reviews for high-turnover roles (Sales Representatives 39.8%, Lab Technicians 23.9%).
        </p>
        <h3 style="color:#0F172A;">4. Hybrid Commute Policy &amp; Live Diagnostic Reporting</h3>
        <p style="font-size:8.6pt; margin-bottom:0;">
            Approve structured hybrid/telework accommodations for employees commuting &gt;15 miles (20.7% attrition, 10.0% High Burnout) to support workforce sustainability and green transit goals. Institutionalize quarterly executive reporting using the <strong>Live Streamlit Diagnostic Dashboard</strong>, tracking the five standardized KPIs across departments as part of enterprise ESG/human-capital governance.
        </p>
    </div>
</div>

<h1>4. Phased Implementation &amp; Accountability Roadmap</h1>
<table>
    <thead>
        <tr>
            <th>Implementation Horizon</th>
            <th>Priority Governance Action</th>
            <th>Target Cohort</th>
            <th>Measurable Success Target</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Phase 1: Immediate (Days 1–90)</strong></td>
            <td>Deploy Streamlit Early-Warning Dashboard; initiate manager stay interviews for Dual Crisis &amp; Silent Burnout cohorts; freeze Level 1 overtime overload.</td>
            <td>456 Priority Flagged Staff (126 High Burnout)</td>
            <td>100% outreach coverage; cut High Burnout attrition from 33.3% → &lt;20%.</td>
        </tr>
        <tr>
            <td><strong>Phase 2: Structural (Months 3–6)</strong></td>
            <td>Launch Frequent-Travel recovery policy; implement hybrid flexibility for 16–29 mi commuters; mandate ≥2 training hours/yr &amp; new-manager check-ins.</td>
            <td>Travel-Freq + Long Commute + New Hires</td>
            <td>Raise org Work-Life Balance Index from 2.76 → ≥2.95; reduce 0–1 yr exits by 25%.</td>
        </tr>
        <tr>
            <td><strong>Phase 3: Institutional (Months 6–12)</strong></td>
            <td>Establish 3-year career progression audits for Job Level 3 &amp; 6+ yr stagnant staff; integrate quarterly KPI scorecard into Board/Stakeholder governance.</td>
            <td>Enterprise-Wide (N = {ctx['n']:,}+)</td>
            <td>Lift Engagement Index from 57.4 → ≥63.0; reduce overall attrition below 12.0%.</td>
        </tr>
    </tbody>
</table>

</body>
</html>
"""
    out_path = ROOT / "Executive_Summary_Government_Stakeholders.pdf"
    weasyprint.HTML(string=html).write_pdf(str(out_path))
    return out_path


def render_pdf_previews(pdf_path: Path, prefix: str) -> list[Path]:
    import pymupdf
    prev_dir = ROOT / "outputs" / "pdf_previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=140)
        p = prev_dir / f"{prefix}_p{i+1}.png"
        pix.save(str(p))
        paths.append(p)
    print(f"{pdf_path.name}: {len(doc)} pages rendered to {prev_dir}")
    return paths


if __name__ == "__main__":
    df, _ = load_diagnostic_data()
    ctx = compute_report_context(df)
    rp = build_research_paper(ctx)
    es = build_executive_summary(ctx)
    render_pdf_previews(rp, "research")
    render_pdf_previews(es, "exec")
