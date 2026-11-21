"""
Publication-grade Matplotlib/Seaborn figure generator for
Palo Alto Networks Employee Engagement, Satisfaction, and Burnout Diagnostic Analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from data_pipeline import load_diagnostic_data

FIG_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

# Cohesive executive color palette
NAVY = "#0F172A"
SLATE = "#475569"
TEAL = "#0D9488"
BLUE = "#2563EB"
CORAL = "#EA580C"
RED = "#DC2626"
AMBER = "#D97706"
BG = "#FFFFFF"
GRID = "#E2E8F0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def make_fig1(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": [1.05, 1.1], "wspace": 0.28})

    stayed = df[df["Attrition"] == 0]["Engagement_Index"]
    left = df[df["Attrition"] == 1]["Engagement_Index"]
    bins = np.linspace(5, 100, 16)

    ax1.hist(stayed, bins=bins, density=True, alpha=0.35, color=TEAL, edgecolor=TEAL,
             label=f"Stayed (n={len(stayed)}, Mean={stayed.mean():.1f})")
    ax1.hist(left, bins=bins, density=True, alpha=0.40, color=CORAL, edgecolor=CORAL,
             label=f"Left (n={len(left)}, Mean={left.mean():.1f})")
    sns.kdeplot(stayed, ax=ax1, color=TEAL, lw=2.3, bw_adjust=1.15)
    sns.kdeplot(left, ax=ax1, color=CORAL, lw=2.3, bw_adjust=1.15)

    ax1.axvline(df["Engagement_Index"].mean(), color=NAVY, ls="--", lw=1.6,
                label=f"Org Mean = {df['Engagement_Index'].mean():.1f}")
    ax1.axvline(41.67, color=RED, ls=":", lw=1.6,
                label="Low-Engagement Threshold (41.7)")
    ax1.set_title("A. Engagement Index Distribution (Stayed vs. Left)", pad=12, color=NAVY)
    ax1.set_xlabel("Composite Engagement Index (0–100 Scale)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 0.038)
    ax1.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5, loc="upper left")

    dims = [
        ("JobInvolvement", "Job Involvement"),
        ("JobSatisfaction", "Job Satisfaction"),
        ("EnvironmentSatisfaction", "Environment Satisfaction"),
        ("RelationshipSatisfaction", "Relationship Satisfaction"),
        ("WorkLifeBalance", "Work-Life Balance"),
    ]
    pct_matrix, labels = [], []
    for col, lbl in dims:
        vc = df[col].value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0) * 100
        pct_matrix.append(vc.values)
        labels.append(f"{lbl}\n(μ = {df[col].mean():.2f})")

    pct_arr = np.array(pct_matrix)
    colors = ["#DC2626", "#F59E0B", "#3B82F6", "#0D9488"]
    level_names = ["1 (Low / Poor)", "2 (Moderate)", "3 (High)", "4 (Very High / Exc.)"]
    y_pos = np.arange(len(labels))
    left_cum = np.zeros(len(labels))

    for idx in range(4):
        vals = pct_arr[:, idx]
        bars = ax2.barh(y_pos, vals, left=left_cum, height=0.58, color=colors[idx], edgecolor="white", label=level_names[idx])
        for b, v in zip(bars, vals):
            if v >= 7.5:
                ax2.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2,
                         f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        left_cum += vals

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of Workforce (%)")
    ax2.set_title("B. Ordinal Score Distribution Across Core Dimensions (N=1,470)", pad=12, color=NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=4, frameon=False, fontsize=8.5)

    fig.savefig(FIG_DIR / "fig1_engagement_overview.png")
    plt.close(fig)


def make_fig2(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [0.95, 1.15], "wspace": 0.30})

    order = ["Low", "Medium", "High"]
    tier_stats = df.groupby("Burnout_Risk_Level").agg(
        n=("Attrition", "count"),
        pct=("Attrition", lambda s: len(s) / len(df) * 100),
        attr=("Attrition", lambda s: s.mean() * 100),
    ).reindex(order)

    x = np.arange(len(order))
    w = 0.36
    b1 = ax1.bar(x - w / 2, tier_stats["pct"], width=w, color=BLUE, label="Workforce Share (%)")
    b2 = ax1.bar(x + w / 2, tier_stats["attr"], width=w, color=CORAL, label="Contextual Attrition Rate (%)")

    for bar, n_val in zip(b1, tier_stats["n"]):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%\n(n={n_val})",
                 ha="center", va="bottom", fontsize=8.5, color=NAVY, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, color=CORAL, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Low Risk\n(No OT & WLB ≥ 3)", "Medium Risk\n(OT or Low WLB)", "High Risk\n(OT & WLB ≤ 2)"], fontsize=9)
    ax1.set_ylim(0, 62)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("A. Burnout Risk Tiers: Share vs. Attrition\n(χ² = 67.2, p < 1e-14)", pad=10, color=NAVY)
    ax1.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5, loc="upper right")

    piv_attr = df.pivot_table(index="OverTime", columns="WorkLifeBalance", values="Attrition", aggfunc="mean").loc[["No", "Yes"], [1, 2, 3, 4]] * 100
    piv_cnt = df.pivot_table(index="OverTime", columns="WorkLifeBalance", values="Attrition", aggfunc="count").loc[["No", "Yes"], [1, 2, 3, 4]]

    annot = piv_attr.copy().astype(str)
    for r in piv_attr.index:
        for c in piv_attr.columns:
            annot.loc[r, c] = f"{piv_attr.loc[r, c]:.1f}%\n(n={piv_cnt.loc[r, c]})"

    sns.heatmap(
        piv_attr,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Contextual Attrition Rate (%)"},
        linewidths=1.5,
        linecolor="white",
        ax=ax2,
        vmin=0,
        vmax=48,
        annot_kws={"size": 9.5, "weight": "bold"},
    )
    rect = mpatches.Rectangle((0.02, 1.02), 1.96, 0.96, fill=False, edgecolor=NAVY, lw=2.8, ls="--")
    ax2.add_patch(rect)
    ax2.text(1.0, 1.09, "HIGH BURNOUT ZONE (n=126, 33.3% Attr.)",
             ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.22", facecolor=NAVY, alpha=0.85))

    ax2.set_title("B. OverTime × Work-Life Balance Matrix\n(Attrition % & Headcount)", pad=10, color=NAVY)
    ax2.set_xlabel("Work-Life Balance Rating (1 = Poor, 4 = Excellent)")
    ax2.set_ylabel("OverTime Status")
    ax2.set_yticklabels(["No OverTime", "OverTime = Yes"], rotation=0, fontsize=9.5)
    ax2.set_xticklabels(["1 (Poor)", "2 (Fair)", "3 (Good)", "4 (Excellent)"], fontsize=9)

    fig.savefig(FIG_DIR / "fig2_burnout_risk_matrix.png")
    plt.close(fig)


def make_fig3(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"wspace": 0.34})

    combos = [
        ("No", "Non-Travel", "No OT\nNon-Travel"),
        ("No", "Travel_Rarely", "No OT\nRare Travel"),
        ("No", "Travel_Frequently", "No OT\nFreq Travel"),
        ("Yes", "Non-Travel", "OT=Yes\nNon-Travel"),
        ("Yes", "Travel_Rarely", "OT=Yes\nRare Travel"),
        ("Yes", "Travel_Frequently", "OT=Yes\nFreq Travel"),
    ]
    labels, attr_vals, eng_vals = [], [], []
    for ot, tr, lbl in combos:
        sub = df[(df["OverTime"] == ot) & (df["BusinessTravel"] == tr)]
        labels.append(f"{lbl}\n(n={len(sub)})")
        attr_vals.append(sub["Attrition"].mean() * 100)
        eng_vals.append(sub["Engagement_Index"].mean())

    x = np.arange(len(labels))
    bar_colors = [TEAL, TEAL, "#3B82F6", AMBER, CORAL, RED]
    bars = ax1.bar(x, attr_vals, width=0.52, color=bar_colors, alpha=0.88, label="Attrition Rate (%)")
    for idx, (b, v) in enumerate(zip(bars, attr_vals)):
        if v > 25:
            ax1.text(b.get_x() + b.get_width() / 2, v - 5.0, f"{v:.1f}%", ha="center", va="top", fontsize=8.8, fontweight="bold", color="white")
        else:
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY)

    ax1_r = ax1.twinx()
    ax1_r.plot(x, eng_vals, color=NAVY, marker="o", lw=2.2, ms=6.5, label="Mean Engagement Index (0–100)")
    for xi, ev in zip(x, eng_vals):
        ax1_r.text(xi, ev + 1.5, f"{ev:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=NAVY,
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 56)
    ax1_r.set_ylim(38, 67)
    ax1.set_ylabel("Contextual Attrition Rate (%)", color=CORAL, fontweight="bold")
    ax1_r.set_ylabel("Mean Engagement Index (0–100)", color=NAVY, fontweight="bold")
    ax1.set_title("A. Compounding Stress: OverTime × Travel Intensity", pad=12, color=NAVY)
    ax1.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1_r.spines["top"].set_visible(False)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.2, loc="upper left")

    cm = df.groupby("Commute_Band", observed=False).agg(
        n=("Attrition", "count"),
        attr=("Attrition", lambda s: s.mean() * 100),
        high_burn=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
    )
    xc = np.arange(len(cm))
    w = 0.34
    b_a = ax2.bar(xc - w / 2, cm["attr"], width=w, color=CORAL, label="Attrition Rate (%)")
    b_b = ax2.bar(xc + w / 2, cm["high_burn"], width=w, color=BLUE, label="High Burnout Risk (%)")

    for b in b_a:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=CORAL)
    for b in b_b:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=BLUE)

    ax2.set_xticks(xc)
    ax2.set_xticklabels([f"{idx}\n(n={n_val})" for idx, n_val in zip(cm.index, cm["n"])], fontsize=9.5)
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("B. Commute Distance vs. Attrition & High Burnout\n(Distance-Attrition r = 0.078, p = 0.003)", pad=10, color=NAVY)
    ax2.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5, loc="upper left")

    fig.savefig(FIG_DIR / "fig3_workload_stress_drivers.png")
    plt.close(fig)


def make_fig4(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.50})

    role_df = df.groupby("JobRole").agg(
        n=("Attrition", "count"),
        eng=("Engagement_Index", "mean"),
        burn=("Burnout_Risk_Score", "mean"),
        high_burn=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
        low_eng=("Engagement_Index", lambda s: (s <= 41.67).mean() * 100),
        attr=("Attrition", lambda s: s.mean() * 100),
    ).reset_index()

    sc = ax1.scatter(
        role_df["eng"],
        role_df["burn"],
        s=role_df["n"] * 2.2,
        c=role_df["attr"],
        cmap="YlOrRd",
        alpha=0.88,
        edgecolors=NAVY,
        linewidths=1.2,
        vmin=2,
        vmax=40,
    )
    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("Contextual Attrition Rate (%)", fontsize=9)

    ax1.axvline(df["Engagement_Index"].mean(), color=SLATE, ls="--", lw=1.0)
    ax1.axhline(df["Burnout_Risk_Score"].mean(), color=SLATE, ls="--", lw=1.0)

    label_pos = {
        "Research Director": (54.40, 35.15),
        "Sales Representative": (54.30, 30.75),
        "Laboratory Technician": (56.95, 31.55),
        "Human Resources": (57.45, 29.35),
        "Sales Executive": (54.60, 36.85),
        "Manager": (58.85, 35.75),
        "Healthcare Representative": (57.50, 37.35),
        "Research Scientist": (55.70, 39.70),
        "Manufacturing Director": (58.95, 32.35),
    }
    for _, row in role_df.iterrows():
        tx, ty = label_pos[row["JobRole"]]
        ax1.annotate(
            f"{row['JobRole']}\n(n={int(row['n'])}, {row['attr']:.1f}%)",
            xy=(row["eng"], row["burn"]),
            xytext=(tx, ty),
            fontsize=7.8,
            color=NAVY,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.8, shrinkA=2, shrinkB=5),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=GRID, alpha=0.9),
        )

    ax1.set_xlim(54.2, 61.3)
    ax1.set_ylim(28.8, 41.2)
    ax1.set_xlabel("Mean Engagement Index (0–100)")
    ax1.set_ylabel("Mean Burnout Risk Score (0–100)")
    ax1.set_title("A. Job Role Risk Landscape (Size = Headcount, Color = Attrition %)", pad=12, color=NAVY)
    ax1.grid(color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    role_sorted = role_df.sort_values("attr", ascending=True)
    yr = np.arange(len(role_sorted))
    h = 0.36
    ax2.barh(yr + h / 2, role_sorted["low_eng"], height=h, color=BLUE, label="Low Engagement (≤41.7) %")
    ax2.barh(yr - h / 2, role_sorted["high_burn"], height=h, color=CORAL, label="High Burnout Risk %")

    for i, (_, r) in enumerate(role_sorted.iterrows()):
        ax2.text(r["low_eng"] + 0.5, i + h / 2, f"{r['low_eng']:.1f}%", va="center", fontsize=8, color=BLUE, fontweight="bold")
        ax2.text(r["high_burn"] + 0.5, i - h / 2, f"{r['high_burn']:.1f}%", va="center", fontsize=8, color=CORAL, fontweight="bold")

    ax2.set_yticks(yr)
    ax2.set_yticklabels([f"{r['JobRole']} ({r['attr']:.1f}% Attr)" for _, r in role_sorted.iterrows()], fontsize=8.5)
    ax2.set_xlim(0, 38)
    ax2.set_xlabel("Share of Role Workforce (%)")
    ax2.set_title("B. Disengagement & High Burnout Prevalence by Role", pad=12, color=NAVY)
    ax2.grid(axis="x", color=GRID, ls="--", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False, fontsize=8.5)

    fig.savefig(FIG_DIR / "fig4_role_department_diagnostics.png")
    plt.close(fig)


def make_fig5(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"wspace": 0.34})

    jl_all = df.groupby("JobLevel").agg(
        n=("Attrition", "count"),
        eng=("Engagement_Index", "mean"),
    )
    jl_ot = df[df["OverTime"] == "Yes"].groupby("JobLevel")["Attrition"].mean() * 100
    jl_no = df[df["OverTime"] == "No"].groupby("JobLevel")["Attrition"].mean() * 100

    x = np.arange(1, 6)
    w = 0.28
    ax1.bar(x - w / 2, jl_no.values, width=w, color=TEAL, label="Attrition: No OverTime (%)")
    ax1.bar(x + w / 2, jl_ot.values, width=w, color=CORAL, label="Attrition: OverTime = Yes (%)")
    for xi, v in zip(x + w / 2, jl_ot.values):
        ax1.text(xi, v + 0.9, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=CORAL)
    for xi, v in zip(x - w / 2, jl_no.values):
        ax1.text(xi, v + 0.9, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=TEAL)

    ax1_r = ax1.twinx()
    ax1_r.plot(x, jl_all["eng"].values, color=NAVY, marker="D", lw=2.2, ms=6.5, label="Mean Engagement Index")
    for xi, ev in zip(x, jl_all["eng"].values):
        ax1_r.text(xi, ev + 1.5, f"{ev:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY,
                   bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.85))

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Level {lvl}\n(n={jl_all.loc[lvl, 'n']})" for lvl in x], fontsize=9)
    ax1.set_ylim(0, 62)
    ax1_r.set_ylim(41, 66)
    ax1.set_ylabel("Attrition Rate (%)", fontweight="bold", color=CORAL)
    ax1_r.set_ylabel("Mean Engagement Index (0–100)", fontweight="bold", color=NAVY)
    ax1.set_title("A. Career Stage: Job Level Attrition & Engagement Dip", pad=12, color=NAVY)
    ax1.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1_r.spines["top"].set_visible(False)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, frameon=True, facecolor="white", edgecolor=GRID, fontsize=8, loc="upper right")

    pg = df.groupby("Promo_Gap_Band", observed=False).agg(
        n=("Attrition", "count"),
        eng=("Engagement_Index", "mean"),
        js=("JobSatisfaction", lambda s: (s.mean() - 1) / 3 * 100),
        attr=("Attrition", lambda s: s.mean() * 100),
    )
    xp = np.arange(len(pg))
    bars = ax2.bar(xp, pg["attr"], width=0.44, color=[AMBER, BLUE, TEAL, RED], alpha=0.88, label="Attrition Rate (%)")
    for b, v in zip(bars, pg["attr"]):
        ax2.text(b.get_x() + b.get_width() / 2, v - 1.5, f"{v:.1f}%", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    ax2_r = ax2.twinx()
    ax2_r.plot(xp, pg["eng"], color=NAVY, marker="o", lw=2.2, ms=6.5, label="Engagement Index (0–100)")
    ax2_r.plot(xp, pg["js"], color=BLUE, marker="s", ls="--", lw=1.8, ms=5.5, label="Job Satisfaction (0–100)")

    for xi, (ev, jv) in enumerate(zip(pg["eng"], pg["js"])):
        if ev >= jv:
            ax2_r.text(xi, ev + 1.0, f"{ev:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=NAVY)
            ax2_r.text(xi, jv - 1.1, f"{jv:.1f}", ha="center", va="top", fontsize=8.2, fontweight="bold", color=BLUE)
        else:
            ax2_r.text(xi, jv + 1.0, f"{jv:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=BLUE)
            ax2_r.text(xi, ev - 1.1, f"{ev:.1f}", ha="center", va="top", fontsize=8.2, fontweight="bold", color=NAVY)

    ax2.set_xticks(xp)
    ax2.set_xticklabels([f"{idx}\n(n={n_v})" for idx, n_v in zip(pg.index, pg["n"])], fontsize=8.5)
    ax2.set_ylim(0, 44)
    ax2_r.set_ylim(41, 68)
    ax2.set_ylabel("Attrition Rate (%)", fontweight="bold", color=NAVY)
    ax2_r.set_ylabel("Normalized Score (0–100)", fontweight="bold", color=BLUE)
    ax2.set_title("B. Promotion Gap: Stagnation Drop at 6+ Years", pad=12, color=NAVY)
    ax2.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2_r.spines["top"].set_visible(False)
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, frameon=True, facecolor="white", edgecolor=GRID, fontsize=8, loc="upper right")

    fig.savefig(FIG_DIR / "fig5_career_stage_stagnation.png")
    plt.close(fig)


def make_fig6(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.42})

    metrics = [
        ("Engagement_Index", "Composite Engagement Index"),
        ("JobInvolvement_Norm", "Job Involvement"),
        ("JobSatisfaction_Norm", "Job Satisfaction"),
        ("EnvironmentSatisfaction_Norm", "Environment Satisfaction"),
        ("RelationshipSatisfaction_Norm", "Relationship Satisfaction"),
        ("WorkLifeBalance_Norm", "Work-Life Balance"),
        ("Satisfaction_Stability_Score", "Satisfaction Stability"),
        ("Burnout_Risk_Score", "Burnout Risk Score"),
    ]
    stayed = df[df["Attrition"] == 0]
    left = df[df["Attrition"] == 1]

    y = np.arange(len(metrics))
    s_vals = [stayed[c].mean() for c, _ in metrics]
    l_vals = [left[c].mean() for c, _ in metrics]
    lbls = [lbl for _, lbl in metrics]

    h = 0.36
    ax1.barh(y - h / 2, s_vals, height=h, color=TEAL, label=f"Stayed (n={len(stayed)})")
    ax1.barh(y + h / 2, l_vals, height=h, color=CORAL, label=f"Left (n={len(left)})")

    for i, (sv, lv) in enumerate(zip(s_vals, l_vals)):
        ax1.text(sv + 0.8, i - h / 2, f"{sv:.1f}", va="center", fontsize=8.2, color=TEAL, fontweight="bold")
        ax1.text(lv + 0.8, i + h / 2, f"{lv:.1f}", va="center", fontsize=8.2, color=CORAL, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(lbls, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 72)
    ax1.set_xlabel("Mean Score (0–100 Scale)")
    ax1.set_title("A. Contextual Profile: Stayed vs. Left (0–100 Scale)", pad=12, color=NAVY)
    ax1.grid(axis="x", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.14), ncol=2, frameon=False, fontsize=8.8)

    X = pd.DataFrame({
        "const": 1.0,
        "OverTime = Yes": df["OverTime_Flag"],
        "Travel Frequently": (df["BusinessTravel"] == "Travel_Frequently").astype(int),
        "Commute (+10 miles)": df["DistanceFromHome"] / 10.0,
        "Years Since Promotion (+1 yr)": df["YearsSinceLastPromotion"],
        "Tenure at Company (+1 yr)": df["YearsAtCompany"],
        "Work-Life Balance (+1 pt)": df["WorkLifeBalance"],
        "Age (+10 years)": df["Age"] / 10.0,
        "Job Level (+1 level)": df["JobLevel"],
        "Engagement Index (+10 pts)": df["Engagement_Index"] / 10.0,
    })
    logit = sm.Logit(df["Attrition"], X).fit(disp=0)
    params = logit.params.drop("const")
    conf = logit.conf_int().drop("const")
    pvals = logit.pvalues.drop("const")

    or_df = pd.DataFrame({
        "OR": np.exp(params),
        "CI_low": np.exp(conf[0]),
        "CI_high": np.exp(conf[1]),
        "p": pvals,
    }).sort_values("OR", ascending=True)

    yf = np.arange(len(or_df))
    ax2.axvline(1.0, color=NAVY, ls="--", lw=1.3)
    for i, (_, row) in enumerate(or_df.iterrows()):
        c = TEAL if row["OR"] < 1.0 else CORAL
        ax2.errorbar(
            row["OR"], i,
            xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
            fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=4, ms=7,
        )
        p_txt = "p<0.001" if row["p"] < 0.001 else f"p={row['p']:.3f}"
        ax2.text(row["CI_high"] + 0.14, i, f"OR={row['OR']:.2f} ({p_txt})", va="center", fontsize=8.2, color=NAVY, fontweight="bold")

    ax2.set_yticks(yf)
    ax2.set_yticklabels(or_df.index, fontsize=9)
    ax2.set_xlim(0.35, 8.6)
    ax2.set_xlabel("Adjusted Odds Ratio for Attrition (95% CI, AUC = 0.803)")
    ax2.set_title("B. Independent Attrition Drivers (Multivariate Logit)", pad=12, color=NAVY)
    ax2.grid(axis="x", color=GRID, ls="--", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig6_contextual_attrition_drivers.png")
    plt.close(fig)


def make_fig7(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    band_order = ["Low (≤41.7)", "Moderate (41.8–58.3)", "High (58.4–75.0)", "Very High (>75.0)"]
    burn_order = ["Low", "Medium", "High"]

    piv_attr = df.pivot_table(index="Engagement_Band", columns="Burnout_Risk_Level", values="Attrition", aggfunc="mean", observed=False).reindex(index=band_order, columns=burn_order) * 100
    piv_cnt = df.pivot_table(index="Engagement_Band", columns="Burnout_Risk_Level", values="Attrition", aggfunc="count", observed=False).reindex(index=band_order, columns=burn_order)

    x = np.arange(len(band_order))
    w = 0.25
    colors = [TEAL, AMBER, RED]

    for idx, b_lvl in enumerate(burn_order):
        vals = piv_attr[b_lvl].values
        cnts = piv_cnt[b_lvl].values
        bars = ax.bar(x + (idx - 1) * w, vals, width=w, color=colors[idx], label=f"{b_lvl} Burnout Risk")
        for bar, v, c in zip(bars, vals, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{v:.1f}%\n(n={int(c)})",
                    ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=NAVY)

    ax.annotate(
        "Engaged Burnout Paradox:\nEven Very High Engagement staff\nexperience 11× attrition under\nHigh Burnout (33.3% vs. 3.0%)",
        xy=(3 + w - 0.11, 30.0),
        xytext=(2.05, 43.0),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, connectionstyle="arc3,rad=-0.12"),
        fontsize=8.8,
        fontweight="bold",
        color=RED,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor=RED, lw=1.0),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Engagement:\n{b}" for b in band_order], fontsize=9.5)
    ax.set_ylim(0, 63)
    ax.set_ylabel("Contextual Attrition Rate (%)")
    ax.set_title("The Engaged Burnout Paradox: Engagement Band × Burnout Risk Interaction", pad=12, color=NAVY)
    ax.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=9, loc="upper left")

    fig.savefig(FIG_DIR / "fig7_engaged_burnout_paradox.png")
    plt.close(fig)


def make_fig8(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.28})

    tier_order = [
        "Dual Crisis (Low Eng + High Burnout)",
        "Silent Burnout Risk (High Burnout)",
        "Disengaged Alert (Low Engagement)",
        "Overload Watch (OverTime)",
        "Stable / Monitor",
    ]
    t_df = df.groupby("Priority_Alert_Tier").agg(
        n=("Attrition", "count"),
        attr_n=("Attrition", "sum"),
        attr_rate=("Attrition", lambda s: s.mean() * 100),
    ).reindex(tier_order)
    t_df["wf_share"] = t_df["n"] / len(df) * 100
    t_df["attr_share"] = t_df["attr_n"] / df["Attrition"].sum() * 100

    y = np.arange(len(tier_order))
    h = 0.36
    ax1.barh(y - h / 2, t_df["wf_share"], height=h, color=BLUE, label="Share of Workforce (%)")
    ax1.barh(y + h / 2, t_df["attr_share"], height=h, color=CORAL, label="Share of Total Attrition (%)")

    for i, (_, r) in enumerate(t_df.iterrows()):
        ax1.text(r["wf_share"] + 0.8, i - h / 2, f"{r['wf_share']:.1f}% (n={int(r['n'])})", va="center", fontsize=8.3, color=BLUE, fontweight="bold")
        ax1.text(r["attr_share"] + 0.8, i + h / 2, f"{r['attr_share']:.1f}% ({r['attr_rate']:.1f}% rate)", va="center", fontsize=8.3, color=CORAL, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(tier_order, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 76)
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("A. Manager Action Tiers: Workforce Share vs. Attrition Capture", pad=12, color=NAVY)
    ax1.grid(axis="x", color=GRID, ls="--", lw=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.5, loc="upper right")

    dept_df = df.groupby("Department").agg(
        n=("Attrition", "count"),
        high_burn=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
        low_eng=("Engagement_Index", lambda s: (s <= 41.67).mean() * 100),
        attr=("Attrition", lambda s: s.mean() * 100),
    ).reindex(["Research & Development", "Sales", "Human Resources"])

    xd = np.arange(len(dept_df))
    wd = 0.25
    ax2.bar(xd - wd, dept_df["low_eng"], width=wd, color=BLUE, label="Low Engagement (≤41.7) %")
    ax2.bar(xd, dept_df["high_burn"], width=wd, color=RED, label="High Burnout Risk %")
    ax2.bar(xd + wd, dept_df["attr"], width=wd, color=AMBER, label="Attrition Rate %")

    for i, (_, r) in enumerate(dept_df.iterrows()):
        ax2.text(i - wd, r["low_eng"] + 0.5, f"{r['low_eng']:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=BLUE)
        ax2.text(i, r["high_burn"] + 0.5, f"{r['high_burn']:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=RED)
        ax2.text(i + wd, r["attr"] + 0.5, f"{r['attr']:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=AMBER)

    ax2.set_xticks(xd)
    ax2.set_xticklabels([f"{idx}\n(n={int(r['n'])})" for idx, r in dept_df.iterrows()], fontsize=9)
    ax2.set_ylim(0, 32)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("B. Department-Level Diagnostic Exposure", pad=12, color=NAVY)
    ax2.grid(axis="y", color=GRID, ls="--", lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.3, loc="upper right")

    fig.savefig(FIG_DIR / "fig8_priority_intervention_segments.png")
    plt.close(fig)


def generate_all_charts() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_diagnostic_data()
    make_fig1(df)
    make_fig2(df)
    make_fig3(df)
    make_fig4(df)
    make_fig5(df)
    make_fig6(df)
    make_fig7(df)
    make_fig8(df)
    print(f"Regenerated 8 polished figures in {FIG_DIR}")


if __name__ == "__main__":
    generate_all_charts()
