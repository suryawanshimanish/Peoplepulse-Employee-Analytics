"""
Palo Alto Networks — Employee Engagement, Satisfaction, and Burnout Diagnostic Dashboard
Interactive Streamlit Web Application driven strictly by uploaded CSV data,
with Light, Dark, and System Appearance Modes.
"""

from __future__ import annotations

import html as html_lib
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import process_uploaded_dataframe

# Page configuration
st.set_page_config(
    page_title="Palo Alto Networks | Engagement & Burnout Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== THEME & APPEARANCE MODE ====================
if "theme_mode_picker" not in st.session_state:
    st.session_state["theme_mode_picker"] = "System"

# Top utility bar with Appearance Mode switcher (always visible even if sidebar is collapsed)
top_spacer, top_ctrl = st.columns([2.8, 1.5], vertical_alignment="center")
with top_ctrl:
    st.segmented_control(
        "Appearance Mode",
        options=["System", "Light", "Dark"],
        selection_mode="single",
        format_func=lambda m: {"System": "🖥️ System", "Light": "☀️ Light", "Dark": "🌙 Dark"}[m],
        label_visibility="collapsed",
        key="theme_mode_picker",
        width="stretch",
    )

theme_mode = st.session_state["theme_mode_picker"] or "System"
raw_context_theme = getattr(getattr(st.context, "theme", None), "type", None)
native_theme_type = raw_context_theme or "light"
if theme_mode == "System":
    effective_dark = native_theme_type == "dark"
else:
    effective_dark = theme_mode == "Dark"

# Theme-aware chart color tokens
PRIMARY = "#818CF8" if effective_dark else "#4F46E5"
TEAL = "#2DD4BF" if effective_dark else "#0D9488"
BLUE = "#60A5FA" if effective_dark else "#2563EB"
CORAL = "#FB923C" if effective_dark else "#EA580C"
RED = "#F87171" if effective_dark else "#DC2626"
AMBER = "#FBBF24" if effective_dark else "#D97706"
SLATE = "#94A3B8" if effective_dark else "#475569"
LINE_INK = "#E2E8F0" if effective_dark else "#1E293B"
GRID_COLOR = "#263449" if effective_dark else "#E2E8F0"
TEXT_COLOR = "#F1F5F9" if effective_dark else "#0F172A"
PLOTLY_TEMPLATE = "plotly_dark" if effective_dark else "plotly_white"

LIGHT_VARS = """
    --bg-app: #F8FAFC;
    --bg-sidebar: #F1F5F9;
    --bg-card: #FFFFFF;
    --bg-input: #FFFFFF;
    --border-color: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --hero-bg: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 52%, #DBEAFE 100%);
    --hero-border: #C7D2FE;
    --hero-accent: #4F46E5;
    --hero-tag: #3730A3;
    --hero-title: #0F172A;
    --hero-sub: #334155;
    --insight-bg: #EFF6FF;
    --insight-border: #2563EB;
    --warn-bg: #FEF2F2;
    --warn-border: #DC2626;
    --table-head-bg: #F1F5F9;
    --table-head-text: #0F172A;
    --table-stripe: #F8FAFC;
    --pill-bg: #E2E8F0;
    --accent-main: #4F46E5;
"""

DARK_VARS = """
    --bg-app: #0B1120;
    --bg-sidebar: #111827;
    --bg-card: #162032;
    --bg-input: #1E293B;
    --border-color: #263449;
    --text-primary: #F1F5F9;
    --text-secondary: #CBD5E1;
    --text-muted: #94A3B8;
    --hero-bg: linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #1E293B 100%);
    --hero-border: rgba(129, 140, 248, 0.30);
    --hero-accent: #6366F1;
    --hero-tag: #A5B4FC;
    --hero-title: #F8FAFC;
    --hero-sub: #CBD5E1;
    --insight-bg: rgba(37, 99, 235, 0.14);
    --insight-border: #60A5FA;
    --warn-bg: rgba(248, 113, 113, 0.14);
    --warn-border: #F87171;
    --table-head-bg: #1E293B;
    --table-head-text: #F1F5F9;
    --table-stripe: rgba(30, 41, 59, 0.48);
    --pill-bg: #1E293B;
    --accent-main: #818CF8;
"""

if theme_mode == "System":
    if raw_context_theme == "dark":
        vars_block = f":root {{ {DARK_VARS} }}"
    elif raw_context_theme == "light":
        vars_block = f":root {{ {LIGHT_VARS} }}"
    else:
        vars_block = f"""
        :root {{ {LIGHT_VARS} }}
        @media (prefers-color-scheme: dark) {{
            :root {{ {DARK_VARS} }}
        }}
        """
elif theme_mode == "Dark":
    vars_block = f":root {{ {DARK_VARS} }}"
else:
    vars_block = f":root {{ {LIGHT_VARS} }}"

# Explicit surface overrides whenever user picks a specific mode
surface_overrides = ""
if theme_mode != "System":
    surface_overrides = """
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
        background-color: var(--bg-sidebar) !important;
        color: var(--text-primary) !important;
        border-right-color: var(--border-color) !important;
    }
    [data-testid="stSidebar"] *, .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6, label, [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] p, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--text-primary) !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        background-color: var(--bg-input) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    [data-baseweb="select"] span, [data-baseweb="select"] input {
        color: var(--text-primary) !important;
    }
    [data-baseweb="tag"], [data-baseweb="tag"] * {
        background-color: var(--accent-main) !important;
        color: #FFFFFF !important;
    }
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="menu"], [role="listbox"], ul[data-baseweb="menu"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    [role="option"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    [role="option"]:hover, [aria-selected="true"][role="option"] {
        background-color: var(--table-head-bg) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--bg-input) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"] {
        color: var(--text-primary) !important;
    }
    [data-testid="stAlert"] {
        background-color: var(--insight-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    [data-testid="stAlert"] * {
        color: var(--text-primary) !important;
    }
    button[role="tab"] {
        color: var(--text-muted) !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: var(--accent-main) !important;
        border-bottom-color: var(--accent-main) !important;
    }
    [data-testid="stDownloadButton"] button, [data-testid="stBaseButton-secondary"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    [data-testid="stBaseButton-segmented_control"],
    [data-testid="stBaseButton-segmented_controlActive"] {
        border-color: var(--border-color) !important;
    }
    [data-testid="stBaseButton-segmented_control"] {
        background-color: var(--bg-card) !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stBaseButton-segmented_controlActive"] {
        background-color: var(--accent-main) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stBaseButton-segmented_controlActive"] p {
        color: #FFFFFF !important;
    }
    """

st.markdown(
    f"""
    <style>
    {vars_block}
    {surface_overrides}
    .block-container {{ padding-top: 1.0rem; padding-bottom: 2rem; }}
    .hero-banner {{
        background: var(--hero-bg);
        color: var(--hero-title);
        padding: 1.35rem 1.7rem;
        border-radius: 12px;
        border: 1px solid var(--hero-border);
        border-left: 5px solid var(--hero-accent);
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.0rem;
    }}
    .hero-tag {{
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--hero-tag);
        margin-bottom: 0.25rem;
    }}
    .hero-title {{
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
        color: var(--hero-title);
    }}
    .hero-sub {{
        font-size: 0.92rem;
        color: var(--hero-sub);
        margin-top: 0.35rem;
    }}
    .kpi-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-top: 4px solid #4F46E5;
        border-radius: 9px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
        height: 100%;
    }}
    .kpi-card.teal {{ border-top-color: #0D9488; }}
    .kpi-card.coral {{ border-top-color: #EA580C; }}
    .kpi-card.amber {{ border-top-color: #D97706; }}
    .kpi-card.indigo {{ border-top-color: #6366F1; }}
    .kpi-card.red {{ border-top-color: #DC2626; }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }}
    .kpi-value {{
        font-size: 1.50rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0.15rem 0;
    }}
    .kpi-delta {{
        font-size: 0.78rem;
        color: var(--text-muted);
    }}
    .kpi-delta b {{
        color: var(--text-primary);
    }}
    .insight-box {{
        background: var(--insight-bg);
        border-left: 4px solid var(--insight-border);
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
    }}
    .warn-box {{
        background: var(--warn-bg);
        border-left: 4px solid var(--warn-border);
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
    }}
    .themed-table-wrap {{
        max-height: 335px;
        overflow: auto;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        background: var(--bg-card);
        margin-bottom: 0.75rem;
    }}
    .themed-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        color: var(--text-primary);
    }}
    .themed-table th {{
        position: sticky;
        top: 0;
        background: var(--table-head-bg);
        color: var(--table-head-text);
        text-align: left;
        padding: 0.55rem 0.65rem;
        border-bottom: 1px solid var(--border-color);
        font-weight: 700;
        white-space: nowrap;
        z-index: 1;
    }}
    .themed-table td {{
        padding: 0.45rem 0.65rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-secondary);
        white-space: nowrap;
    }}
    .themed-table tr:nth-child(even) td {{
        background: var(--table-stripe);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_plotly(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent Light/Dark styling and transparent surface to Plotly figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
    )
    return fig


def render_themed_table(df_tbl: pd.DataFrame, formatters: dict | None = None, height: int = 335) -> None:
    """Use native st.dataframe when mode aligns with Streamlit canvas; render themed HTML table on override."""
    matches_native = (theme_mode == "System") or (effective_dark == (native_theme_type == "dark"))
    if matches_native:
        styler = df_tbl.style.format(formatters) if formatters else df_tbl
        st.dataframe(styler, width="stretch", height=height, hide_index=True)
    else:
        fmt = formatters or {}
        headers = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in df_tbl.columns)
        rows_html = []
        for _, row in df_tbl.iterrows():
            cells = []
            for col in df_tbl.columns:
                val = row[col]
                if col in fmt and pd.notna(val):
                    txt = fmt[col].format(val)
                else:
                    txt = "" if pd.isna(val) else str(val)
                cells.append(f"<td>{html_lib.escape(txt)}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        st.markdown(
            f"""
            <div class="themed-table-wrap" style="max-height:{height}px;">
                <table class="themed-table">
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{''.join(rows_html)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def clear_analysis_state() -> None:
    """Immediately purge any filter/analysis keys stored in session state."""
    for key in list(st.session_state.keys()):
        if key.startswith("flt_"):
            del st.session_state[key]


# ==================== HERO HEADER ====================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-tag">Palo Alto Networks • Preventive People Analytics</div>
        <div class="hero-title">Employee Engagement, Satisfaction &amp; Burnout Diagnostic Platform</div>
        <div class="hero-sub">Early-warning workforce health intelligence: upload a workforce CSV file to analyze engagement, satisfaction, workload stress, and burnout risk.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== CSV FILE UPLOAD GATE ====================
uploaded_file = st.file_uploader(
    "Upload Employee Dataset (CSV)",
    type=["csv"],
    accept_multiple_files=False,
    key="csv_uploader",
    help="Upload an HR/workforce CSV file to generate diagnostics. Removing the file clears all analysis immediately.",
)

st.sidebar.markdown("## 🛡️ Diagnostic Controls")
st.sidebar.caption(f"Mode: **{theme_mode}** ({'Dark' if effective_dark else 'Light'} active)")

if uploaded_file is None:
    clear_analysis_state()
    st.sidebar.info("Please upload a CSV file to begin analysis.")
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# Parse and validate strictly from currently uploaded bytes (never cached or persisted)
file_sig = f"{uploaded_file.name}:{uploaded_file.size}:{getattr(uploaded_file, 'file_id', '')}"
if st.session_state.get("flt_active_file_sig") != file_sig:
    clear_analysis_state()
    st.session_state["flt_active_file_sig"] = file_sig

try:
    raw_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df_all, audit_info = process_uploaded_dataframe(raw_df)
except Exception as exc:
    clear_analysis_state()
    st.sidebar.error("Uploaded CSV could not be analyzed.")
    st.error(f"Unable to analyze uploaded CSV (`{uploaded_file.name}`): {exc}")
    st.stop()

# ==================== SIDEBAR FILTERS (STRICTLY FROM UPLOADED CSV) ====================
st.sidebar.caption(f"Active File: **{uploaded_file.name}** ({len(df_all):,} rows)")

all_depts = sorted(df_all["Department"].unique().tolist())
selected_depts = st.sidebar.multiselect(
    "Department",
    options=all_depts,
    default=all_depts,
    key=f"flt_depts_{file_sig}",
    help="Select one or more departments to inspect.",
)

if selected_depts:
    avail_roles = sorted(
        df_all[df_all["Department"].isin(selected_depts)]["JobRole"].unique().tolist()
    )
else:
    avail_roles = sorted(df_all["JobRole"].unique().tolist())

selected_roles = st.sidebar.multiselect(
    "Job Role",
    options=avail_roles,
    default=avail_roles,
    key=f"flt_roles_{file_sig}",
    help="Roles cascade automatically based on selected departments.",
)

overtime_mode = st.sidebar.radio(
    "OverTime Toggle",
    options=["All Employees", "OverTime = Yes", "OverTime = No"],
    index=0,
    key=f"flt_ot_{file_sig}",
    help="Isolate overtime-exposed personnel vs. standard-schedule staff.",
)

eng_threshold = st.sidebar.slider(
    "Engagement Alert Threshold (0–100)",
    min_value=10.0,
    max_value=90.0,
    value=42.0,
    step=1.0,
    key=f"flt_eng_thresh_{file_sig}",
    help="Employees below this composite Engagement Index trigger Low-Engagement Alerts.",
)

min_ten, max_ten = int(df_all["YearsAtCompany"].min()), int(df_all["YearsAtCompany"].max())
if min_ten < max_ten:
    tenure_range = st.sidebar.slider(
        "Company Tenure Range (YearsAtCompany)",
        min_value=min_ten,
        max_value=max_ten,
        value=(min_ten, max_ten),
        step=1,
        key=f"flt_tenure_{file_sig}",
    )
else:
    st.sidebar.caption(f"Company Tenure: {min_ten} yrs (uniform in dataset)")
    tenure_range = (min_ten, max_ten)

selected_burnout = st.sidebar.multiselect(
    "Burnout Risk Tiers",
    options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"],
    key=f"flt_burnout_{file_sig}",
)

# Apply filters
mask = (
    df_all["Department"].isin(selected_depts)
    & df_all["JobRole"].isin(selected_roles)
    & (df_all["YearsAtCompany"] >= tenure_range[0])
    & (df_all["YearsAtCompany"] <= tenure_range[1])
    & df_all["Burnout_Risk_Level"].isin(selected_burnout)
)
if overtime_mode == "OverTime = Yes":
    mask &= df_all["OverTime"] == "Yes"
elif overtime_mode == "OverTime = No":
    mask &= df_all["OverTime"] == "No"

df = df_all[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric(
    "Active Cohort Size",
    f"{len(df):,} / {len(df_all):,}",
    f"{len(df)/len(df_all)*100:.1f}% of org",
)
st.sidebar.caption(
    f"Validation Audit: {audit_info['total_missing_values']} missing values • "
    f"{audit_info['duplicate_rows']} duplicates • Scales [1–4] verified."
)

if df.empty:
    st.warning("No employees match the current filter combination. Please broaden your sidebar selections.")
    st.stop()

# ==================== TOP KPI STRIP ====================
org_eng = df_all["Engagement_Index"].mean()
org_burn = df_all["Burnout_Risk_Score"].mean()
org_wlb = df_all["WorkLifeBalance"].mean()
org_stab = df_all["Satisfaction_Stability_Score"].mean()

cur_eng = df["Engagement_Index"].mean()
cur_burn = df["Burnout_Risk_Score"].mean()
cur_wlb = df["WorkLifeBalance"].mean()
cur_stab = df["Satisfaction_Stability_Score"].mean()
cur_ws = df["Workload_Stress_Indicator"].mean()

low_eng_cnt = int((df["Engagement_Index"] < eng_threshold).sum())
high_burn_cnt = int((df["Burnout_Risk_Level"] == "High").sum())
alert_union_cnt = int(((df["Engagement_Index"] < eng_threshold) | (df["Burnout_Risk_Level"] == "High")).sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(
        f"""<div class="kpi-card teal">
        <div class="kpi-label">Engagement Index</div>
        <div class="kpi-value">{cur_eng:.1f} / 100</div>
        <div class="kpi-delta">Δ vs Org ({org_eng:.1f}): <b>{cur_eng - org_eng:+.1f}</b> • Raw {df['Engagement_Raw'].mean():.2f}/4</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card coral">
        <div class="kpi-label">Burnout Risk Score</div>
        <div class="kpi-value">{cur_burn:.1f} / 100</div>
        <div class="kpi-delta">High Risk: <b>{high_burn_cnt}</b> ({high_burn_cnt/len(df)*100:.1f}%) • Org {org_burn:.1f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Work-Life Balance</div>
        <div class="kpi-value">{cur_wlb:.2f} / 4.00</div>
        <div class="kpi-delta">Δ vs Org ({org_wlb:.2f}): <b>{cur_wlb - org_wlb:+.2f}</b> • {(df['WorkLifeBalance']<=2).mean()*100:.1f}% Low</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card amber">
        <div class="kpi-label">Satisfaction Stability</div>
        <div class="kpi-value">{cur_stab:.1f} / 100</div>
        <div class="kpi-delta">Δ vs Org ({org_stab:.1f}): <b>{cur_stab - org_stab:+.1f}</b> • Consistency</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""<div class="kpi-card indigo">
        <div class="kpi-label">Workload Stress</div>
        <div class="kpi-value">{cur_ws:.1f} / 100</div>
        <div class="kpi-delta">OT: <b>{df['OverTime_Flag'].mean()*100:.1f}%</b> • Freq Travel: {(df['BusinessTravel']=='Travel_Frequently').mean()*100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k6:
    st.markdown(
        f"""<div class="kpi-card red">
        <div class="kpi-label">Priority Alerts</div>
        <div class="kpi-value">{alert_union_cnt:,} ({alert_union_cnt/len(df)*100:.1f}%)</div>
        <div class="kpi-delta">&lt;{eng_threshold:.0f} Eng: <b>{low_eng_cnt}</b> • Context Attr: {df['Attrition'].mean()*100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

# ==================== 4 DASHBOARD MODULES ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Engagement Health Overview",
    "🔥 Burnout Risk Dashboard",
    "🧭 Role & Career Stage Analysis",
    "🚨 Manager Action Panel",
])

# -------------------- MODULE 1: ENGAGEMENT HEALTH OVERVIEW --------------------
with tab1:
    st.markdown(
        f"""<div class="insight-box">
        <b>Diagnostic Summary:</b> The active cohort (<b>n = {len(df):,}</b>) posts a Composite Engagement Index of <b>{cur_eng:.1f}/100</b> (Z = {df['Engagement_Z'].mean():+.2f}) and Satisfaction Stability of <b>{cur_stab:.1f}/100</b>. Currently, <b>{low_eng_cnt} employees ({low_eng_cnt/len(df)*100:.1f}%)</b> sit below your alert threshold of <b>{eng_threshold:.0f}</b>.
        </div>""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.05, 1.0])
    with c1:
        fig_hist = go.Figure()
        stayed_s = df[df["Attrition"] == 0]["Engagement_Index"]
        left_s = df[df["Attrition"] == 1]["Engagement_Index"]
        if len(stayed_s) > 0:
            fig_hist.add_trace(
                go.Histogram(
                    x=stayed_s, name=f"Stayed (n={len(stayed_s)})",
                    nbinsx=14, opacity=0.65, marker_color=TEAL, histnorm="percent",
                )
            )
        if len(left_s) > 0:
            fig_hist.add_trace(
                go.Histogram(
                    x=left_s, name=f"Left (n={len(left_s)})",
                    nbinsx=14, opacity=0.65, marker_color=CORAL, histnorm="percent",
                )
            )
        fig_hist.add_vline(
            x=cur_eng, line_dash="dash", line_color=LINE_INK, line_width=2,
            annotation_text=f"Cohort Mean: {cur_eng:.1f}", annotation_position="top right",
            annotation_font_color=TEXT_COLOR,
        )
        fig_hist.add_vline(
            x=eng_threshold, line_dash="dot", line_color=RED, line_width=2.2,
            annotation_text=f"Alert Threshold: {eng_threshold:.0f}", annotation_position="top left",
            annotation_font_color=RED,
        )
        fig_hist.update_layout(
            title="Engagement Index Distribution (Cohort vs. Alert Threshold)",
            xaxis_title="Composite Engagement Index (0–100)",
            yaxis_title="Share of Group (%)",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(style_plotly(fig_hist, 390), width="stretch")

    with c2:
        dims = [
            ("JobInvolvement", "Job Involvement"),
            ("JobSatisfaction", "Job Satisfaction"),
            ("EnvironmentSatisfaction", "Environment Satisfaction"),
            ("RelationshipSatisfaction", "Relationship Satisfaction"),
            ("WorkLifeBalance", "Work-Life Balance"),
        ]
        dist_rows = []
        for col, lbl in dims:
            vc = df[col].value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0) * 100
            for rating, pct in vc.items():
                dist_rows.append({
                    "Dimension": f"{lbl} (μ={df[col].mean():.2f})",
                    "Rating": f"{rating} - {['Low/Poor', 'Moderate', 'High', 'Very High'][rating-1]}",
                    "Percent": pct,
                })
        dist_df = pd.DataFrame(dist_rows)
        fig_stack = px.bar(
            dist_df,
            y="Dimension",
            x="Percent",
            color="Rating",
            orientation="h",
            text=dist_df["Percent"].map(lambda v: f"{v:.1f}%" if v >= 8 else ""),
            color_discrete_sequence=[RED, AMBER, BLUE, TEAL],
            title="Ordinal Rating Distribution (1–4 Scale) Across Core Dimensions",
        )
        fig_stack.update_layout(
            barmode="stack",
            xaxis_title="Share of Filtered Workforce (%)",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5, title=""),
        )
        fig_stack.update_yaxes(autorange="reversed")
        st.plotly_chart(style_plotly(fig_stack, 390), width="stretch")

    c3, c4 = st.columns([1.0, 1.0])
    with c3:
        stab_grp = (
            df.groupby(["Engagement_Index", "Satisfaction_Stability_Score", "Burnout_Risk_Level"], observed=False)
            .agg(Headcount=("Attrition", "count"), Attrition_Rate=("Attrition", lambda s: s.mean() * 100))
            .reset_index()
        )
        fig_stab = px.scatter(
            stab_grp,
            x="Engagement_Index",
            y="Satisfaction_Stability_Score",
            size="Headcount",
            color="Burnout_Risk_Level",
            color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": RED},
            hover_data={"Attrition_Rate": ":.1f", "Headcount": True},
            title="Engagement Index vs. Satisfaction Stability (Bubble = Employee Cluster)",
        )
        fig_stab.add_vline(x=eng_threshold, line_dash="dot", line_color=RED)
        fig_stab.update_layout(
            xaxis_title="Composite Engagement Index (0–100)",
            yaxis_title="Satisfaction Stability Score (0–100)",
        )
        st.plotly_chart(style_plotly(fig_stab, 370), width="stretch")

    with c4:
        comp_metrics = [
            ("Engagement_Index", "Composite Engagement"),
            ("JobInvolvement_Norm", "Job Involvement"),
            ("JobSatisfaction_Norm", "Job Satisfaction"),
            ("EnvironmentSatisfaction_Norm", "Environment Sat."),
            ("RelationshipSatisfaction_Norm", "Relationship Sat."),
            ("WorkLifeBalance_Norm", "Work-Life Balance"),
            ("Satisfaction_Stability_Score", "Sat. Stability"),
            ("Burnout_Risk_Score", "Burnout Risk Score"),
        ]
        comp_rows = []
        for col, lbl in comp_metrics:
            for status_val, status_lbl in [(0, "Stayed"), (1, "Left")]:
                sub = df[df["Attrition"] == status_val]
                if len(sub) > 0:
                    comp_rows.append({"Metric": lbl, "Status": status_lbl, "Score": sub[col].mean()})
        comp_df = pd.DataFrame(comp_rows)
        fig_comp = px.bar(
            comp_df,
            y="Metric",
            x="Score",
            color="Status",
            barmode="group",
            orientation="h",
            text=comp_df["Score"].map(lambda v: f"{v:.1f}"),
            color_discrete_map={"Stayed": TEAL, "Left": CORAL},
            title="Contextual Validation: Stayed vs. Left Mean Scores (0–100)",
        )
        fig_comp.update_layout(
            xaxis_title="Normalized Mean Score (0–100)",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_comp.update_yaxes(autorange="reversed")
        st.plotly_chart(style_plotly(fig_comp, 370), width="stretch")

# -------------------- MODULE 2: BURNOUT RISK DASHBOARD --------------------
with tab2:
    high_b = df[df["Burnout_Risk_Level"] == "High"]
    med_b = df[df["Burnout_Risk_Level"] == "Medium"]
    st.markdown(
        f"""<div class="warn-box">
        <b>Engaged Burnout Alert:</b> Within the active selection, <b>{len(high_b)} employees ({len(high_b)/len(df)*100:.1f}%)</b> are in <b>High Burnout Risk</b> (OverTime = Yes &amp; Work-Life Balance ≤ 2) and <b>{len(med_b)} ({len(med_b)/len(df)*100:.1f}%)</b> are in <b>Medium Risk</b>. High-risk personnel show a mean Engagement Index of <b>{high_b['Engagement_Index'].mean() if len(high_b) else 0:.1f}</b> alongside <b>{high_b['Attrition'].mean()*100 if len(high_b) else 0:.1f}% contextual attrition</b>—demonstrating how burnout overrides high engagement.
        </div>""",
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1.0, 1.15])
    with b1:
        tier_summary = (
            df.groupby("Burnout_Risk_Level")
            .agg(Headcount=("Attrition", "count"), Attrition_Rate=("Attrition", lambda s: s.mean() * 100), Eng_Mean=("Engagement_Index", "mean"))
            .reindex(["Low", "Medium", "High"])
            .dropna()
            .reset_index()
        )
        tier_summary["Workforce_Pct"] = tier_summary["Headcount"] / len(df) * 100
        fig_tier = go.Figure()
        fig_tier.add_trace(
            go.Bar(
                x=tier_summary["Burnout_Risk_Level"],
                y=tier_summary["Workforce_Pct"],
                name="Workforce Share (%)",
                marker_color=BLUE,
                text=tier_summary.apply(lambda r: f"{r['Workforce_Pct']:.1f}% (n={int(r['Headcount'])})", axis=1),
                textposition="outside",
            )
        )
        fig_tier.add_trace(
            go.Bar(
                x=tier_summary["Burnout_Risk_Level"],
                y=tier_summary["Attrition_Rate"],
                name="Contextual Attrition (%)",
                marker_color=CORAL,
                text=tier_summary["Attrition_Rate"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
            )
        )
        max_tier_y = float(tier_summary[["Workforce_Pct", "Attrition_Rate"]].max().max()) if not tier_summary.empty else 50.0
        fig_tier.update_layout(
            title="Burnout Risk Tiers: Workforce Share vs. Attrition Rate",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, max(62, max_tier_y + 12)],
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(style_plotly(fig_tier, 390), width="stretch")

    with b2:
        piv_attr = df.pivot_table(index="OverTime", columns="WorkLifeBalance", values="Attrition", aggfunc="mean") * 100
        piv_cnt = df.pivot_table(index="OverTime", columns="WorkLifeBalance", values="Attrition", aggfunc="count")
        rows_ot = [r for r in ["No", "Yes"] if r in piv_attr.index]
        cols_wlb = [c for c in [1, 2, 3, 4] if c in piv_attr.columns]
        piv_attr = piv_attr.reindex(index=rows_ot, columns=cols_wlb)
        piv_cnt = piv_cnt.reindex(index=rows_ot, columns=cols_wlb).fillna(0)

        text_mat = [
            [
                f"{piv_attr.loc[r, c]:.1f}%<br>(n={int(piv_cnt.loc[r, c])})" if pd.notna(piv_attr.loc[r, c]) else "n=0"
                for c in cols_wlb
            ]
            for r in rows_ot
        ]
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=piv_attr.values,
                x=[f"WLB {c}" for c in cols_wlb],
                y=[f"OverTime: {r}" for r in rows_ot],
                text=text_mat,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale="YlOrRd",
                zmin=0,
                zmax=50,
                colorbar=dict(title="Attrition %"),
            )
        )
        fig_hm.update_layout(
            title="OverTime × Work-Life Balance Matrix (Attrition % & Headcount)",
            xaxis_title="Work-Life Balance (1 = Poor, 4 = Excellent)",
        )
        st.plotly_chart(style_plotly(fig_hm, 390), width="stretch")

    b3, b4 = st.columns([1.05, 1.0])
    with b3:
        paradox_df = (
            df.groupby(["Engagement_Band", "Burnout_Risk_Level"], observed=False)
            .agg(Headcount=("Attrition", "count"), Attrition_Rate=("Attrition", lambda s: s.mean() * 100))
            .reset_index()
            .dropna()
        )
        fig_paradox = px.bar(
            paradox_df,
            x="Engagement_Band",
            y="Attrition_Rate",
            color="Burnout_Risk_Level",
            barmode="group",
            text=paradox_df.apply(lambda r: f"{r['Attrition_Rate']:.1f}% (n={int(r['Headcount'])})", axis=1),
            color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": RED},
            title="Engaged Burnout Paradox: Engagement Band × Burnout Risk",
        )
        fig_paradox.update_traces(textposition="outside")
        max_par_y = float(paradox_df["Attrition_Rate"].max()) if not paradox_df.empty else 50.0
        fig_paradox.update_layout(
            xaxis_title="Engagement Band",
            yaxis_title="Contextual Attrition Rate (%)",
            yaxis_range=[0, max(64, max_par_y + 12)],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Burnout Tier"),
        )
        st.plotly_chart(style_plotly(fig_paradox, 390), width="stretch")

    with b4:
        ws_view = st.radio(
            "Workload Stress Driver View",
            ["OverTime × Business Travel", "Commute Distance Bands"],
            horizontal=True,
            key=f"flt_ws_view_{file_sig}",
        )
        if ws_view == "OverTime × Business Travel":
            tr_df = (
                df.groupby(["OverTime", "BusinessTravel"])
                .agg(Headcount=("Attrition", "count"), Attrition_Rate=("Attrition", lambda s: s.mean() * 100), Eng=("Engagement_Index", "mean"))
                .reset_index()
            )
            tr_df["Segment"] = "OT=" + tr_df["OverTime"] + " | " + tr_df["BusinessTravel"].str.replace("_", " ")
            tr_df = tr_df.sort_values("Attrition_Rate")
            fig_ws = px.bar(
                tr_df,
                y="Segment",
                x="Attrition_Rate",
                color="Eng",
                color_continuous_scale="Tealrose",
                orientation="h",
                text=tr_df.apply(lambda r: f"{r['Attrition_Rate']:.1f}% (n={int(r['Headcount'])}, Eng={r['Eng']:.1f})", axis=1),
                title="Workload Stress Indicator: OverTime × Travel Intensity",
            )
            fig_ws.update_traces(textposition="outside")
            max_tr_x = float(tr_df["Attrition_Rate"].max()) if not tr_df.empty else 40.0
            fig_ws.update_layout(
                xaxis_title="Contextual Attrition Rate (%)",
                xaxis_range=[0, max(58, max_tr_x + 18)],
                yaxis_title="",
            )
        else:
            cm_df = (
                df.groupby("Commute_Band", observed=False)
                .agg(
                    Headcount=("Attrition", "count"),
                    Attrition_Rate=("Attrition", lambda s: s.mean() * 100),
                    High_Burnout_Pct=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
                )
                .reset_index()
                .dropna()
            )
            cm_melt = cm_df.melt(
                id_vars=["Commute_Band", "Headcount"],
                value_vars=["Attrition_Rate", "High_Burnout_Pct"],
                var_name="Metric",
                value_name="Percentage",
            )
            fig_ws = px.bar(
                cm_melt,
                x="Commute_Band",
                y="Percentage",
                color="Metric",
                barmode="group",
                text=cm_melt["Percentage"].map(lambda v: f"{v:.1f}%"),
                color_discrete_map={"Attrition_Rate": CORAL, "High_Burnout_Pct": BLUE},
                title="Commute Distance Impact on Attrition & High Burnout",
            )
            fig_ws.update_traces(textposition="outside")
            max_cm_y = float(cm_melt["Percentage"].max()) if not cm_melt.empty else 20.0
            fig_ws.update_layout(
                xaxis_title="Commute Distance Band",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, max(28, max_cm_y + 8)],
            )
        st.plotly_chart(style_plotly(fig_ws, 340), width="stretch")

# -------------------- MODULE 3: ROLE & CAREER STAGE ANALYSIS --------------------
with tab3:
    r1, r2 = st.columns([1.1, 1.0])
    with r1:
        role_df = (
            df.groupby("JobRole")
            .agg(
                Headcount=("Attrition", "count"),
                Engagement_Index=("Engagement_Index", "mean"),
                Burnout_Risk_Score=("Burnout_Risk_Score", "mean"),
                High_Burnout_Pct=("Burnout_Risk_Level", lambda s: (s == "High").mean() * 100),
                Low_Eng_Pct=("Engagement_Index", lambda s: (s < eng_threshold).mean() * 100),
                Attrition_Rate=("Attrition", lambda s: s.mean() * 100),
                Avg_Income=("MonthlyIncome", "mean"),
            )
            .reset_index()
        )
        fig_role = px.scatter(
            role_df,
            x="Engagement_Index",
            y="Burnout_Risk_Score",
            size="Headcount",
            color="Attrition_Rate",
            color_continuous_scale="YlOrRd",
            text="JobRole",
            hover_data={"Avg_Income": ":$,.0f", "High_Burnout_Pct": ":.1f", "Low_Eng_Pct": ":.1f"},
            title="Job Role Diagnostic Landscape (Engagement vs. Burnout Risk)",
        )
        fig_role.update_traces(textposition="top center", textfont_size=10, textfont_color=TEXT_COLOR)
        fig_role.add_vline(x=cur_eng, line_dash="dash", line_color=SLATE)
        fig_role.add_hline(y=cur_burn, line_dash="dash", line_color=SLATE)
        fig_role.update_layout(
            xaxis_title="Mean Engagement Index (0–100)",
            yaxis_title="Mean Burnout Risk Score (0–100)",
        )
        st.plotly_chart(style_plotly(fig_role, 410), width="stretch")

    with r2:
        jl_df = (
            df.groupby("JobLevel")
            .agg(
                Headcount=("Attrition", "count"),
                Engagement_Index=("Engagement_Index", "mean"),
                Burnout_Risk_Score=("Burnout_Risk_Score", "mean"),
                Attrition_Rate=("Attrition", lambda s: s.mean() * 100),
                WLB=("WorkLifeBalance", "mean"),
            )
            .reset_index()
        )
        fig_jl = go.Figure()
        fig_jl.add_trace(
            go.Bar(
                x=[f"Level {l}<br>(n={n})" for l, n in zip(jl_df["JobLevel"], jl_df["Headcount"])],
                y=jl_df["Attrition_Rate"],
                name="Attrition Rate (%)",
                marker_color=CORAL,
                text=jl_df["Attrition_Rate"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                yaxis="y",
            )
        )
        fig_jl.add_trace(
            go.Scatter(
                x=[f"Level {l}<br>(n={n})" for l, n in zip(jl_df["JobLevel"], jl_df["Headcount"])],
                y=jl_df["Engagement_Index"],
                name="Engagement Index",
                mode="lines+markers+text",
                line=dict(color=LINE_INK, width=3),
                marker=dict(size=9, symbol="diamond"),
                text=jl_df["Engagement_Index"].map(lambda v: f"{v:.1f}"),
                textposition="top center",
                textfont=dict(color=TEXT_COLOR),
                yaxis="y2",
            )
        )
        max_jl_y = float(jl_df["Attrition_Rate"].max()) if not jl_df.empty else 30.0
        fig_jl.update_layout(
            title="Career Stage: Job Level Attrition & Engagement Curve",
            yaxis=dict(title="Attrition Rate (%)", range=[0, max(40, max_jl_y + 10)]),
            yaxis2=dict(title="Engagement Index (0–100)", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(style_plotly(fig_jl, 410), width="stretch")

    r3, r4 = st.columns([1.0, 1.0])
    with r3:
        tenure_axis = st.selectbox(
            "Career & Stagnation Dimension",
            [
                "YearsAtCompany (Tenure_Band)",
                "YearsInCurrentRole (Role_Tenure_Band)",
                "YearsSinceLastPromotion (Promo_Gap_Band)",
                "YearsWithCurrManager (Manager_Tenure_Band)",
            ],
            key=f"flt_ten_axis_{file_sig}",
        )
        col_map = {
            "YearsAtCompany (Tenure_Band)": "Tenure_Band",
            "YearsInCurrentRole (Role_Tenure_Band)": "Role_Tenure_Band",
            "YearsSinceLastPromotion (Promo_Gap_Band)": "Promo_Gap_Band",
            "YearsWithCurrManager (Manager_Tenure_Band)": "Manager_Tenure_Band",
        }
        band_col = col_map[tenure_axis]
        ten_df = (
            df.groupby(band_col, observed=False)
            .agg(
                Headcount=("Attrition", "count"),
                Engagement_Index=("Engagement_Index", "mean"),
                Job_Sat_Norm=("JobSatisfaction_Norm", "mean"),
                Attrition_Rate=("Attrition", lambda s: s.mean() * 100),
            )
            .reset_index()
            .dropna()
        )
        fig_ten = go.Figure()
        fig_ten.add_trace(
            go.Bar(
                x=[f"{b}<br>(n={n})" for b, n in zip(ten_df[band_col], ten_df["Headcount"])],
                y=ten_df["Attrition_Rate"],
                name="Attrition Rate (%)",
                marker_color=BLUE,
                opacity=0.85,
                text=ten_df["Attrition_Rate"].map(lambda v: f"{v:.1f}%"),
                textposition="inside",
                yaxis="y",
            )
        )
        fig_ten.add_trace(
            go.Scatter(
                x=[f"{b}<br>(n={n})" for b, n in zip(ten_df[band_col], ten_df["Headcount"])],
                y=ten_df["Engagement_Index"],
                name="Engagement Index",
                mode="lines+markers+text",
                line=dict(color=LINE_INK, width=2.8),
                text=ten_df["Engagement_Index"].map(lambda v: f"{v:.1f}"),
                textposition="top center",
                textfont=dict(color=TEXT_COLOR),
                yaxis="y2",
            )
        )
        fig_ten.add_trace(
            go.Scatter(
                x=[f"{b}<br>(n={n})" for b, n in zip(ten_df[band_col], ten_df["Headcount"])],
                y=ten_df["Job_Sat_Norm"],
                name="Job Satisfaction (0–100)",
                mode="lines+markers",
                line=dict(color=TEAL, width=2.2, dash="dash"),
                yaxis="y2",
            )
        )
        max_ten_y = float(ten_df["Attrition_Rate"].max()) if not ten_df.empty else 30.0
        fig_ten.update_layout(
            title=f"Tenure & Stagnation Trend: {tenure_axis.split(' ')[0]}",
            yaxis=dict(title="Attrition Rate (%)", range=[0, max(45, max_ten_y + 18)]),
            yaxis2=dict(title="Normalized Score (0–100)", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(style_plotly(fig_ten, 380), width="stretch")

    with r4:
        st.markdown("#### Role Diagnostic Scorecard")
        role_display = role_df.sort_values("Attrition_Rate", ascending=False).rename(
            columns={
                "JobRole": "Job Role",
                "Headcount": "n",
                "Engagement_Index": "Eng. Index",
                "Burnout_Risk_Score": "Burnout Score",
                "High_Burnout_Pct": "High Burn %",
                "Low_Eng_Pct": "Low Eng %",
                "Attrition_Rate": "Attrition %",
                "Avg_Income": "Avg Income",
            }
        )
        render_themed_table(
            role_display,
            formatters={
                "Eng. Index": "{:.1f}",
                "Burnout Score": "{:.1f}",
                "High Burn %": "{:.1f}%",
                "Low Eng %": "{:.1f}%",
                "Attrition %": "{:.1f}%",
                "Avg Income": "${:,.0f}",
            },
            height=340,
        )

# -------------------- MODULE 4: MANAGER ACTION PANEL --------------------
with tab4:
    def dynamic_tier(row: pd.Series) -> str:
        low_e = row["Engagement_Index"] < eng_threshold
        high_b = row["Burnout_Risk_Level"] == "High"
        med_ot = (row["Burnout_Risk_Level"] == "Medium") and (row["OverTime_Flag"] == 1)
        if low_e and high_b:
            return "1. Dual Crisis (Low Eng + High Burnout)"
        if high_b:
            return "2. Silent Burnout Risk (High Burnout)"
        if low_e:
            return "3. Disengaged Alert (Low Engagement)"
        if med_ot:
            return "4. Overload Watch (OverTime)"
        return "5. Stable / Monitor"

    df["Dynamic_Alert_Tier"] = df.apply(dynamic_tier, axis=1)

    priority_df = df[df["Dynamic_Alert_Tier"] != "5. Stable / Monitor"].copy()
    st.markdown(
        f"""<div class="warn-box">
        <b>Live Early-Warning Trigger Summary (Threshold &lt; {eng_threshold:.0f}):</b> <b>{len(priority_df):,} employees ({len(priority_df)/len(df)*100:.1f}% of filtered cohort)</b> require proactive manager review—capturing <b>{int(priority_df['Attrition'].sum())} of {int(df['Attrition'].sum())} attrition events ({priority_df['Attrition'].sum()/max(df['Attrition'].sum(),1)*100:.1f}%)</b> in this cohort.
        </div>""",
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns([1.05, 1.0])
    with m1:
        tier_tbl = (
            df.groupby("Dynamic_Alert_Tier")
            .agg(
                Headcount=("Attrition", "count"),
                Mean_Eng=("Engagement_Index", "mean"),
                Mean_Burnout=("Burnout_Risk_Score", "mean"),
                Attrition_Rate=("Attrition", lambda s: s.mean() * 100),
                Exits=("Attrition", "sum"),
            )
            .sort_index()
            .reset_index()
        )
        tier_tbl["Workforce_Pct"] = tier_tbl["Headcount"] / len(df) * 100
        tier_tbl["Exit_Capture_Pct"] = tier_tbl["Exits"] / max(df["Attrition"].sum(), 1) * 100

        fig_act = go.Figure()
        fig_act.add_trace(
            go.Bar(
                y=tier_tbl["Dynamic_Alert_Tier"],
                x=tier_tbl["Workforce_Pct"],
                name="Workforce Share (%)",
                orientation="h",
                marker_color=BLUE,
                text=tier_tbl.apply(lambda r: f"{r['Workforce_Pct']:.1f}% (n={int(r['Headcount'])})", axis=1),
                textposition="outside",
            )
        )
        fig_act.add_trace(
            go.Bar(
                y=tier_tbl["Dynamic_Alert_Tier"],
                x=tier_tbl["Exit_Capture_Pct"],
                name="Attrition Capture (%)",
                orientation="h",
                marker_color=CORAL,
                text=tier_tbl.apply(lambda r: f"{r['Exit_Capture_Pct']:.1f}% ({r['Attrition_Rate']:.1f}% rate)", axis=1),
                textposition="outside",
            )
        )
        max_act_x = float(tier_tbl[["Workforce_Pct", "Exit_Capture_Pct"]].max().max()) if not tier_tbl.empty else 80.0
        fig_act.update_layout(
            title="Priority Intervention Tiers: Workforce Share vs. Attrition Capture",
            xaxis_title="Percentage (%)",
            xaxis_range=[0, max(88, max_act_x + 22)],
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_act.update_yaxes(autorange="reversed")
        st.plotly_chart(style_plotly(fig_act, 380), width="stretch")

    with m2:
        st.markdown("#### Recommended HR & Manager Playbooks")
        playbooks = pd.DataFrame([
            {
                "Action Tier": "1. Dual Crisis",
                "Trigger": f"Eng < {eng_threshold:.0f} & High Burnout",
                "Immediate Playbook": "48-hr skip-level stay check-in; mandatory overtime relief; root-cause environment/role audit.",
            },
            {
                "Action Tier": "2. Silent Burnout",
                "Trigger": "High Burnout (OT=Yes & WLB≤2)",
                "Immediate Playbook": "Rebalance sprint/travel load; award recovery PTO; protect high performers before sudden exit.",
            },
            {
                "Action Tier": "3. Disengaged Alert",
                "Trigger": f"Eng < {eng_threshold:.0f} (Low/Mod Burnout)",
                "Immediate Playbook": "Career growth/promotion review (6+ yr gap check); manager coaching; training enrollment.",
            },
            {
                "Action Tier": "4. Overload Watch",
                "Trigger": "OT=Yes & WLB ≥ 3",
                "Immediate Playbook": "Monitor travel intensity & Level 1 overtime exposure; cap consecutive OT months.",
            },
        ])
        render_themed_table(playbooks, height=300)

    st.markdown("#### Priority Employee Watchlist (Filtered Alerts)")
    alert_opts = sorted(df["Dynamic_Alert_Tier"].unique().tolist())
    alert_default = [t for t in alert_opts if not t.startswith("5.")]
    alert_filter = st.multiselect(
        "Show Alert Tiers in Watchlist",
        options=alert_opts,
        default=alert_default,
        key=f"flt_watchlist_{file_sig}",
    )
    watch_cols = [
        "Dynamic_Alert_Tier",
        "Department",
        "JobRole",
        "JobLevel",
        "Age",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "OverTime",
        "WorkLifeBalance",
        "BusinessTravel",
        "Engagement_Index",
        "Satisfaction_Stability_Score",
        "Burnout_Risk_Score",
        "Workload_Stress_Indicator",
        "Attrition_Label",
    ]
    watchlist = (
        df[df["Dynamic_Alert_Tier"].isin(alert_filter)][watch_cols]
        .sort_values(["Burnout_Risk_Score", "Engagement_Index"], ascending=[False, True])
        .reset_index(drop=True)
    )
    render_themed_table(
        watchlist,
        formatters={
            "Engagement_Index": "{:.1f}",
            "Satisfaction_Stability_Score": "{:.1f}",
            "Burnout_Risk_Score": "{:.1f}",
            "Workload_Stress_Indicator": "{:.1f}",
        },
        height=310,
    )
    csv_bytes = watchlist.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Priority Action Watchlist (CSV)",
        data=csv_bytes,
        file_name="priority_action_watchlist.csv",
        mime="text/csv",
    )
