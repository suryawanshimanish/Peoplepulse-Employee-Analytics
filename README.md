# Palo Alto Networks — Employee Engagement, Satisfaction & Burnout Diagnostic Analysis

Preventive HR analytics platform shifting workforce monitoring from reactive attrition counting to early-warning engagement, well-being, and burnout risk intelligence (`N = 1,470` employees, 31 fields).

## Project Deliverables
1. **Executive Summary for Government Stakeholders**: `Executive_Summary_Government_Stakeholders.pdf` — Policy & workforce resilience brief covering cybersecurity talent continuity, overtime/travel governance, and phased implementation roadmap.
2. **Live Streamlit Dashboard**: `app.py` — Interactive analytics with 4 modules (*Engagement Health Overview*, *Burnout Risk Dashboard*, *Role & Career Stage Analysis*, *Manager Action Panel*) and dynamic Department, Job Role, OverTime, Engagement Threshold, Tenure Range, and Burnout Tier filters.

## Core Analytical Pipeline
- `data_pipeline.py` — Data integrity validation, [1, 4] ordinal checks, 0–100 normalization, Z-score standardization, and KPI computation (`Engagement_Index`, `Burnout_Risk_Score`, `WorkLifeBalance`, `Satisfaction_Stability_Score`, `Workload_Stress_Indicator`).
- `generate_charts.py` — Generates 8 publication-grade 300-DPI diagnostic figures in `outputs/figures/`.
- `build_reports.py` — Programmatically builds both PDF deliverables from verified analytical tables and embedded figures.

## Running Locally
```bash
pip install -r requirements.txt
python3 data_pipeline.py
streamlit run app.py
```
