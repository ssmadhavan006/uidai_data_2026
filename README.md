# Aadhaar Pulse
## Child Update Intelligence Platform

*Fusion-driven, AI-first decision support for proactive, equitable Aadhaar service delivery*

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd aadhaar-pulse

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/dashboard.py
```

**Demo Credentials:**
- Analyst: `analyst` / `analyst123` (full access)
- Viewer: `viewer` / `viewer123` (read-only)

---

## 📁 Project Structure

```
aadhaar-pulse/
├── app/                    # Streamlit dashboards
│   ├── dashboard.py       # Main unified dashboard (6 tabs)
│   ├── components/        # Reusable view components
│   └── utils/             # Auth, data loading, export
├── api/                    # FastAPI endpoints
│   └── main.py            # /forecast, /bottleneck/analyze, /recommend
├── src/                    # Core analytics modules
│   ├── agg_etl.py         # ETL pipeline
│   ├── privacy_guard.py   # K-anonymity enforcement
│   ├── features.py        # Feature engineering
│   ├── forecast_lightgbm.py # LightGBM forecasting
│   ├── bottleneck_fusion.py # 5-type bottleneck detection
│   ├── simulator.py       # Monte Carlo policy simulator
│   └── fairness_audit.py  # Equity analysis
├── notebooks/              # Analysis & validation
├── config/                 # Intervention definitions
├── docs/                   # Documentation, runbooks, SLAs
├── outputs/                # Generated files (priority_scores, etc.)
└── data/                   # Raw and processed data
```

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Bottleneck Fusion** | 5 diagnostic types: Operational, Demographic Surge, Capacity Strain, Inclusion Gap, Anomaly |
| **District Comparison** | **(New)** Multi-district side-by-side analysis, trend classification, and peer benchmarking |
| **Demand Forecasting** | LightGBM with hierarchical reconciliation, SMAPE <70% |
| **Policy Simulator** | Monte Carlo simulation with 90% confidence intervals |
| **Explainability** | SHAP feature importance, per-district rationale |
| **Privacy-First** | k=10 anonymity, SHA-256 hashing, differential privacy exports |
| **RBAC** | Analyst (full) / Viewer (masked) role-based access |
| **Pilot Framework** | Treatment/control selection, DiD causal analysis |

---

## 🛠️ Run Commands

```bash
# Main dashboard
streamlit run app/dashboard.py

# FastAPI
uvicorn api.main:app --reload

# Run full pipeline
python src/agg_etl.py
python src/features.py  
python src/forecast_lightgbm.py
python src/bottleneck_fusion.py

# Docker (if installed)
docker-compose up -d
```

---

## 📊 Dashboards

| Tab | Features |
|-----|----------|
| 📍 Hotspot Map | Priority visualization (Heatmap/Scatter), state summaries |
| 🔍 District Analysis | SHAP explanations, action recommendations |
| 📊 Compare Districts | **(New)** Side-by-side comparison, radar charts, trend analysis |
| 🎮 Policy Simulator | Intervention testing with Monte Carlo & 90% CIs |
| 📊 Overview | Child metrics, bottleneck distribution, top 10 list |
| 📈 Pilot Monitor | Treatment vs Control trends, action tracker |
| 🔧 System Health | Data drift (PSI), MAPE trends, alerts |

---

## 🔒 Privacy & Security

- **K-anonymity:** k=10 threshold, all values <10 suppressed
- **Hashing:** SHA-256 with cryptographic salt
- **Differential Privacy:** Laplace/Gaussian noise for exports (ε=1.0)
- **RBAC:** Role-based dashboard access
- **Audit Logging:** All actions logged to `outputs/audit_logs/`

---

## 📖 Documentation

- [Pilot Charter](docs/pilot_charter.md)
- [SLA & Alerts](docs/SLA.md)
- [Privacy Checklist](docs/privacy_checklist.md)
- [Architecture](docs/architecture.md)
- [Runbooks](docs/runbooks/)

---

## 🏆 Built for UIDAI Hackathon 2025

**Judge Pitch:**
> *"Fusion-driven bottleneck diagnosis, demand forecasting, and policy simulation—with DiD causal evaluation and production-ready Docker deployment. One command to run."*
