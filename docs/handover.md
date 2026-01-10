# Handover Document
## Aadhaar Pulse - Child Update Intelligence Platform

**Date:** 2026-01-10  
**Prepared by:** Analytics Team

---

## 1. System Overview

**Purpose:** Predictive analytics platform for optimizing child biometric update operations across 742 districts.

**Key Capabilities:**
- Demand forecasting (LightGBM, SMAPE <70%)
- Bottleneck detection (rule-based classifier)
- Priority scoring (composite index)
- Policy simulation (Monte Carlo)
- Differential privacy for exports

---

## 2. Repository Location

```
GitHub: github.com/uidai/aadhaar-pulse
Local: D:\UIDAI\code
```

---

## 3. Quick Start

```powershell
# 1. Clone and setup
git clone <repo-url>
cd aadhaar-pulse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run dashboard
streamlit run app/dashboard.py

# 3. Or use Docker
docker-compose up -d
```

---

## 4. Key Contacts

| Role | Contact | Responsibility |
|------|---------|----------------|
| Technical Lead | lead@uidai.gov.in | Architecture decisions |
| Data Engineer | data@uidai.gov.in | ETL pipeline |
| On-Call | oncall@uidai.gov.in | Production issues |

---

## 5. Service Endpoints

| Service | URL | Port |
|---------|-----|------|
| Dashboard | localhost:8501 | 8501 |
| API | localhost:8000 | 8000 |
| Pilot Monitor | localhost:8502 | 8502 |
| MLflow | localhost:5000 | 5000 |

---

## 6. Documentation

| Document | Purpose |
|----------|---------|
| `docs/SLA.md` | Service level objectives |
| `docs/pilot_charter.md` | Pilot design |
| `docs/runbooks/*.md` | Operational procedures |
| `monitoring/config.yaml` | Alert configuration |

---

## 7. Scheduled Tasks

| Task | Schedule | Command |
|------|----------|---------|
| Weekly Retraining | Sun 00:00 UTC | CI/CD auto-triggered |
| Data Refresh | Daily 06:00 IST | Manual upload |

---

## 8. Stopping/Starting Services

```powershell
# Docker
docker-compose down
docker-compose up -d

# Local
# Dashboard: Ctrl+C in terminal
# Restart: streamlit run app/dashboard.py
```

---

**System handover acknowledged by:** _________________

**Date:** _________________
