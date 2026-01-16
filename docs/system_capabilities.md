# Aadhaar Pulse - System Capabilities Document
## Child Update Intelligence Platform

**Version:** 1.0  
**Date:** 2026-01-13

---

## Executive Summary

Aadhaar Pulse is an AI-powered decision support platform that transforms raw Aadhaar operational data into **actionable intelligence** for proactive service delivery. It addresses the critical challenge of managing the mandatory biometric update wave for children aged 5-15 across 742+ districts.

**Core Value Proposition:**
> "Know where pressure will occur, why it happens, and what intervention works best—before the system is under strain."

---

## 1. Data Processing & Privacy

### 1.1 ETL Pipeline
| Capability | Description |
|------------|-------------|
| Multi-source fusion | Combines Enrollment, Demographic Update, and Biometric Update datasets |
| Temporal aggregation | District × Week granularity for trend analysis |
| Age-group focus | Special handling for 5-15 child age bucket |

### 1.2 Privacy Safeguards
| Feature | Implementation |
|---------|----------------|
| **K-Anonymity** | k=10 threshold; values <10 suppressed as -1 |
| **Secure Hashing** | SHA-256 with cryptographic salt for identifiers |
| **Differential Privacy** | Laplace/Gaussian noise for public exports (ε=1.0) |
| **Audit Logging** | JSON logs of all user actions |

**Files:** `src/01_privacy_guard.py`, `outputs/suppression_log.csv`

---

## 2. Analytics & Intelligence

### 2.1 Demand Forecasting
| Metric | Value |
|--------|-------|
| Model | LightGBM with hierarchical reconciliation |
| Accuracy | SMAPE ~62% (realistic for 742-district cross-sectional data) |
| Horizon | 4-week forecast with confidence intervals |
| Features | Lag features, rolling means, week-over-week changes |

**Use Case:** Predict which districts will face demand surge next month.

### 2.2 Bottleneck Detection (5 Types)

| Type | Detection Logic | Recommended Action |
|------|-----------------|-------------------|
| **OPERATIONAL_BOTTLENECK** | Low completion rate + high backlog | Device upgrade, staff training |
| **DEMOGRAPHIC_SURGE** | Demand 50%+ above average, normal success | Mobile camps, extended hours |
| **CAPACITY_STRAIN** | High saturation + declining throughput | Add enrollment centers |
| **INCLUSION_GAP** | Low saturation + very low activity | Awareness campaigns, outreach |
| **ANOMALY_DETECTED** | Isolation Forest flags unusual patterns | Investigation required |

**Use Case:** Diagnose *why* a district is struggling, not just *that* it is.

### 2.3 Priority Scoring
Composite index weighing:
- 30% Forecasted demand
- 30% Incompletion rate
- 25% Backlog size
- 15% Inequity indicator

**Output:** Ranked list of 742 districts for resource allocation.

### 2.4 Anomaly Detection
- Isolation Forest on multivariate features
- Z-score analysis for time-series outliers
- Spatial outlier detection across districts

### 2.5 Trend Classification & Benchmarking (New)
- **Trend Classification:** Automatic segmentation of districts into "Improving", "Declining", or "Stable" based on 4-week linear regression slopes.
- **Peer Benchmarking:** Clustering districts by bottleneck type and priority score to identify "Like-for-Like" comparisons.
- **Comparison Engine:** Side-by-side metric evaluation of 2-5 districts with normalized radar charts.

**Use Case:** "How does District A compare to similar peers, and is it improvement trend sustainable?"

---

## 3. Policy Simulation Engine

### 3.1 Intervention Catalog
| Intervention | Cost | Capacity | Duration |
|--------------|------|----------|----------|
| Mobile Camp | ₹50,000 | 500/week | 4 weeks |
| Device Upgrade | ₹250,000 | 2,000/week | 52 weeks |
| Staff Training | ₹30,000 | 200/week | 8 weeks |
| Extended Hours | ₹15,000 | 300/week | 12 weeks |
| School Partnership | ₹20,000 | 400/week | 16 weeks |

### 3.2 Monte Carlo Simulation
- 500 simulation runs per scenario
- Samples from demand uncertainty range
- Varies effectiveness (conservative/median/optimistic)
- Outputs 90% confidence intervals

### 3.3 Simulator Outputs
| Metric | Description |
|--------|-------------|
| Backlog Reduction | Projected decrease in pending updates |
| Cost per Update | ROI calculation |
| Fairness Index | Equity impact (0-1 scale) |
| Confidence Interval | 90% CI for all projections |

**Use Case:** "If we deploy 2 mobile camps for 4 weeks, what's the expected ROI?"

---

## 4. Explainability & Trust

### 4.1 SHAP Analysis
- Feature importance for each prediction
- Per-district explainability
- Top 5 drivers displayed in dashboard

### 4.2 Rationale Generation
Every flagged district includes:
- Human-readable explanation of bottleneck
- Specific metrics triggering the classification
- Recommended intervention with justification

**Use Case:** "Why is District X flagged as high priority?"

---

## 5. Fairness & Equity Auditing

### 5.1 Disparity Analysis
- State-level variance in child update rates
- Geographic equity assessment
- Identification of underserved regions

### 5.2 Inclusion Gap Detection
New bottleneck type specifically for:
- Low enrollment saturation (<30%)
- Very low update activity
- Potential access barriers

**Use Case:** Identify districts where children may be excluded due to access issues.

---

## 6. Dashboard & Visualization

### 6.1 Unified Dashboard (7 Tabs)
| Tab | Features |
|-----|----------|
| 📍 Hotspot Map | **Enhanced** heatmap/scatter modes, state summaries, priority filters |
| 🔍 District Analysis | Detailed view, SHAP, trends, rationale |
| 📊 Compare Districts | **(New)** Multi-district side-by-side, radar charts, trend classification |
| 🎮 Policy Simulator | Interactive intervention testing, 90% confidence intervals |
| 📊 Overview | Child metrics, bottleneck distribution, top 10 list |
| 📈 Pilot Monitor | Treatment vs Control trends, action tracker |
| 🔧 System Health | Data drift (PSI), MAPE trends, alerts |

### 6.2 Role-Based Access Control
| Role | Access |
|------|--------|
| Analyst | Full access to all 6 tabs, exports, simulations |
| Viewer | Hotspot Map + Overview only, masked data |

### 6.3 Export Capabilities
- PDF Action Packs with rationale
- HTML reports
- JSON data exports
- DP-protected CSV (differential privacy)

---

## 7. API Endpoints

```
GET  /districts                          - List all districts
GET  /interventions                      - Available interventions
GET  /bottleneck/analyze/{state}/{district}  - Bottleneck diagnosis
GET  /forecast/{state}/{district}        - Demand forecast
POST /recommend_action                   - Intervention recommendation
```

**Use Case:** Integration with existing UIDAI operational systems.

---

## 8. Pilot Evaluation Framework

### 8.1 Treatment/Control Design
- Select top 5 priority districts as treatment
- Match 5 control districts by state/criteria
- Define KPIs: bio_update_child, completion_rate, backlog

### 8.2 Synthetic Data Generation
- Post-intervention data with +30% treatment effect
- Control group follows baseline trend
- Enables rigorous evaluation methodology

### 8.3 Causal Analysis (Difference-in-Differences)
```
KPI = β₀ + β₁(post) + β₂(treatment) + β₃(post×treatment) + ε
```
Where β₃ = causal treatment effect

**Result:** +30% effect with p<0.001 on synthetic data

---

## 9. Production Readiness

### 9.1 Containerization
- `Dockerfile` for portable deployment
- `docker-compose.yml` with 4 services:
  - Dashboard (port 8501)
  - API (port 8000)
  - Pilot Monitor (port 8502)
  - MLflow (port 5000)

### 9.2 CI/CD Pipeline
- Weekly automated retraining (GitHub Actions)
- MAPE validation gates (fails if >70%)
- Model versioning with MLflow

### 9.3 Monitoring & Alerting
| Metric | Threshold | Alert |
|--------|-----------|-------|
| Data Drift (PSI) | >0.25 | Critical |
| Forecast MAPE | >70% | High |
| Pipeline Success | <95% | Medium |

---

## 10. Real-World Impact

### For Policy Analysts
- Evidence-based resource allocation
- Fairness metrics for equity auditing
- Scenario comparison before decisions

### For Ops Planners
- Ranked priority district list
- Clear intervention recommendations
- Cost-benefit projections

### For Field Teams
- Step-by-step runbooks
- Action packs with logistics
- Mobile camp deployment guides

### For UIDAI Leadership
- Proactive rather than reactive operations
- Reduced cost per update
- Improved child coverage equity

---

## Summary of Deliverables

| Category | Files |
|----------|-------|
| **Core ETL** | `01_agg_etl.py`, `01_privacy_guard.py`, `02_features.py` |
| **ML Models** | `forecast_lightgbm.py`, `anomaly_detection.py`, `bottleneck_fusion.py` |
| **Policy Engine** | `simulator.py`, `action_recommender.py` |
| **Dashboard** | `app/dashboard.py` (6 tabs unified) |
| **API** | `api/main.py` (5 endpoints) |
| **Pilot** | `synthetic_pilot.py`, `08_causal_evaluation.ipynb` |
| **Docs** | `SLA.md`, `pilot_charter.md`, `architecture.md`, runbooks |
| **DevOps** | `Dockerfile`, `docker-compose.yml`, `retrain.yml` |

---

**Aadhaar Pulse transforms data into governance—proactive, equitable, and accountable.**
