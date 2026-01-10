# Runbook: Incident Response
## High Forecast Error Alert

**Severity:** High  
**Triggered by:** MAPE >80% for 2 consecutive weeks

---

## 1. Initial Assessment (15 min)

### Verify the Alert
1. Check monitoring dashboard: `streamlit run app/monitor.py`
2. Confirm MAPE values in `outputs/forecast_validation.csv`
3. Rule out false positive from data delay

### Quick Checks
```bash
# Check latest data timestamp
python -c "import pandas as pd; f = pd.read_parquet('data/processed/model_features.parquet'); print(f['week_number'].max())"

# Check pipeline status
cat outputs/audit_logs/audit_$(date +%Y-%m-%d).json | tail -20
```

---

## 2. Diagnosis (30 min)

### Check for Data Drift
```python
import pandas as pd
from scipy import stats

current = pd.read_parquet('data/processed/model_features.parquet')
historical = current[current['week_number'] < current['week_number'].max() - 4]
recent = current[current['week_number'] >= current['week_number'].max() - 4]

# Compare distributions
for col in ['bio_update_child', 'demo_update_child']:
    stat, p = stats.ks_2samp(historical[col], recent[col])
    print(f"{col}: KS stat={stat:.3f}, p={p:.4f}")
```

### Check District-Level Errors
- Identify which districts have highest errors
- Check if specific states/regions are affected
- Look for external events (holidays, campaigns)

---

## 3. Remediation Options

### Option A: Data Quality Issue
1. Contact data source team
2. Re-run ETL with corrected data
3. Validate with spot checks

### Option B: Model Needs Retraining
```bash
# Trigger manual retraining
python src/02_features.py
python src/03_model.py
python src/validation_forecast.py
```

### Option C: Feature/Concept Drift
1. Review feature importance
2. Consider adding new features
3. Retrain with expanded window

---

## 4. Resolution

- [ ] Root cause identified and documented
- [ ] Fix implemented and tested
- [ ] MAPE returned to acceptable levels
- [ ] Incident report filed
- [ ] Alert resolved in dashboard

---

## 5. Post-Incident Review

- Schedule retrospective within 1 week
- Update runbook if needed
- Consider preventive measures
