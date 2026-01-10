# Runbook: Manual Model Retraining

**Use Case:** When automated retraining fails or manual intervention is needed.

---

## Prerequisites

- Python 3.11+ with project venv activated
- Access to `data/raw/` directory with latest CSVs
- Write permissions to `outputs/` and `data/processed/`

---

## Steps

### 1. Activate Environment
```powershell
cd d:\UIDAI\code
.\.venv\Scripts\Activate.ps1
```

### 2. Run ETL Pipeline
```powershell
python src/01_agg_etl.py
python src/01_privacy_guard.py
```

Expected output:
- `data/processed/master.parquet`
- `outputs/suppression_log.csv`

### 3. Generate Features
```powershell
python src/02_features.py
```

Expected output:
- `data/processed/model_features.parquet`

### 4. Train Model & Generate Forecasts
```powershell
python src/03_model.py
```

Expected output:
- `models/lgb_model.pkl`
- `outputs/forecasts.csv`

### 5. Update Priority Scores
```powershell
python src/04_priority.py
```

Expected output:
- `outputs/priority_scores.csv`
- `outputs/bottleneck_labels.csv`

### 6. Validate Model
```powershell
python src/validation_forecast.py
```

Check:
- MAPE ≤ 70% (for cross-sectional data)
- No errors in output

---

## Verification Checklist

- [ ] `master.parquet` updated with latest week
- [ ] `forecasts.csv` contains predictions
- [ ] `priority_scores.csv` has all districts
- [ ] Validation MAPE within threshold
- [ ] Dashboard shows updated data

---

## Rollback

If retraining fails:
```powershell
# Restore previous outputs
git checkout HEAD~1 -- outputs/
```

---

## Post-Retraining

1. Clear Streamlit cache: Restart dashboard
2. Update audit log entry
3. Notify team of successful retraining
