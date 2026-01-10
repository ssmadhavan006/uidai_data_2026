# Runbook: Dashboard Maintenance

**Purpose:** Procedures for updating and maintaining the Streamlit dashboards.

---

## 1. Adding New Districts

When new districts are added to the source data:

```python
# 1. Update priority_scores.csv
python src/04_priority.py

# 2. Verify new districts appear
import pandas as pd
df = pd.read_csv('outputs/priority_scores.csv')
print(df['district'].nunique())

# 3. Clear Streamlit cache
# Restart the dashboard or press 'R' in browser
```

---

## 2. Modifying Dashboard Layout

### Adding a New Tab
1. Open `app/dashboard.py`
2. Add tab to the `st.tabs()` call
3. Add content block with `with tab_name:`

### Adding New Metrics
1. Ensure metric exists in `priority_scores.csv` or `model_features.parquet`
2. Add `st.metric()` or chart in appropriate section
3. Test with `streamlit run app/dashboard.py`

---

## 3. Updating Intervention Options

Edit `config/interventions.json`:
```json
{
  "new_intervention": {
    "description": "Description here",
    "cost": 100000,
    "capacity_per_week": 500,
    "duration_weeks": 8,
    "effectiveness": {
      "conservative": 0.15,
      "median": 0.25,
      "optimistic": 0.35
    }
  }
}
```

---

## 4. Troubleshooting

### Dashboard Won't Start
```powershell
# Check if port is in use
netstat -ano | findstr :8501

# Kill process if needed
taskkill /PID <pid> /F

# Restart
streamlit run app/dashboard.py
```

### Data Not Refreshing
1. Check data file timestamps
2. Clear Streamlit cache: Delete `.streamlit/cache`
3. Restart dashboard

### Charts Not Rendering
- Verify Plotly is installed: `pip install plotly`
- Check for NaN values in data
- Review browser console for JS errors

---

## 5. Backup Procedures

Before major changes:
```powershell
# Backup current dashboard
cp app/dashboard.py app/dashboard.py.bak

# Backup config
cp config/interventions.json config/interventions.json.bak
```

---

## 6. Performance Optimization

If dashboard is slow:
1. Add `@st.cache_data` to data loading functions
2. Limit data to recent weeks only
3. Use sampling for large visualizations
