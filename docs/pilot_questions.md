# Pilot Questions

These research questions will guide the analysis and dashboard development.

## 1. Bottleneck Detection
**Question**: Which districts show the largest growing gap between demographic updates and biometric updates for children aged 5-15, indicating a potential systemic or resource bottleneck?

**Metric**: `bio_demo_gap_child = demo_update_child - bio_update_child`

**Threshold**: Districts where `bio_demo_ratio_child < 0.5` are flagged as potential bottlenecks.

**Action**: Prioritize resource allocation (devices, staff) to these districts.

---

## 2. Surge Forecasting
**Question**: Can we model and forecast seasonal surges in child biometric updates (e.g., post-summer or pre-academic year) at the district level to pre-allocate resources?

**Approach**:
- Time-series analysis using Prophet or similar
- Weekly aggregation to capture seasonality
- Identify school calendar correlations

**Metric**: `bio_update_child` weekly trend and 4-week forecast

---

## 3. Priority Scoring
**Question**: How can we develop a composite 'priority score' for districts that combines update backlog, success rate, and total child population to guide intervention campaigns?

**Proposed Formula**:
```
priority_score = (backlog_weight * bio_demo_gap_child) + 
                 (ratio_weight * (1 - bio_demo_ratio_child)) + 
                 (population_weight * enroll_child)
```

**Weights**: To be calibrated based on domain expertise (default: 0.4, 0.3, 0.3)

---

## Success Criteria
- [ ] Identify top 20 bottleneck districts with actionable insights
- [ ] Generate 4-week forecasts for high-volume states
- [ ] Produce priority-ranked district list for campaign planning
