# Privacy Checklist

## Data Classification
- **Data Type**: Pre-aggregated counts (enrolment, demographic updates, biometric updates)
- **PII Level**: None - All data is aggregated at pincode/district level
- **Sensitivity**: Low - No individual-level information present

## Safeguards

### 1. K-Anonymity (k=10)
All aggregated outputs (tables, charts, exports) must represent **at least 10 individuals** per cell. Any district-week combination with fewer than 10 transactions will be:
- Suppressed from public outputs
- Flagged in internal analysis
- Not used for individual-level inference

### 2. Analytical Minimization
This analysis will:
- ✅ Use only aggregated metrics (counts, ratios, rates)
- ✅ Focus on district/state-level patterns
- ✅ Avoid any disaggregation attempts
- ❌ NOT attempt to derive individual-level information
- ❌ NOT link data to external sources that could enable re-identification

### 3. Output Review Protocol
Before sharing any results:
1. Review all figures for small-count cells
2. Verify no district-week has < 10 transactions in displayed data
3. Ensure bottleneck districts are identified by aggregate patterns, not individual cases
4. Dashboard filters will enforce minimum aggregation levels

## Compliance Statement
This analysis complies with:
- UIDAI data handling guidelines for aggregated statistics
- Hackathon data usage terms
- General privacy best practices for government data

---

**Reviewed by**: [Analyst Name]  
**Date**: [Date]  
**Status**: ☐ Pending Review  ☑ Approved
