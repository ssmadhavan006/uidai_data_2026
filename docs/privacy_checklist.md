# Privacy Checklist
## Aadhaar Pulse - Child Update Intelligence Platform

**Version:** 2.1  
**Last Updated:** 2026-01-18

---

## Data Classification

| Attribute | Value |
|-----------|-------|
| Data Type | Pre-aggregated counts (enrolment, demographic, biometric) |
| PII Level | None - district-level aggregates only |
| Sensitivity | Low - no individual information |

---

## Technical Safeguards

### 1. K-Anonymity (k=10)

- **Threshold:** k = 10 minimum per cell
- **Method:** Values < k replaced with -1 (suppression marker)
- **Code:** `src/privacy_guard.py::apply_k_anonymity()`
- **Logging:** All suppressions logged to `suppression_log.csv`

### 2. Secure Hashing

- **Algorithm:** SHA-256
- **Salt Management:** 
  - Production: Streamlit `secrets.toml` → `privacy.hash_salt`
  - Dev/CI: `AADHAAR_SALT` environment variable
  - Rotation: Annual recommended
- **Code:** `src/privacy_guard.py::hash_identifier()`

### 3. Differential Privacy (Exports)

- **Counts:** Laplace mechanism (sensitivity = 1)
- **Rates:** Gaussian mechanism (δ = 10⁻⁵)
- **Budget:** ε = 1.0 (configurable)
- **Code:** `src/dp_export.py::dp_export_dataframe()`

### 4. AI Chatbot Safety (New)

- **Provider:** Google Gemini 2.5 Flash
- **Safety Settings:** BLOCK_MEDIUM_AND_ABOVE for all harm categories
- **Context Control:** Only aggregated district-level data passed to model
- **No PII Exposure:** Chatbot only receives priority scores, bottleneck labels, interventions
- **Code:** `src/chatbot.py::_initialize_model()`

---

## Access Controls

### Role-Based Access

| Role | Dashboard | Export | Simulation | AI Chat | Admin |
|------|-----------|--------|------------|---------|-------|
| Analyst | Full | With DP | Yes | Yes | No |
| Viewer | Masked | No | No | Yes | No |
| Admin | Full | Full | Yes | Yes | Yes |

### Audit Logging

- **Format:** JSON
- **Location:** `outputs/audit_logs/audit_YYYY-MM-DD.json`
- **Events:** login, logout, export, simulation, view, ai_assistant
- **Code:** `app/utils/audit_logger.py`

---

## Analytical Minimization

- ✅ Only aggregated metrics (counts, ratios, rates)
- ✅ District/state-level patterns only
- ✅ No disaggregation attempts
- ❌ No individual-level inference
- ❌ No external data linkage

---

## Output Review Protocol

1. Review all figures for small-count cells
2. Verify no district-week has < k transactions
3. Apply DP noise before public export
4. Log all exports in audit trail

---

## Compliance References

- [Data Governance Policy](data_governance.md)
- [Privacy Impact Assessment](privacy_impact_assessment.md)
- UIDAI Data Handling Guidelines

---

**Reviewed by:** _________________ | **Date:** __________ | **Status:** ☐ Pending ☑ Approved
