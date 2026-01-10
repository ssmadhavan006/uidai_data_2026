# Data Governance Policy
## Aadhaar Pulse - Child Update Intelligence Platform

**Version:** 1.0  
**Effective Date:** 2026-01-10  
**Last Review:** 2026-01-10

---

## 1. Purpose

This policy establishes the rules and responsibilities for handling Aadhaar-related data within the Aadhaar Pulse analytics platform, ensuring compliance with data protection principles and UIDAI guidelines.

---

## 2. Data Stewards

| Role | Responsibility | Contact |
|------|----------------|---------|
| **Data Owner** | UIDAI designated officer | uidai-data@gov.in |
| **Data Custodian** | Platform administrator | Manages technical access |
| **Data Analyst** | Authorized analysts | Consumes aggregated data |
| **Data Viewer** | Read-only stakeholders | Views dashboards only |

---

## 3. Data Classification

| Classification | Description | Examples |
|----------------|-------------|----------|
| **Confidential** | Individual-level data | Raw CSV with identifiers |
| **Restricted** | District-aggregated data | `master.parquet`, `model_features.parquet` |
| **Internal** | Analytical outputs | Priority scores, forecasts |
| **Public** | Suppressed/DP-protected | Exported reports with k+DP |

---

## 4. Retention Policy

| Data Type | Retention Period | Disposal Method |
|-----------|------------------|-----------------|
| Raw CSVs | 90 days from ingestion | Secure deletion |
| Processed Parquet | 1 year | Archive then delete |
| Model outputs | 2 years | Archive |
| Audit logs | 5 years | Immutable storage |

---

## 5. Access Control

### 5.1 Role-Based Access

| Role | Dashboard Access | Data Export | Simulation | Admin |
|------|------------------|-------------|------------|-------|
| Analyst | Full | With DP | Yes | No |
| Viewer | Masked only | No | No | No |
| Admin | Full | Full | Yes | Yes |

### 5.2 Authentication Requirements

- All users must authenticate via the dashboard login
- Sessions expire after 24 hours of inactivity
- Multi-factor authentication recommended for Admin role

---

## 6. Audit Logging

All the following actions are logged:

| Action | Logged Data |
|--------|-------------|
| Login/Logout | Timestamp, user_id, role, IP |
| Data Export | Timestamp, user_id, dataset, row_count |
| Simulation Run | Timestamp, user_id, parameters |
| Priority View | Timestamp, user_id, district accessed |

Logs are stored in `outputs/audit_logs/` in JSON format.

---

## 7. Data Use Approval

### New Use Cases

1. Submit request to Data Owner with purpose and scope
2. Privacy Impact Assessment required for new data combinations
3. Approval documented in `docs/approvals/` folder

### Export Requests

1. All exports must apply k-anonymity (k≥10)
2. Public releases must additionally apply differential privacy (ε≤1.0)
3. Exports logged with requestor and purpose

---

## 8. Incident Response

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Critical (breach suspected) | Immediate | Data Owner + Legal |
| High (policy violation) | 4 hours | Data Custodian |
| Medium (access anomaly) | 24 hours | Platform Admin |

---

## 9. Compliance Checklist

- [ ] Annual review of this policy
- [ ] Quarterly audit log review
- [ ] Privacy Impact Assessment updated annually
- [ ] Access roster verified monthly

---

## 10. References

- [Privacy Checklist](privacy_checklist.md)
- [Privacy Impact Assessment](privacy_impact_assessment.md)
- UIDAI Data Sharing Guidelines

---

*This document is maintained by the Aadhaar Pulse platform team.*
