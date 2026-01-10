# Service Level Agreement (SLA)
## Aadhaar Pulse Production System

**Version:** 1.0  
**Effective Date:** 2026-01-10

---

## 1. Service Level Objectives (SLOs)

### Data Pipeline

| Metric | Target | Measurement |
|--------|--------|-------------|
| ETL Completion | Within 2 hours of data arrival | 95% of weeks |
| Data Freshness | ≤7 days old | 99% of time |
| Pipeline Success Rate | ≥95% | Weekly |

### Model Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Forecast MAPE | ≤70% | Weekly validation |
| Priority Score Stability | ±10% week-over-week | Correlation check |
| Model Retraining | Weekly | Automated |

### Dashboard Availability

| Metric | Target | Measurement |
|--------|--------|-------------|
| Uptime | 99.5% | Monthly |
| Response Time | <3 seconds for page load | 95th percentile |
| Error Rate | <1% of requests | Daily |

---

## 2. Alert Rules

### Critical (Page On-Call)
- Pipeline failure for >2 consecutive runs
- Data drift PSI >0.30 for any key feature
- Dashboard downtime >15 minutes

### High (Ticket + Email)
- MAPE >80% for 2 consecutive weeks
- Data staleness >10 days
- API error rate >5%

### Medium (Email)
- PSI between 0.20-0.30
- Pipeline duration >3 hours
- Minor data quality issues

---

## 3. Incident Response

| Severity | Response Time | Resolution Target |
|----------|---------------|-------------------|
| Critical | 15 minutes | 2 hours |
| High | 1 hour | 8 hours |
| Medium | 4 hours | 24 hours |

---

## 4. Maintenance Windows

- **Scheduled:** Sundays 02:00-06:00 IST
- **Notification:** 48 hours advance notice for planned downtime
- **Emergency:** Immediate notification via all channels

---

## 5. Contacts

| Role | Contact |
|------|---------|
| Primary On-Call | analytics-oncall@uidai.gov.in |
| Escalation | data-team-lead@uidai.gov.in |
| Vendor Support | support@vendor.com |

---

**Reviewed by:** _________________  
**Approved by:** _________________  
**Date:** _________________
