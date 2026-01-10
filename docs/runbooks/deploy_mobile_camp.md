# Runbook: Deploy Mobile Camp

## Overview
Deploy a mobile enrollment camp to accelerate child biometric updates in an underserved district.

---

## Pre-Deployment Checklist

- [ ] District identified via priority_scores.csv (rank ≤ 20)
- [ ] Camp location confirmed (school/community center)
- [ ] Equipment requisitioned (3 biometric devices)
- [ ] Staff assigned (2 operators + 1 supervisor)
- [ ] Community notification sent (7 days prior)

---

## Day-of Deployment

### Setup (0800-0900)
1. Arrive at location with equipment
2. Test all biometric devices connectivity
3. Set up registration desk with forms
4. Verify network/satellite connectivity
5. Brief staff on daily target (200 updates)

### Operations (0900-1700)
1. Check child's existing Aadhaar status
2. Capture biometric (fingerprint + iris)
3. Log update in tracking system
4. Issue confirmation receipt
5. Update daily tally sheet

### Closure (1700-1800)
1. Sync all devices to central server
2. Count and reconcile daily updates
3. Secure equipment for transport
4. Report daily metrics to dashboard

---

## Metrics to Track

| Metric | Target | Method |
|--------|--------|--------|
| Updates/Day | 200 | Device logs |
| Queue Time | <30 min | Sampling |
| Failure Rate | <5% | Error logs |

---

## Escalation Contacts

| Issue | Contact |
|-------|---------|
| Device Failure | Tech Support: xxx-xxx |
| Staff Shortage | District Coordinator |
| Community Issues | Local Administration |

---

## Post-Event

1. Upload all data within 24 hours
2. Submit camp completion report
3. Update pilot_monitor dashboard
4. Schedule follow-up if needed
