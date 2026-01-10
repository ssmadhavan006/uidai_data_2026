# Runbook: Device Upgrade Protocol

## Overview
Upgrade biometric capture devices at permanent enrollment centers to improve throughput and reduce failure rates.

---

## Pre-Upgrade Checklist

- [ ] District identified via priority_scores.csv (high failure rate)
- [ ] Center list confirmed (5 centers minimum)
- [ ] New devices procured (Morpho/Mantra upgraded models)
- [ ] Technical staff scheduled for installation
- [ ] Backup plan for service continuity

---

## Upgrade Phases

### Phase 1: Assessment (Week 1)
1. Audit current device inventory at each center
2. Record device age, model, failure logs
3. Prioritize centers by failure rate
4. Schedule upgrade windows (low-traffic hours)

### Phase 2: Procurement (Week 2)
1. Requisition devices from central inventory
2. Verify firmware versions are current
3. Pre-configure network settings
4. Prepare installation kits

### Phase 3: Installation (Week 3-4)
For each center:
1. Arrive during scheduled window
2. Backup existing device configurations
3. Disconnect old device, install new
4. Test connectivity to CIDR servers
5. Perform 10 test captures (5 fingerprint, 5 iris)
6. Train operators on new interface
7. Document installation in asset register

### Phase 4: Monitoring (Week 5+)
1. Track failure rates daily for first week
2. Compare to pre-upgrade baseline
3. Address any anomalies immediately
4. Report metrics to pilot_monitor dashboard

---

## Technical Specifications

| Component | Requirement |
|-----------|-------------|
| Fingerprint Scanner | 500 DPI, FBI PIV certified |
| Iris Camera | NIR illumination, 640x480 min |
| Network | 1 Mbps minimum uplink |
| Power | UPS backup required |

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Capture Success Rate | 85% | 95%+ |
| Average Capture Time | 45 sec | 30 sec |
| Device Downtime | 5%/week | <1%/week |

---

## Escalation Contacts

| Issue | Contact |
|-------|---------|
| Hardware Failure | Vendor Support |
| Network Issues | IT Helpdesk |
| Training Gaps | District Trainer |

---

## Post-Upgrade Validation

- [ ] All centers operational
- [ ] Failure rate decreased
- [ ] Operators trained and comfortable
- [ ] Asset register updated
- [ ] Pilot dashboard reflecting new data
