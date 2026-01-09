# Data Issues Log

Generated: 2026-01-09 21:12:45

## Issues Identified

1. **Date range mismatch**: Enrolment (2025-03-02 to 2025-12-31), Biometric (2025-03-01 to 2025-12-29)

2. **District name mismatches**: 81 districts not present in all 3 datasets

3. **Age bucket alignment**: Enrolment has 0-5 age group with no equivalent in update datasets

4. **Districts with zero bio updates for 4+ consecutive weeks**: 100 districts
   - Sample: ['100000', 'Bajali', 'Aurangabad(BH)', 'Bhabua', 'Monghyr', 'Purbi Champaran', 'Samstipur', 'Sitamarhi', 'Rupnagar', 'Janjgir - Champa']

5. **Potential bottleneck districts** (bio/demo ratio < 0.5): 616 district-weeks

