# Data Dictionary

## 1. Enrolment Dataset
- date: DD-MM-YYYY
- state: Indian state name
- district: District name  
- pincode: 6-digit PIN code
- age_0_5: Count of enrolments age 0-5
- age_5_17: Count of enrolments age 5-17
- age_18_greater: Count of enrolments age 18+

## 2. Demographic Update Dataset
- date: DD-MM-YYYY
- state: Indian state name
- district: District name
- pincode: 6-digit PIN code
- demo_age_5_17: Demographic updates for age 5-17
- demo_age_17_: Demographic updates for age 17+

## 3. Biometric Update Dataset
- date: DD-MM-YYYY
- state: Indian state name
- district: District name
- pincode: 6-digit PIN code
- bio_age_5_17: Biometric updates for age 5-17
- bio_age_17_: Biometric updates for age 17+

## Key Observations:
1. Age groups don't perfectly align (enrolment has 0-5, others don't)
2. Can merge on (date, state, district, pincode)
3. All counts are aggregated - no PII
