# Canonical Schema: District-Week Master Table

## Overview
The canonical dataset aggregates all three source datasets (Enrolment, Demographic Updates, Biometric Updates) into a single **district-week** grain table for analysis.

## Grain
- **Primary Keys**: `state`, `district`, `year`, `week_number`
- **Granularity**: One row per state-district-year-week combination

## Schema Definition

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `state` | string | Indian state name | All datasets |
| `district` | string | District name | All datasets |
| `year` | int | Calendar year (2025) | Derived from date |
| `week_number` | int | ISO week number (1-52) | Derived from date |
| `enroll_child` | int | Total enrolments age 5-17 | Enrolment.age_5_17 |
| `enroll_total` | int | Total enrolments all ages | Enrolment (sum of all age cols) |
| `demo_update_child` | int | Demographic updates age 5-17 | Demographic.demo_age_5_17 |
| `demo_update_total` | int | Total demographic updates | Demographic (sum of all cols) |
| `bio_update_child` | int | Biometric updates age 5-17 | Biometric.bio_age_5_17 |
| `bio_update_total` | int | Total biometric updates | Biometric (sum of all cols) |
| `bio_demo_ratio_child` | float | bio_update_child / demo_update_child | Computed |
| `bio_demo_gap_child` | int | demo_update_child - bio_update_child | Computed |
| `lag1_bio_update_child` | int | Previous week's bio_update_child | Lag feature |
| `lag1_bio_demo_ratio` | float | Previous week's ratio | Lag feature |

## Merge Logic
1. Parse all dates to datetime (`DD-MM-YYYY` → datetime)
2. Extract `year` and `week_number` from date
3. Aggregate each dataset to state-district-year-week
4. Outer merge on `[state, district, year, week_number]`
5. Fill missing values with 0 for counts
6. Compute derived metrics

## Output
- **File**: `data/processed/master.parquet`
- **Format**: Parquet (efficient for large data)
- **Estimated rows**: ~50,000-100,000 (depending on coverage)

## Notes
- Districts not present in all datasets will have 0 for missing metrics
- The 5-17 age bucket is used as proxy for 5-15 (closest available)
