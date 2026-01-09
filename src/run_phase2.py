"""
run_phase2.py
Run the complete Phase 2 pipeline: ETL + Features
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from privacy_guard import sanitize_dataframe, validate_privacy, apply_k_anonymity

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("PHASE 2: ETL + FEATURES PIPELINE")
print("=" * 60)

# === ETL SECTION ===
print("\n" + "=" * 40)
print("STEP 1: ETL PIPELINE")
print("=" * 40)

# Load data
def load_dataset(pattern):
    files = glob.glob(os.path.join('data/raw', pattern))
    print(f"Loading {len(files)} files for '{pattern}'")
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("\n[1/6] Loading raw data...")
enrolment = load_dataset('*enrolment*.csv')
demographic = load_dataset('*demographic*.csv')
biometric = load_dataset('*biometric*.csv')
print(f"Loaded: Enrolment={len(enrolment):,}, Demo={len(demographic):,}, Bio={len(biometric):,}")

# Parse dates
print("\n[2/6] Parsing dates...")
for df in [enrolment, demographic, biometric]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year.astype('Int64')
    df['week_number'] = df['date'].dt.isocalendar().week.astype('Int64')

# Standardize names
print("\n[3/6] Standardizing names...")
DISTRICT_NORMALIZATION = {
    'Bangalore Urban': 'Bengaluru Urban',
    'Bangalore Rural': 'Bengaluru Rural',
    'Mysore': 'Mysuru',
}
for df in [enrolment, demographic, biometric]:
    df['district'] = df['district'].replace(DISTRICT_NORMALIZATION)
    df['district'] = df['district'].str.strip().str.replace(r'\s*\*\s*', '', regex=True)
    df['state'] = df['state'].str.strip()

# Identify age columns
def get_child_col(df):
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    child = [c for c in age_cols if '5' in c and '17' in c]
    return child[0] if child else None

enrol_child = get_child_col(enrolment)
demo_child = get_child_col(demographic)
bio_child = get_child_col(biometric)
print(f"Child columns: Enrol={enrol_child}, Demo={demo_child}, Bio={bio_child}")

# Aggregate
print("\n[4/6] Aggregating to district-week...")
keys = ['state', 'district', 'year', 'week_number']

def get_num_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns 
            if c not in ['year', 'week_number', 'pincode']]

enrol_num = get_num_cols(enrolment)
demo_num = get_num_cols(demographic)
bio_num = get_num_cols(biometric)

enrol_agg = enrolment.groupby(keys, as_index=False)[enrol_num].sum()
demo_agg = demographic.groupby(keys, as_index=False)[demo_num].sum()
bio_agg = biometric.groupby(keys, as_index=False)[bio_num].sum()

enrol_agg['enroll_total'] = enrol_agg[enrol_num].sum(axis=1)
enrol_agg['enroll_child'] = enrol_agg[enrol_child] if enrol_child else 0
demo_agg['demo_update_total'] = demo_agg[demo_num].sum(axis=1)
demo_agg['demo_update_child'] = demo_agg[demo_child] if demo_child else 0
bio_agg['bio_update_total'] = bio_agg[bio_num].sum(axis=1)
bio_agg['bio_update_child'] = bio_agg[bio_child] if bio_child else 0
bio_agg['bio_attempts_child'] = bio_agg['bio_update_child']

enrol_final = enrol_agg[keys + ['enroll_child', 'enroll_total']]
demo_final = demo_agg[keys + ['demo_update_child', 'demo_update_total']]
bio_final = bio_agg[keys + ['bio_update_child', 'bio_update_total', 'bio_attempts_child']]

master = enrol_final.merge(demo_final, on=keys, how='outer')
master = master.merge(bio_final, on=keys, how='outer')

count_cols = ['enroll_child', 'enroll_total', 'demo_update_child', 
              'demo_update_total', 'bio_update_child', 'bio_update_total', 'bio_attempts_child']
master[count_cols] = master[count_cols].fillna(0).astype(int)
print(f"Aggregated shape: {master.shape}")

# Apply privacy
print("\n[5/6] Applying privacy guard (k=10)...")
master = sanitize_dataframe(master, k=10, count_columns=count_cols)
report = validate_privacy(master, k=10)
print(f"Privacy validation: {'PASS' if report['valid'] else 'FAIL'}")

# Derived metrics
master['bio_demo_ratio_child'] = master['bio_update_child'] / master['demo_update_child'].replace(0, np.nan)
master['bio_demo_gap_child'] = master['demo_update_child'] - master['bio_update_child']
master = master.sort_values(['state', 'district', 'year', 'week_number']).reset_index(drop=True)

# Save
print("\n[6/6] Saving master.parquet...")
Path('data/processed').mkdir(parents=True, exist_ok=True)
master.to_parquet('data/processed/master.parquet', index=False)
print(f"Saved: data/processed/master.parquet ({len(master):,} rows)")

# === FEATURES SECTION ===
print("\n" + "=" * 40)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 40)

grouped = master.groupby(['state', 'district'])

# Temporal features
print("\n[1/3] Adding temporal features...")
master['lag_1w_bio_update_child'] = grouped['bio_update_child'].shift(1)
master['lag_2w_bio_update_child'] = grouped['bio_update_child'].shift(2)
master['rolling_4w_mean_bio_update_child'] = grouped['bio_update_child'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
master['lag_1w_enroll_child'] = grouped['enroll_child'].shift(1)
master['wow_change_bio_update_child'] = master['bio_update_child'] - master['lag_1w_bio_update_child']

# Performance features
print("\n[2/3] Adding performance features...")
master['failure_rate_child'] = 0.0  # No separate attempts data
master['update_backlog_child'] = master['demo_update_child'] - master['bio_update_child']
master['saturation_proxy'] = master['enroll_child'] / master['enroll_total'].replace(0, np.nan)
master['completion_rate_child'] = master['bio_update_child'] / master['demo_update_child'].replace(0, np.nan)

# Priority score
print("\n[3/3] Adding priority score...")
def minmax_scale(series):
    min_val, max_val = series.min(), series.max()
    if max_val - min_val == 0:
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)

backlog = master['update_backlog_child'].clip(lower=0)
completion = master['completion_rate_child'].fillna(0).clip(0, 1)
volume = master['enroll_child'].clip(lower=0)

master['priority_score'] = (
    0.4 * minmax_scale(backlog) + 
    0.3 * (1 - completion) + 
    0.3 * minmax_scale(volume)
)

# Save features
master.to_parquet('data/processed/model_features.parquet', index=False)
print(f"\nSaved: data/processed/model_features.parquet ({len(master):,} rows)")

# === VALIDATION ===
print("\n" + "=" * 40)
print("VALIDATION")
print("=" * 40)

print(f"\nRow counts:")
print(f"  - Raw Enrolment: {len(enrolment):,}")
print(f"  - Raw Demographic: {len(demographic):,}")
print(f"  - Raw Biometric: {len(biometric):,}")
print(f"  - Aggregated Master: {len(master):,}")

print(f"\nSuppression log:")
log_path = 'data/inventory/suppression_log.csv'
if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
    print(f"  - {len(log_df)} cells suppressed")
else:
    print("  - No suppressions logged")

print(f"\nFeature columns added:")
new_cols = [c for c in master.columns if 'lag' in c or 'rolling' in c or 'priority' in c or 'backlog' in c]
print(f"  - {new_cols}")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
