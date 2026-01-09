"""
Phase 1 Verification Script
Runs ETL and QC logic to generate all required outputs.
"""
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

print("=" * 60)
print("PHASE 1 VERIFICATION")
print("=" * 60)

# --- 1. Load Data ---
def load_dataset(pattern):
    files = glob.glob(os.path.join('data/raw', pattern))
    print(f"Loading {len(files)} files for '{pattern}'")
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("\n[1/6] Loading datasets...")
enrolment = load_dataset('*enrolment*.csv')
demo_update = load_dataset('*demographic*.csv')
bio_update = load_dataset('*biometric*.csv')
print(f"Loaded: Enrolment={len(enrolment):,}, Demo={len(demo_update):,}, Bio={len(bio_update):,}")

# --- 2. Parse Dates ---
print("\n[2/6] Parsing dates...")
for df in [enrolment, demo_update, bio_update]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year.astype('Int64')
    df['week_number'] = df['date'].dt.isocalendar().week.astype('Int64')
print("Done.")

# --- 3. Identify Age Columns ---
print("\n[3/6] Identifying age columns...")
enrol_age_cols = [c for c in enrolment.columns if 'age' in c.lower()]
demo_age_cols = [c for c in demo_update.columns if 'age' in c.lower()]
bio_age_cols = [c for c in bio_update.columns if 'age' in c.lower()]

enrol_child_col = next((c for c in enrol_age_cols if '5' in c and '17' in c), None)
demo_child_col = next((c for c in demo_age_cols if '5' in c and '17' in c), None)
bio_child_col = next((c for c in bio_age_cols if '5' in c and '17' in c), None)

print(f"Child columns: Enrol={enrol_child_col}, Demo={demo_child_col}, Bio={bio_child_col}")

# --- 4. Aggregate to District-Week ---
print("\n[4/6] Aggregating to district-week...")
keys = ['state', 'district', 'year', 'week_number']

enrol_num = [c for c in enrolment.select_dtypes(include=[np.number]).columns if c not in ['year', 'week_number']]
demo_num = [c for c in demo_update.select_dtypes(include=[np.number]).columns if c not in ['year', 'week_number']]
bio_num = [c for c in bio_update.select_dtypes(include=[np.number]).columns if c not in ['year', 'week_number']]

enrol_agg = enrolment.groupby(keys, as_index=False)[enrol_num].sum()
demo_agg = demo_update.groupby(keys, as_index=False)[demo_num].sum()
bio_agg = bio_update.groupby(keys, as_index=False)[bio_num].sum()

# Create final columns
enrol_agg['enroll_total'] = enrol_agg[enrol_num].sum(axis=1)
enrol_agg['enroll_child'] = enrol_agg[enrol_child_col] if enrol_child_col else 0
enrol_final = enrol_agg[keys + ['enroll_child', 'enroll_total']]

demo_agg['demo_update_total'] = demo_agg[demo_num].sum(axis=1)
demo_agg['demo_update_child'] = demo_agg[demo_child_col] if demo_child_col else 0
demo_final = demo_agg[keys + ['demo_update_child', 'demo_update_total']]

bio_agg['bio_update_total'] = bio_agg[bio_num].sum(axis=1)
bio_agg['bio_update_child'] = bio_agg[bio_child_col] if bio_child_col else 0
bio_final = bio_agg[keys + ['bio_update_child', 'bio_update_total']]

# Merge
master = enrol_final.merge(demo_final, on=keys, how='outer')
master = master.merge(bio_final, on=keys, how='outer')

count_cols = ['enroll_child', 'enroll_total', 'demo_update_child', 'demo_update_total', 'bio_update_child', 'bio_update_total']
master[count_cols] = master[count_cols].fillna(0).astype(int)

# Derived metrics
master['bio_demo_ratio_child'] = master['bio_update_child'] / master['demo_update_child'].replace(0, np.nan)
master['bio_demo_gap_child'] = master['demo_update_child'] - master['bio_update_child']
master = master.sort_values(['state', 'district', 'year', 'week_number']).reset_index(drop=True)
master['lag1_bio_update_child'] = master.groupby(['state', 'district'])['bio_update_child'].shift(1)
master['lag1_bio_demo_ratio'] = master.groupby(['state', 'district'])['bio_demo_ratio_child'].shift(1)

print(f"Master dataset: {len(master):,} rows, {len(master.columns)} columns")

# --- 5. Save master.parquet ---
print("\n[5/6] Saving master.parquet...")
os.makedirs('data/processed', exist_ok=True)
master.to_parquet('data/processed/master.parquet', index=False)
print(f"Saved to data/processed/master.parquet")

# --- 6. Generate QC outputs ---
print("\n[6/6] Generating QC outputs...")

# Data inventory
os.makedirs('data/inventory', exist_ok=True)
inventory = pd.DataFrame({
    'dataset': ['Enrolment', 'Demographic', 'Biometric'],
    'rows': [len(enrolment), len(demo_update), len(bio_update)],
    'date_min': [enrolment['date'].min(), demo_update['date'].min(), bio_update['date'].min()],
    'date_max': [enrolment['date'].max(), demo_update['date'].max(), bio_update['date'].max()],
    'states': [enrolment['state'].nunique(), demo_update['state'].nunique(), bio_update['state'].nunique()],
    'districts': [enrolment['district'].nunique(), demo_update['district'].nunique(), bio_update['district'].nunique()],
})
inventory.to_csv('data/inventory/data_inventory.csv', index=False)
print("Saved data/inventory/data_inventory.csv")

# Data issues
enrol_districts = set(enrolment['district'].unique())
demo_districts = set(demo_update['district'].unique())
bio_districts = set(bio_update['district'].unique())
common = enrol_districts & demo_districts & bio_districts
mismatch_count = len(enrol_districts | demo_districts | bio_districts) - len(common)

# Find districts with zero bio updates for 4+ consecutive weeks
zero_weeks = master.groupby(['state', 'district']).apply(
    lambda x: ((x['bio_update_child'] == 0).rolling(4).sum() >= 4).any()
).reset_index(name='has_zero_streak')
districts_with_zero_streak = zero_weeks[zero_weeks['has_zero_streak']]['district'].tolist()

with open('data/inventory/data_issues.md', 'w') as f:
    f.write('# Data Issues Log\n\n')
    f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write('## Issues Identified\n\n')
    f.write(f'1. **Date range mismatch**: Enrolment ({enrolment["date"].min().date()} to {enrolment["date"].max().date()}), ')
    f.write(f'Biometric ({bio_update["date"].min().date()} to {bio_update["date"].max().date()})\n\n')
    f.write(f'2. **District name mismatches**: {mismatch_count} districts not present in all 3 datasets\n\n')
    f.write(f'3. **Age bucket alignment**: Enrolment has 0-5 age group with no equivalent in update datasets\n\n')
    f.write(f'4. **Districts with zero bio updates for 4+ consecutive weeks**: {len(districts_with_zero_streak)} districts\n')
    if districts_with_zero_streak[:10]:
        f.write(f'   - Sample: {districts_with_zero_streak[:10]}\n\n')
    f.write(f'5. **Potential bottleneck districts** (bio/demo ratio < 0.5): {(master["bio_demo_ratio_child"] < 0.5).sum()} district-weeks\n\n')

print("Saved data/inventory/data_issues.md")

print("\n" + "=" * 60)
print("PHASE 1 VERIFICATION COMPLETE")
print("=" * 60)
print(f"\nOutputs generated:")
print(f"  - data/processed/master.parquet ({len(master):,} rows)")
print(f"  - data/inventory/data_inventory.csv")
print(f"  - data/inventory/data_issues.md")
