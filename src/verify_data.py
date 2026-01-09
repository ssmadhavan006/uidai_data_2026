import pandas as pd
import glob
import os

def load_dataset(pattern):
    search_path = os.path.join('../data/raw', pattern)
    files = glob.glob(search_path)
    print(f"pattern '{pattern}': found {len(files)} files")
    if not files: return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("--- Verifying Data Loading ---")
# Adjust path because we run from src/ or root?
# If running from root, path is data/raw. If from src, ../data/raw.
# Let's assume running from d:\UIDAI\code, so path is data/raw
def load_dataset_root(pattern):
    search_path = os.path.join('data/raw', pattern)
    files = glob.glob(search_path)
    print(f"pattern '{pattern}': found {len(files)} files")
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

enrolment = load_dataset_root('*enrolment*.csv')
demo = load_dataset_root('*demographic*.csv')
bio = load_dataset_root('*biometric*.csv')

print(f"\nEnrolment Rows: {len(enrolment)}")
print(f"Demographic Rows: {len(demo)}")
print(f"Biometric Rows: {len(bio)}")

print("\n--- Identifying Questions ---")

# Question 1: District Consistency
if not enrolment.empty and not demo.empty:
    enr_dist = set(enrolment['district'].unique())
    demo_dist = set(demo['district'].unique())
    common = enr_dist.intersection(demo_dist)
    diff = enr_dist.symmetric_difference(demo_dist)
    print(f"Common Districts: {len(common)}")
    print(f"Districts Mismatch Count: {len(diff)}")
    if len(diff) > 0:
        print(f"Sample Mismatches: {list(diff)[:5]}")
else:
    print("Skipping district check (empty data)")

# Question 2: Date overlaps
if not enrolment.empty:
    print(f"Enrolment Date Range: {enrolment['date'].min()} to {enrolment['date'].max()}")
if not bio.empty:
    print(f"Biometric Date Range: {bio['date'].min()} to {bio['date'].max()}")

print("\nDone.")
