"""
agg_etl.py
ETL pipeline to transform raw CSVs into canonical district-week dataset.

Steps:
1. Load & standardize (district name normalization)
2. Map age buckets (5-17 as proxy for 5-15)
3. Aggregate to district-week
4. Apply privacy guard
5. Save sanitized master.parquet
"""
import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path
from datetime import datetime

# Import privacy guard
from src.privacy_guard import sanitize_dataframe, validate_privacy


# District name normalization mapping
DISTRICT_NORMALIZATION = {
    'Bangalore Urban': 'Bengaluru Urban',
    'Bangalore Rural': 'Bengaluru Rural',
    'Mysore': 'Mysuru',
    'Mangalore': 'Mangaluru',
    'Bellary': 'Ballari',
    'Shimoga': 'Shivamogga',
    'Tumkur': 'Tumakuru',
    'Gulbarga': 'Kalaburagi',
    'Bijapur': 'Vijayapura',
    'Belgaum': 'Belagavi',
    'Bombay': 'Mumbai',
    'Calcutta': 'Kolkata',
    'Madras': 'Chennai',
    'Trivandrum': 'Thiruvananthapuram',
    'Pondicherry': 'Puducherry',
    # Add more as discovered
}


def load_raw_data(data_dir: str = 'data/raw') -> tuple:
    """
    Load all raw CSV files.
    
    Returns:
        Tuple of (enrolment_df, demographic_df, biometric_df)
    """
    def load_pattern(pattern):
        files = glob.glob(os.path.join(data_dir, pattern))
        print(f"Loading {len(files)} files for '{pattern}'")
        if not files:
            print(f"  ⚠️ Warning: No files found for pattern '{pattern}'")
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)    
    enrolment = load_pattern('*enrolment*.csv')
    demographic = load_pattern('*demographic*.csv')
    biometric = load_pattern('*biometric*.csv')
    
    return enrolment, demographic, biometric


def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize district and state names.
    """
    df = df.copy()
    
    # Apply district normalization
    df['district'] = df['district'].replace(DISTRICT_NORMALIZATION)
    
    # Clean whitespace
    df['district'] = df['district'].str.strip()
    df['state'] = df['state'].str.strip()
    
    # Remove special characters like asterisks
    df['district'] = df['district'].str.replace(r'\s*\*\s*', '', regex=True)
    
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date column and extract year/week.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year.astype('Int64')
    df['week_number'] = df['date'].dt.isocalendar().week.astype('Int64')
    return df


def identify_age_columns(df: pd.DataFrame) -> dict:
    """
    Identify age-related columns in a DataFrame.
    
    Returns:
        Dict with 'child' and 'all' column lists
    """
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    
    # Child column (5-17 as proxy for 5-15)
    child_cols = [c for c in age_cols if '5' in c and '17' in c]
    
    return {
        'child': child_cols[0] if child_cols else None,
        'all': age_cols
    }


def aggregate_to_district_week(
    enrolment: pd.DataFrame,
    demographic: pd.DataFrame,
    biometric: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate all datasets to district-week grain.
    """
    keys = ['state', 'district', 'year', 'week_number']
    
    # Get age columns
    enrol_age = identify_age_columns(enrolment)
    demo_age = identify_age_columns(demographic)
    bio_age = identify_age_columns(biometric)
    
    # Get numeric columns for aggregation
    def get_num_cols(df):
        return [c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in ['year', 'week_number', 'pincode']]
    
    enrol_num = get_num_cols(enrolment)
    demo_num = get_num_cols(demographic)
    bio_num = get_num_cols(biometric)
    
    # Aggregate
    enrol_agg = enrolment.groupby(keys, as_index=False)[enrol_num].sum()
    demo_agg = demographic.groupby(keys, as_index=False)[demo_num].sum()
    bio_agg = biometric.groupby(keys, as_index=False)[bio_num].sum()
    
    # Create standardized columns
    enrol_agg['enroll_total'] = enrol_agg[enrol_num].sum(axis=1)
    enrol_agg['enroll_child'] = enrol_agg[enrol_age['child']] if enrol_age['child'] else 0
    
    demo_agg['demo_update_total'] = demo_agg[demo_num].sum(axis=1)
    demo_agg['demo_update_child'] = demo_agg[demo_age['child']] if demo_age['child'] else 0
    
    bio_agg['bio_update_total'] = bio_agg[bio_num].sum(axis=1)
    bio_agg['bio_update_child'] = bio_agg[bio_age['child']] if bio_age['child'] else 0
    
    # Also capture attempts if available (using total as proxy)
    bio_agg['bio_attempts_child'] = bio_agg['bio_update_child']  # Will refine if data available
    
    # Select final columns
    enrol_final = enrol_agg[keys + ['enroll_child', 'enroll_total']]
    demo_final = demo_agg[keys + ['demo_update_child', 'demo_update_total']]
    bio_final = bio_agg[keys + ['bio_update_child', 'bio_update_total', 'bio_attempts_child']]
    
    # Merge
    master = enrol_final.merge(demo_final, on=keys, how='outer')
    master = master.merge(bio_final, on=keys, how='outer')
    
    # Fill NaN with 0
    count_cols = [c for c in master.columns if c not in keys]
    master[count_cols] = master[count_cols].fillna(0).astype(int)
    
    return master


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed metrics to the master DataFrame.
    """
    df = df.copy()
    
    # Bio/Demo ratio for children
    df['bio_demo_ratio_child'] = df['bio_update_child'] / df['demo_update_child'].replace(0, np.nan)
    
    # Gap metric (backlog proxy)
    df['bio_demo_gap_child'] = df['demo_update_child'] - df['bio_update_child']
    
    # Saturation proxy
    df['saturation_proxy'] = df['enroll_child'] / df['enroll_total'].replace(0, np.nan)
    
    # Sort for lag calculation
    df = df.sort_values(['state', 'district', 'year', 'week_number']).reset_index(drop=True)
    
    return df


def run_etl(
    data_dir: str = 'data/raw',
    output_path: str = 'data/processed/master.parquet',
    apply_privacy: bool = True,
    k: int = 10
) -> pd.DataFrame:
    """
    Run the full ETL pipeline.
    
    Args:
        data_dir: Path to raw data
        output_path: Path for output parquet
        apply_privacy: Whether to apply privacy guard
        k: K-anonymity threshold
    
    Returns:
        Processed DataFrame
    """
    print("=" * 60)
    print("AADHAAR ETL PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    print("\n[1/6] Loading raw data...")
    enrolment, demographic, biometric = load_raw_data(data_dir)
    print(f"Loaded: Enrolment={len(enrolment):,}, Demo={len(demographic):,}, Bio={len(biometric):,}")
    
    # Step 2: Standardize
    print("\n[2/6] Standardizing names...")
    enrolment = standardize_names(enrolment)
    demographic = standardize_names(demographic)
    biometric = standardize_names(biometric)
    
    # Step 3: Parse dates
    print("\n[3/6] Parsing dates...")
    enrolment = parse_dates(enrolment)
    demographic = parse_dates(demographic)
    biometric = parse_dates(biometric)
    
    # Step 4: Aggregate
    print("\n[4/6] Aggregating to district-week...")
    master = aggregate_to_district_week(enrolment, demographic, biometric)
    print(f"Aggregated shape: {master.shape}")
    
    # Step 5: Derived metrics
    print("\n[5/6] Computing derived metrics...")
    master = add_derived_metrics(master)
    
    # Step 6: Apply privacy
    if apply_privacy:
        print(f"\n[6/6] Applying privacy guard (k={k})...")
        count_cols = ['enroll_child', 'enroll_total', 'demo_update_child', 
                      'demo_update_total', 'bio_update_child', 'bio_update_total',
                      'bio_attempts_child']
        master = sanitize_dataframe(master, k=k, count_columns=count_cols)
        
        # Validate
        report = validate_privacy(master, k=k)
        if report['valid']:
            print("Privacy validation: PASS")
        else:
            print("Privacy validation: FAIL")
            print(f"Violations: {report.get('violations', 'Unknown')}")
            raise ValueError(f"Privacy validation failed with k={k}")    else:
        print("\n[6/6] Skipping privacy guard (disabled)...")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {master.shape}")
    
    print("\n" + "=" * 60)
    print("ETL COMPLETE")
    print("=" * 60)
    
    return master


if __name__ == '__main__':
    # Run from project root
    import sys
    sys.path.insert(0, '.')
    
    master = run_etl()
    print(f"\nSample data:")
    print(master.head())
