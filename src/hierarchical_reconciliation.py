"""
hierarchical_reconciliation.py
Hierarchical forecast reconciliation: state-level forecasts disaggregated to districts.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def compute_historical_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical district share of state totals.
    """
    # Aggregate to state-week
    state_totals = df.groupby(['state', 'year', 'week_number']).agg({
        'bio_update_child': 'sum',
        'enroll_child': 'sum'
    }).reset_index()
    state_totals.columns = ['state', 'year', 'week_number', 'state_bio_total', 'state_enroll_total']
    
    # Merge back
    df = df.merge(state_totals, on=['state', 'year', 'week_number'], how='left')
    
    # Compute shares
    df['district_bio_share'] = df['bio_update_child'] / df['state_bio_total'].replace(0, np.nan)
    df['district_enroll_share'] = df['enroll_child'] / df['state_enroll_total'].replace(0, np.nan)
    
    return df


def get_average_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get average district shares over the historical period.
    """
    shares = df.groupby(['state', 'district']).agg({
        'district_bio_share': 'mean',
        'district_enroll_share': 'mean'
    }).reset_index()
    shares.columns = ['state', 'district', 'avg_bio_share', 'avg_enroll_share']
    
    return shares


def reconcile_forecasts(
    features_path: str = 'data/processed/model_features.parquet',
    forecasts_path: str = 'outputs/forecasts.csv',
    output_path: str = 'outputs/forecasts_reconciled.csv'
) -> pd.DataFrame:
    """
    Reconcile district forecasts to ensure they sum to state totals.
    
    Method: Top-down proportional disaggregation
    1. Sum district forecasts to get implied state total
    2. Adjust district forecasts by historical share
    """
    print("=" * 60)
    print("HIERARCHICAL RECONCILIATION")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    features = pd.read_parquet(features_path)
    
    try:
        forecasts = pd.read_csv(forecasts_path)
    except FileNotFoundError:
        print("Forecasts not found. Run forecast_baseline.py first.")
        return pd.DataFrame()
    
    # Compute historical shares
    print("\n[2/4] Computing historical shares...")
    features = compute_historical_shares(features)
    shares = get_average_shares(features)
    print(f"  Computed shares for {len(shares)} district-state pairs")
    
    # Aggregate forecasts to state
    print("\n[3/4] Reconciling forecasts...")
    
    # Merge shares with forecasts
    forecasts = forecasts.merge(shares, on=['state', 'district'], how='left')
    
    # Get state-level implied totals (sum of district forecasts)
    state_forecast_totals = forecasts.groupby(['state', 'ds'])['yhat'].sum().reset_index()
    state_forecast_totals.columns = ['state', 'ds', 'state_forecast_total']
    
    # Merge back
    forecasts = forecasts.merge(state_forecast_totals, on=['state', 'ds'], how='left')
    
    # Reconcile: district_forecast = state_total * historical_share
    # Use average of original forecast and share-based forecast
    forecasts['yhat_share_based'] = forecasts['state_forecast_total'] * forecasts['avg_bio_share'].fillna(0)
    
    # Blend original and reconciled (weighted average)
    alpha = 0.7  # Weight for original forecast
    forecasts['yhat_reconciled'] = (
        alpha * forecasts['yhat'] + 
        (1 - alpha) * forecasts['yhat_share_based']
    )
    
    # Adjust bounds proportionally
    scale = forecasts['yhat_reconciled'] / forecasts['yhat'].replace(0, 1)
    forecasts['yhat_lower_reconciled'] = forecasts['yhat_lower'] * scale
    forecasts['yhat_upper_reconciled'] = forecasts['yhat_upper'] * scale
    
    # Validation: check sums
    print("\n[4/4] Validating reconciliation...")
    original_state_sum = forecasts.groupby(['state', 'ds'])['yhat'].sum()
    reconciled_state_sum = forecasts.groupby(['state', 'ds'])['yhat_reconciled'].sum()
    
    diff = (reconciled_state_sum - original_state_sum).abs().mean()
    print(f"  Average state sum difference: {diff:.2f}")
    
    # Select output columns
    output_cols = [
        'state', 'district', 'ds', 
        'yhat', 'yhat_reconciled', 
        'yhat_lower', 'yhat_lower_reconciled',
        'yhat_upper', 'yhat_upper_reconciled',
        'avg_bio_share'
    ]
    output_cols = [c for c in output_cols if c in forecasts.columns]
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    forecasts[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved reconciled forecasts to {output_path}")
    
    print("\n" + "=" * 60)
    print("RECONCILIATION COMPLETE")
    print("=" * 60)
    
    return forecasts[output_cols]


if __name__ == '__main__':
    reconciled = reconcile_forecasts()
    print(reconciled.head())
