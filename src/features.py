"""
features.py
Feature engineering module for modeling.

Creates:
- Temporal features (lags, rolling means)
- Performance features (failure rate, backlog)
- Output: model_features.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_canonical_data(path: str = 'data/processed/master.parquet') -> pd.DataFrame:
    """Load the canonical district-week dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal lag and rolling features.
    
    Features created:
    - lag_1w_bio_update_child: Last week's bio updates
    - lag_2w_bio_update_child: 2 weeks ago bio updates
    - rolling_4w_mean_bio_update_child: 4-week rolling average
    - lag_1w_enroll_child: Last week's enrollments
    - rolling_4w_mean_enroll_child: 4-week rolling enrollment average
    """
    df = df.copy()
    
    # Sort by district and time
    df = df.sort_values(['state', 'district', 'year', 'week_number']).reset_index(drop=True)
    
    # Group by district for lag calculations
    grouped = df.groupby(['state', 'district'])
    
    # Bio update lags
    df['lag_1w_bio_update_child'] = grouped['bio_update_child'].shift(1)
    df['lag_2w_bio_update_child'] = grouped['bio_update_child'].shift(2)
    
    # Bio update rolling mean (4-week window)
    df['rolling_4w_mean_bio_update_child'] = grouped['bio_update_child'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    
    # Enrollment lags
    df['lag_1w_enroll_child'] = grouped['enroll_child'].shift(1)
    df['rolling_4w_mean_enroll_child'] = grouped['enroll_child'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    
    # Demo update lags
    df['lag_1w_demo_update_child'] = grouped['demo_update_child'].shift(1)
    df['rolling_4w_mean_demo_update_child'] = grouped['demo_update_child'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    
    # Week-over-week change
    df['wow_change_bio_update_child'] = df['bio_update_child'] - df['lag_1w_bio_update_child']
    
    # Trend indicator (positive = increasing, negative = decreasing)
    df['trend_bio_update_child'] = np.sign(df['wow_change_bio_update_child'])
    
    print(f"Added temporal features: lags, rolling means, trends")
    return df


def add_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add performance and system features.
    
    Features created:
    - failure_rate_child: 1 - (bio_update / bio_attempts)
    - update_backlog_child: demo_update - bio_update
    - saturation_proxy: enroll_child / enroll_total
    - child_focus_ratio: bio_update_child / bio_update_total
    """
    df = df.copy()
    
    # Failure rate (using attempts if available, else approximate)
    if 'bio_attempts_child' in df.columns:
        df['failure_rate_child'] = 1 - (
            df['bio_update_child'] / df['bio_attempts_child'].replace(0, np.nan)
        )
    else:
        # If no attempts data, assume all attempts succeed (failure_rate = 0)
        df['failure_rate_child'] = 0.0
    
    # Update backlog (demo requested - bio completed)
    df['update_backlog_child'] = df['demo_update_child'] - df['bio_update_child']
    
    # Saturation proxy (child proportion of total)
    df['saturation_proxy'] = df['enroll_child'] / df['enroll_total'].replace(0, np.nan)
    
    # Child focus ratio (how much of bio updates are children)
    df['child_focus_ratio'] = df['bio_update_child'] / df['bio_update_total'].replace(0, np.nan)
    
    # Completion rate (bio / demo) - MUST BE CLIPPED TO 0-1 RANGE
    raw_completion = df['bio_update_child'] / df['demo_update_child'].replace(0, np.nan)
    # Clip to valid range: can't be negative or > 100%
    df['completion_rate_child'] = raw_completion.clip(0, 1)
    
    # Backlog severity (normalized)
    max_backlog = df['update_backlog_child'].abs().max()
    if max_backlog > 0:
        df['backlog_severity'] = df['update_backlog_child'] / max_backlog
    else:
        df['backlog_severity'] = 0.0
    
    print(f"Added performance features: failure_rate, backlog, saturation")
    return df


def add_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite priority score for intervention ranking.
    
    Formula:
    priority_score = w1*backlog + w2*(1-completion_rate) + w3*volume
    
    Higher score = higher priority for intervention
    """
    df = df.copy()
    
    # Normalize components to 0-1 scale
    def minmax_scale(series):
        min_val = series.min()
        max_val = series.max()
        if pd.isna(max_val) or pd.isna(min_val) or max_val - min_val == 0:
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # Handle suppressed values (-1) and NaN
    backlog = df['update_backlog_child'].clip(lower=0)  # Only positive backlog
    completion = df['completion_rate_child'].fillna(0).clip(0, 1)
    volume = df['enroll_child'].clip(lower=0)
    
    # Normalize
    backlog_norm = minmax_scale(backlog)
    incompletion_norm = 1 - completion
    volume_norm = minmax_scale(volume)
    
    # Weights (configurable)
    w_backlog = 0.4
    w_incompletion = 0.3
    w_volume = 0.3
    
    df['priority_score'] = (
        w_backlog * backlog_norm + 
        w_incompletion * incompletion_norm + 
        w_volume * volume_norm
    )
    
    # Priority rank (1 = highest priority)
    df['priority_rank'] = df.groupby(['year', 'week_number'])['priority_score'].rank(
        ascending=False, method='dense'
    )
    
    print(f"Added priority score and ranking")
    return df


def run_feature_engineering(
    input_path: str = 'data/processed/master.parquet',
    output_path: str = 'data/processed/model_features.parquet'
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading canonical data...")
    df = load_canonical_data(input_path)
    
    # Add features
    print("\n[2/4] Adding temporal features...")
    df = add_temporal_features(df)
    
    print("\n[3/4] Adding performance features...")
    df = add_performance_features(df)
    
    print("\n[4/4] Adding priority score...")
    df = add_priority_score(df)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"New features: {[c for c in df.columns if 'lag' in c or 'rolling' in c or 'priority' in c or 'backlog' in c]}")
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    
    return df


if __name__ == '__main__':
    df = run_feature_engineering()
    print(f"\nSample features:")
    print(df[['state', 'district', 'week_number', 'bio_update_child', 
              'lag_1w_bio_update_child', 'update_backlog_child', 'priority_score']].head(10))
