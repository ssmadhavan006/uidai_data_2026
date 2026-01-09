"""
03_anomaly_detection.py
Anomaly detection for identifying operational stress in districts.

Methods:
- Isolation Forest for multi-dimensional anomalies
- S-H-ESD (Seasonal Hybrid ESD) for time-series anomalies
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def load_features(path: str = 'data/processed/model_features.parquet') -> pd.DataFrame:
    """Load the feature dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def isolation_forest_anomalies(
    df: pd.DataFrame,
    feature_cols: list,
    contamination: float = 0.05
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.
    
    Args:
        df: DataFrame with features
        feature_cols: Columns to use for anomaly detection
        contamination: Expected proportion of anomalies
    
    Returns:
        DataFrame with anomaly scores and labels
    """
    df = df.copy()
    
    # Prepare features
    X = df[feature_cols].copy()
    
    # Handle missing and suppressed values
    X = X.replace(-1, np.nan)
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    # Predict: -1 for anomalies, 1 for normal
    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    
    # Add to dataframe
    df['anomaly_label'] = predictions
    df['anomaly_score'] = scores
    df['is_anomaly'] = (predictions == -1).astype(int)
    
    return df


def seasonal_esd_anomalies(
    df: pd.DataFrame,
    value_col: str,
    max_anomalies: int = 5,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Detect anomalies using Seasonal Hybrid ESD test.
    
    Uses seasonal decomposition + residual analysis.
    """
    df = df.copy()
    
    if not STATSMODELS_AVAILABLE:
        print("statsmodels not available for S-H-ESD")
        df['shesd_anomaly'] = 0
        return df
    
    # Sort by time
    df = df.sort_values(['year', 'week_number'])
    
    # Get values (handle suppressed -1)
    values = df[value_col].replace(-1, np.nan).fillna(method='ffill').fillna(0)
    
    if len(values) < 8:
        df['shesd_anomaly'] = 0
        return df
    
    try:
        # Seasonal decomposition
        decomposition = seasonal_decompose(
            values, 
            model='additive', 
            period=min(4, len(values) // 2),
            extrapolate_trend='freq'
        )
        
        residuals = decomposition.resid.dropna()
        
        if len(residuals) == 0:
            df['shesd_anomaly'] = 0
            return df
        
        # Generalized ESD test on residuals
        median = residuals.median()
        mad = np.median(np.abs(residuals - median))
        
        if mad == 0:
            mad = residuals.std()
        
        # Modified Z-score
        modified_z = 0.6745 * (residuals - median) / (mad if mad > 0 else 1)
        
        # Flag extreme values
        threshold = 3.5  # Common threshold for modified Z-score
        anomaly_idx = residuals.index[np.abs(modified_z) > threshold]
        
        df['shesd_anomaly'] = 0
        df.loc[df.index.isin(anomaly_idx), 'shesd_anomaly'] = 1
        
    except Exception as e:
        print(f"S-H-ESD error: {e}")
        df['shesd_anomaly'] = 0
    
    return df


def detect_district_anomalies(
    df: pd.DataFrame,
    state: str,
    district: str
) -> pd.DataFrame:
    """
    Detect anomalies for a specific district using S-H-ESD.
    """
    subset = df[(df['state'] == state) & (df['district'] == district)].copy()
    
    if subset.empty or len(subset) < 4:
        subset['shesd_bio_anomaly'] = 0
        subset['shesd_demo_anomaly'] = 0
        return subset
    
    # Bio updates
    subset = seasonal_esd_anomalies(subset, 'bio_update_child')
    subset = subset.rename(columns={'shesd_anomaly': 'shesd_bio_anomaly'})
    
    # Demo updates
    subset = seasonal_esd_anomalies(subset, 'demo_update_child')
    subset = subset.rename(columns={'shesd_anomaly': 'shesd_demo_anomaly'})
    
    return subset


def run_anomaly_detection(
    input_path: str = 'data/processed/model_features.parquet',
    output_path: str = 'outputs/anomalies.csv'
) -> pd.DataFrame:
    """
    Run complete anomaly detection pipeline.
    """
    print("=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = load_features(input_path)
    
    # Define features for Isolation Forest
    feature_cols = [
        'bio_update_child', 'demo_update_child', 'enroll_child',
        'update_backlog_child', 'completion_rate_child'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Isolation Forest on all data
    print("\n[2/4] Running Isolation Forest...")
    df = isolation_forest_anomalies(df, feature_cols, contamination=0.05)
    n_if_anomalies = df['is_anomaly'].sum()
    print(f"  Detected {n_if_anomalies:,} anomalies ({100*n_if_anomalies/len(df):.2f}%)")
    
    # S-H-ESD per district (for most recent week focus)
    print("\n[3/4] Running S-H-ESD per district...")
    districts = df[['state', 'district']].drop_duplicates()
    
    shesd_results = []
    for _, row in districts.iterrows():
        result = detect_district_anomalies(df, row['state'], row['district'])
        if not result.empty:
            shesd_results.append(result)
    
    if shesd_results:
        df_shesd = pd.concat(shesd_results, ignore_index=True)
        # Merge back
        df = df.merge(
            df_shesd[['state', 'district', 'year', 'week_number', 'shesd_bio_anomaly', 'shesd_demo_anomaly']],
            on=['state', 'district', 'year', 'week_number'],
            how='left'
        )
        df['shesd_bio_anomaly'] = df['shesd_bio_anomaly'].fillna(0).astype(int)
        df['shesd_demo_anomaly'] = df['shesd_demo_anomaly'].fillna(0).astype(int)
    else:
        df['shesd_bio_anomaly'] = 0
        df['shesd_demo_anomaly'] = 0
    
    n_shesd_anomalies = df['shesd_bio_anomaly'].sum() + df['shesd_demo_anomaly'].sum()
    print(f"  S-H-ESD flagged {n_shesd_anomalies:,} anomalies")
    
    # Combined anomaly score
    df['combined_anomaly'] = (
        (df['is_anomaly'] == 1) | 
        (df['shesd_bio_anomaly'] == 1) | 
        (df['shesd_demo_anomaly'] == 1)
    ).astype(int)
    
    # Get most recent week anomalies
    print("\n[4/4] Saving results...")
    latest_week = df.groupby(['state', 'district']).tail(1)
    anomalies = latest_week[latest_week['combined_anomaly'] == 1]
    
    # Output columns
    output_cols = [
        'state', 'district', 'year', 'week_number',
        'bio_update_child', 'demo_update_child', 'update_backlog_child',
        'is_anomaly', 'anomaly_score', 
        'shesd_bio_anomaly', 'shesd_demo_anomaly', 'combined_anomaly'
    ]
    output_cols = [c for c in output_cols if c in anomalies.columns]
    
    # Save all with anomaly flags
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    latest_week[output_cols].to_csv(output_path, index=False)
    
    print(f"Saved {len(latest_week):,} rows to {output_path}")
    print(f"Total anomalies in latest week: {anomalies['combined_anomaly'].sum()}")
    
    # Also save just anomalies
    anomaly_only_path = output_path.replace('.csv', '_flagged.csv')
    anomalies[output_cols].to_csv(anomaly_only_path, index=False)
    print(f"Saved {len(anomalies):,} anomalous districts to {anomaly_only_path}")
    
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 60)
    
    return df


if __name__ == '__main__':
    df = run_anomaly_detection()
    print(f"\nSample anomalies:")
    print(df[df['combined_anomaly'] == 1][['state', 'district', 'bio_update_child', 'anomaly_score']].head(10))
