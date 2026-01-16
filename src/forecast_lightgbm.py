"""
forecast_lightgbm.py
Advanced hierarchical forecasting using LightGBM with time-series cross-validation.

Features:
- District/state as categorical features
- Temporal features (lags, rolling means)
- Calendar features (week, month, quarter)
- Hierarchical reconciliation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_features(path: str = 'data/processed/model_features.parquet') -> pd.DataFrame:
    """Load the feature dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features for time-series modeling."""
    df = df.copy()
    
    # Month and quarter from week number (approximate)
    df['month'] = ((df['week_number'] - 1) // 4 + 1).clip(1, 12)
    df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
    
    # Cyclical encoding for week
    df['week_sin'] = np.sin(2 * np.pi * df['week_number'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_number'] / 52)
    
    # Holiday proxy (school vacation periods in India: May-June, Oct-Nov, Dec)
    school_break_weeks = list(range(18, 26)) + list(range(40, 45)) + list(range(49, 53))
    df['is_school_break'] = df['week_number'].isin(school_break_weeks).astype(int)
    
    return df


def add_hierarchical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add state-level aggregated features for hierarchical modeling."""
    df = df.copy()
    
    # State-level aggregates
    state_weekly = df.groupby(['state', 'year', 'week_number']).agg({
        'bio_update_child': 'sum',
        'enroll_child': 'sum',
        'demo_update_child': 'sum'
    }).reset_index()
    state_weekly.columns = ['state', 'year', 'week_number', 
                            'state_bio_update', 'state_enroll', 'state_demo_update']
    
    df = df.merge(state_weekly, on=['state', 'year', 'week_number'], how='left')
    
    # District share of state
    df['district_share_bio'] = df['bio_update_child'] / df['state_bio_update'].replace(0, 1)
    df['district_share_enroll'] = df['enroll_child'] / df['state_enroll'].replace(0, 1)
    
    return df


def add_extended_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended lag features for LightGBM."""
    df = df.copy()
    df = df.sort_values(['state', 'district', 'year', 'week_number']).reset_index(drop=True)
    
    grouped = df.groupby(['state', 'district'])
    
    # Extended lags
    for lag in [1, 2, 3, 4]:
        df[f'lag_{lag}w_bio'] = grouped['bio_update_child'].shift(lag)
        df[f'lag_{lag}w_demo'] = grouped['demo_update_child'].shift(lag)
    
    # Rolling means
    for window in [4, 8, 12]:
        df[f'rolling_{window}w_bio_mean'] = grouped['bio_update_child'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_{window}w_bio_std'] = grouped['bio_update_child'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Trend features
    df['bio_trend_4w'] = df['lag_1w_bio'] - df['lag_4w_bio']
    
    return df


def prepare_lgb_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for LightGBM training.
    
    Returns:
        (X, y, feature_names, label_encoders)
    """
    df = df.copy()
    
    # Add features
    df = add_calendar_features(df)
    df = add_hierarchical_features(df)
    df = add_extended_lag_features(df)
    
    # Encode categoricals
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    
    df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
    df['district_encoded'] = le_district.fit_transform(df['district'].astype(str))
    
    # Feature columns
    feature_cols = [
        'state_encoded', 'district_encoded',
        'week_number', 'month', 'quarter', 'week_sin', 'week_cos', 'is_school_break',
        'lag_1w_bio', 'lag_2w_bio', 'lag_3w_bio', 'lag_4w_bio',
        'lag_1w_demo', 'lag_2w_demo',
        'rolling_4w_bio_mean', 'rolling_8w_bio_mean', 'rolling_12w_bio_mean',
        'rolling_4w_bio_std',
        'bio_trend_4w',
        'state_bio_update', 'district_share_bio', 'district_share_enroll',
        'enroll_child', 'demo_update_child', 'saturation_proxy'
    ]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Target
    target = 'bio_update_child'
    
    # Remove rows with NaN in features or target
    df_clean = df.dropna(subset=feature_cols + [target])
    
    # Handle suppressed values (-1) in target
    df_clean = df_clean[df_clean[target] >= 0]
    
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    return X, y, feature_cols, {'state': le_state, 'district': le_district}, df_clean


def time_series_cv(X: pd.DataFrame, y: pd.Series, df_meta: pd.DataFrame, n_folds: int = 4):
    """
    Expanding window time-series cross-validation.
    
    Args:
        X: Features
        y: Target
        df_meta: Metadata (year, week) for splitting
        n_folds: Number of CV folds
    
    Returns:
        List of (train_idx, val_idx) tuples
    """
    # Get unique weeks sorted
    weeks = df_meta[['year', 'week_number']].drop_duplicates().sort_values(['year', 'week_number'])
    n_weeks = len(weeks)
    
    fold_size = n_weeks // (n_folds + 1)
    
    folds = []
    for i in range(n_folds):
        train_end_week = (i + 1) * fold_size
        val_end_week = min(train_end_week + fold_size, n_weeks)
        
        train_weeks = weeks.iloc[:train_end_week]
        val_weeks = weeks.iloc[train_end_week:val_end_week]
        
        train_idx = df_meta.merge(train_weeks, on=['year', 'week_number']).index.tolist()
        val_idx = df_meta.merge(val_weeks, on=['year', 'week_number']).index.tolist()
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    
    return folds


def train_lightgbm_model(
    input_path: str = 'data/processed/model_features.parquet',
    output_path: str = 'outputs/forecasts_lgb.csv'
) -> dict:
    """
    Train LightGBM model with time-series CV and generate forecasts.
    """
    print("=" * 60)
    print("ADVANCED FORECASTING (LightGBM)")
    print("=" * 60)
    
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available. Skipping.")
        return {}
    
    # Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    df = load_features(input_path)
    X, y, feature_cols, encoders, df_clean = prepare_lgb_data(df)
    print(f"Prepared {len(X):,} samples with {len(feature_cols)} features")
    
    # Time-series CV
    print("\n[2/5] Performing time-series cross-validation...")
    folds = time_series_cv(X, y, df_clean[['year', 'week_number']], n_folds=4)
    
    cv_scores = []
    models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Filter to valid indices
        train_idx = [i for i in train_idx if i in X.index]
        val_idx = [i for i in val_idx if i in X.index]
        
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred = np.clip(y_pred, 0, None)  # Non-negative predictions
        
        # Handle zero values in MAPE
        mask = y_val > 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_val[mask], y_pred[mask])
        else:
            mape = np.nan
        
        mae = mean_absolute_error(y_val, y_pred)
        cv_scores.append({'fold': fold_idx, 'mape': mape, 'mae': mae})
        models.append(model)
        
        print(f"  Fold {fold_idx + 1}: MAPE={mape:.4f}, MAE={mae:.2f}")
    
    # Final model on all data
    print("\n[3/5] Training final model on all data...")
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, train_data, num_boost_round=200)
    
    # Generate forecasts for latest week
    print("\n[4/5] Generating forecasts...")
    latest_week = df_clean.groupby(['state', 'district']).tail(1)
    X_latest = latest_week[feature_cols]
    
    predictions = final_model.predict(X_latest)
    predictions = np.clip(predictions, 0, None)
    
    # Create forecast dataframe
    forecast_df = latest_week[['state', 'district', 'year', 'week_number']].copy()
    forecast_df['yhat'] = predictions
    
    # Approximate uncertainty bands (using CV variance)
    cv_std = np.std([s['mae'] for s in cv_scores if not np.isnan(s['mae'])])
    forecast_df['yhat_lower'] = (forecast_df['yhat'] - 1.96 * cv_std).clip(lower=0)
    forecast_df['yhat_upper'] = forecast_df['yhat'] + 1.96 * cv_std
    
    # Add forecast week
    forecast_df['forecast_week'] = forecast_df['week_number'] + 1
    
    # Save
    print("\n[5/5] Saving results...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"Saved {len(forecast_df):,} forecasts to {output_path}")
    
    # Results
    avg_mape = np.nanmean([s['mape'] for s in cv_scores])
    avg_mae = np.nanmean([s['mae'] for s in cv_scores])
    
    results = {
        'model': final_model,
        'cv_scores': cv_scores,
        'avg_mape': avg_mape,
        'avg_mae': avg_mae,
        'feature_importance': dict(zip(feature_cols, final_model.feature_importance())),
        'forecasts': forecast_df
    }
    
    print(f"\nAverage CV MAPE: {avg_mape:.4f}")
    print(f"Average CV MAE: {avg_mae:.2f}")
    
    # Feature importance
    print("\nTop 10 features:")
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    print(fi.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("LIGHTGBM FORECASTING COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = train_lightgbm_model()
