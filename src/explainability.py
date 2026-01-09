"""
explainability.py
SHAP-based explainability for LightGBM forecasts.

Integrates top feature contributions into bottleneck rationale.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def compute_shap_values(model, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compute SHAP values for predictions.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        top_n: Number of top features to return per prediction
    
    Returns:
        DataFrame with top contributing features per row
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available")
        return pd.DataFrame()
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Build explanations for each row
    explanations = []
    for i in range(len(X)):
        row_shap = shap_values[i]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
        
        top_features = []
        for idx in sorted_idx:
            feat_name = feature_names[idx]
            shap_val = row_shap[idx]
            feat_val = X.iloc[i, idx]
            direction = "↑" if shap_val > 0 else "↓"
            top_features.append(f"{feat_name}={feat_val:.1f} ({direction}{abs(shap_val):.1f})")
        
        explanations.append({
            'shap_explanation': "; ".join(top_features),
            'top_feature': feature_names[sorted_idx[0]],
            'top_shap_value': row_shap[sorted_idx[0]]
        })
    
    return pd.DataFrame(explanations)


def add_shap_to_bottleneck_labels(
    model_path: str = None,
    labels_path: str = 'outputs/bottleneck_labels.csv',
    features_path: str = 'data/processed/model_features.parquet',
    output_path: str = 'outputs/bottleneck_labels_explained.csv'
) -> pd.DataFrame:
    """
    Add SHAP explanations to bottleneck labels.
    """
    print("=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)
    
    if not SHAP_AVAILABLE or not LIGHTGBM_AVAILABLE:
        print("SHAP or LightGBM not available. Creating placeholder explanations.")
        
        # Load labels and add placeholder
        labels = pd.read_csv(labels_path)
        labels['shap_explanation'] = "Feature importance analysis pending"
        labels['top_feature'] = "N/A"
        labels.to_csv(output_path, index=False)
        return labels
    
    # Load data
    print("\n[1/3] Loading data...")
    labels = pd.read_csv(labels_path)
    features = pd.read_parquet(features_path)
    
    # Get latest per district
    latest = features.sort_values(['state', 'district', 'year', 'week_number']).groupby(
        ['state', 'district']
    ).tail(1)
    
    # Prepare features for SHAP
    feature_cols = [
        'week_number', 'bio_update_child', 'demo_update_child', 'enroll_child',
        'update_backlog_child', 'completion_rate_child', 'saturation_proxy',
        'lag_1w_bio_update_child', 'rolling_4w_mean_bio_update_child'
    ]
    feature_cols = [c for c in feature_cols if c in latest.columns]
    
    X = latest[feature_cols].fillna(0)
    
    # Train a quick model for explanations
    print("\n[2/3] Training explanation model...")
    y = latest['bio_update_child'].fillna(0)
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'verbose': -1
    }
    model = lgb.train(params, train_data, num_boost_round=50)
    
    # Compute SHAP values
    print("\n[3/3] Computing SHAP explanations...")
    shap_df = compute_shap_values(model, X, top_n=3)
    
    # Merge with labels
    latest_reset = latest[['state', 'district']].reset_index(drop=True)
    shap_df = pd.concat([latest_reset, shap_df], axis=1)
    
    labels = labels.merge(
        shap_df[['state', 'district', 'shap_explanation', 'top_feature']],
        on=['state', 'district'],
        how='left'
    )
    
    # Update rationale with SHAP
    labels['rationale_with_shap'] = labels.apply(
        lambda r: f"{r['rationale']} | Key factors: {r.get('shap_explanation', 'N/A')}", 
        axis=1
    )
    
    # Save
    labels.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    print("\n" + "=" * 60)
    print("EXPLAINABILITY COMPLETE")
    print("=" * 60)
    
    return labels


if __name__ == '__main__':
    labels = add_shap_to_bottleneck_labels()
    print(labels[['state', 'district', 'bottleneck_label', 'shap_explanation']].head())
