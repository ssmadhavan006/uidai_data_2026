"""
validate_phase3.py
Validation script for Phase 3: MAPE comparison and anomaly precision.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


def compare_mape_prophet_vs_lgb(
    features_path: str = 'data/processed/model_features.parquet',
    prophet_forecasts_path: str = 'outputs/forecasts.csv',
    lgb_forecasts_path: str = 'outputs/forecasts_lgb.csv'
) -> dict:
    """
    Compare MAPE between Prophet baseline and LightGBM.
    """
    print("=" * 60)
    print("MAPE COMPARISON: Prophet vs LightGBM")
    print("=" * 60)
    
    # Load features for actuals
    features = pd.read_parquet(features_path)
    latest = features.sort_values(['state', 'district', 'year', 'week_number']).groupby(
        ['state', 'district']
    ).tail(1)
    
    actuals = latest[['state', 'district', 'bio_update_child']].copy()
    actuals = actuals.rename(columns={'bio_update_child': 'actual'})
    
    results = {}
    
    # Prophet MAPE
    try:
        prophet = pd.read_csv(prophet_forecasts_path)
        # Get first forecast per district as prediction
        prophet_pred = prophet.drop_duplicates(['state', 'district'], keep='first')
        prophet_merged = actuals.merge(prophet_pred[['state', 'district', 'yhat']], on=['state', 'district'])
        
        # Calculate MAPE (avoid division by zero)
        mask = prophet_merged['actual'] > 0
        if mask.sum() > 0:
            prophet_mape = np.mean(np.abs(
                (prophet_merged.loc[mask, 'actual'] - prophet_merged.loc[mask, 'yhat']) / 
                prophet_merged.loc[mask, 'actual']
            ))
            results['prophet_mape'] = prophet_mape
            print(f"\nProphet MAPE: {prophet_mape:.4f} ({prophet_mape*100:.2f}%)")
        else:
            results['prophet_mape'] = np.nan
            print("\nProphet MAPE: N/A (no valid actuals)")
            
    except FileNotFoundError:
        print("\nProphet forecasts not found.")
        results['prophet_mape'] = np.nan
    
    # LightGBM MAPE
    try:
        lgb = pd.read_csv(lgb_forecasts_path)
        lgb_merged = actuals.merge(lgb[['state', 'district', 'yhat']], on=['state', 'district'])
        
        mask = lgb_merged['actual'] > 0
        if mask.sum() > 0:
            lgb_mape = np.mean(np.abs(
                (lgb_merged.loc[mask, 'actual'] - lgb_merged.loc[mask, 'yhat']) / 
                lgb_merged.loc[mask, 'actual']
            ))
            results['lgb_mape'] = lgb_mape
            print(f"LightGBM MAPE: {lgb_mape:.4f} ({lgb_mape*100:.2f}%)")
        else:
            results['lgb_mape'] = np.nan
            print("LightGBM MAPE: N/A (no valid actuals)")
            
    except FileNotFoundError:
        print("LightGBM forecasts not found.")
        results['lgb_mape'] = np.nan
    
    # Compare
    if not np.isnan(results.get('prophet_mape', np.nan)) and not np.isnan(results.get('lgb_mape', np.nan)):
        improvement = (results['prophet_mape'] - results['lgb_mape']) / results['prophet_mape'] * 100
        results['improvement_pct'] = improvement
        print(f"\nLightGBM improvement: {improvement:.1f}%")
        
        if improvement >= 10:
            print("✅ PASS: LightGBM improves MAPE by 10%+")
        else:
            print("⚠️ PARTIAL: LightGBM improvement < 10%")
    
    return results


def validate_anomaly_precision(
    anomalies_path: str = 'outputs/anomalies.csv',
    inject_synthetic: bool = True
) -> dict:
    """
    Validate anomaly detection precision using synthetic test data.
    """
    print("\n" + "=" * 60)
    print("ANOMALY PRECISION VALIDATION")
    print("=" * 60)
    
    try:
        anomalies = pd.read_csv(anomalies_path)
    except FileNotFoundError:
        print("Anomalies file not found.")
        return {'precision': np.nan}
    
    # Create synthetic ground truth based on extreme values
    print("\n[1/2] Creating synthetic ground truth...")
    
    # Mark as "true anomaly" if metric is in top/bottom 5%
    if 'bio_update_child' in anomalies.columns:
        q95 = anomalies['bio_update_child'].quantile(0.95)
        q05 = anomalies['bio_update_child'].quantile(0.05)
        anomalies['synthetic_anomaly'] = (
            (anomalies['bio_update_child'] > q95) | 
            (anomalies['bio_update_child'] < q05)
        ).astype(int)
    elif 'update_backlog_child' in anomalies.columns:
        q95 = anomalies['update_backlog_child'].quantile(0.95)
        anomalies['synthetic_anomaly'] = (anomalies['update_backlog_child'] > q95).astype(int)
    else:
        print("No suitable metric for synthetic ground truth.")
        return {'precision': np.nan}
    
    # Compare detected vs synthetic
    print("\n[2/2] Calculating precision metrics...")
    
    if 'combined_anomaly' in anomalies.columns:
        detected = anomalies['combined_anomaly']
    elif 'is_anomaly' in anomalies.columns:
        detected = anomalies['is_anomaly']
    else:
        print("No anomaly detection column found.")
        return {'precision': np.nan}
    
    ground_truth = anomalies['synthetic_anomaly']
    
    # Calculate metrics
    precision = precision_score(ground_truth, detected, zero_division=0)
    recall = recall_score(ground_truth, detected, zero_division=0)
    f1 = f1_score(ground_truth, detected, zero_division=0)
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detected_count': detected.sum(),
        'synthetic_count': ground_truth.sum()
    }
    
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Detected anomalies: {detected.sum()}")
    print(f"Synthetic anomalies: {ground_truth.sum()}")
    
    if precision >= 0.8:
        print("\n✅ PASS: Precision ≥ 0.8")
    else:
        print(f"\n⚠️ NOTE: Precision {precision:.2f} < 0.8 (based on synthetic data)")
        print("   Real-world validation may differ")
    
    return results


def manual_review_sample(
    labels_path: str = 'outputs/bottleneck_labels.csv',
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Print sample bottleneck labels for manual review.
    """
    print("\n" + "=" * 60)
    print("MANUAL REVIEW: Sample Bottleneck Labels")
    print("=" * 60)
    
    try:
        labels = pd.read_csv(labels_path)
    except FileNotFoundError:
        print("Labels file not found.")
        return pd.DataFrame()
    
    # Sample from each label type
    print(f"\nTotal districts: {len(labels)}")
    print(f"\nLabel distribution:")
    print(labels['bottleneck_label'].value_counts())
    
    print(f"\n--- Sample of {n_samples} high-priority districts ---\n")
    sample = labels.head(n_samples)
    
    for _, row in sample.iterrows():
        print(f"District: {row['district']} ({row['state']})")
        print(f"  Label: {row['bottleneck_label']}")
        print(f"  Rationale: {row['rationale']}")
        print(f"  Bio Updates: {row.get('current_bio_updates', 'N/A')}")
        print(f"  Backlog: {row.get('update_backlog', 'N/A')}")
        print()
    
    return sample


def run_all_validations():
    """Run all Phase 3 validations."""
    print("=" * 60)
    print("PHASE 3 VALIDATION SUITE")
    print("=" * 60)
    
    results = {}
    
    # MAPE comparison
    mape_results = compare_mape_prophet_vs_lgb()
    results['mape'] = mape_results
    
    # Anomaly precision
    anomaly_results = validate_anomaly_precision()
    results['anomaly'] = anomaly_results
    
    # Manual review
    sample = manual_review_sample()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print("\n✅ Completed checks:")
    print(f"  - MAPE Prophet: {mape_results.get('prophet_mape', 'N/A')}")
    print(f"  - MAPE LightGBM: {mape_results.get('lgb_mape', 'N/A')}")
    print(f"  - Improvement: {mape_results.get('improvement_pct', 'N/A')}%")
    print(f"  - Anomaly Precision: {anomaly_results.get('precision', 'N/A')}")
    print(f"  - Anomaly F1: {anomaly_results.get('f1_score', 'N/A')}")
    
    return results


if __name__ == '__main__':
    results = run_all_validations()
