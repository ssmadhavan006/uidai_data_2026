"""
04_bottleneck_fusion.py
Bottleneck Fusion Engine - combines forecasts, anomalies, and rules into actionable diagnoses.

Outputs:
- priority_scores.csv: District priority rankings
- bottleneck_labels.csv: Diagnostic labels with rationale
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_data() -> tuple:
    """Load all required data sources."""
    features = pd.read_parquet('data/processed/model_features.parquet')
    
    # Try to load forecasts and anomalies
    try:
        forecasts = pd.read_csv('outputs/forecasts.csv')
    except FileNotFoundError:
        forecasts = pd.DataFrame()
    
    try:
        anomalies = pd.read_csv('outputs/anomalies.csv')
    except FileNotFoundError:
        anomalies = pd.DataFrame()
    
    return features, forecasts, anomalies


# =============================================================================
# DIAGNOSTIC RULES
# =============================================================================

def flag_operational_bottleneck(row: pd.Series) -> tuple:
    """
    Detect operational/hardware/process bottlenecks.
    
    Triggers:
    - High failure rate (if available) or low completion rate
    - High repeat attempts (bio >> demo suggests retries)
    - Steady demographic updates but low biometric completion
    """
    rationale = []
    
    # Completion rate check
    completion_rate = row.get('completion_rate_child', 1)
    if pd.notna(completion_rate) and completion_rate < 0.7:
        rationale.append(f"Low completion rate ({completion_rate:.2f})")
    
    # Backlog check
    backlog = row.get('update_backlog_child', 0)
    if backlog > 0 and backlog > row.get('bio_update_child', 0) * 0.5:
        rationale.append(f"Significant backlog ({backlog:.0f})")
    
    # Ratio of bio to demo (low means bio is falling behind)
    if row.get('demo_update_child', 0) > 0:
        bio_demo_ratio = row.get('bio_update_child', 0) / row['demo_update_child']
        if bio_demo_ratio < 0.6:
            rationale.append(f"Bio/Demo ratio low ({bio_demo_ratio:.2f})")
    
    if len(rationale) >= 2:
        return True, "OPERATIONAL_BOTTLENECK", "; ".join(rationale)
    return False, None, None


def flag_demographic_surge(row: pd.Series, percentile_threshold: float = 0.9) -> tuple:
    """
    Detect demographic surge - high demand with normal success rate.
    
    Triggers:
    - Forecasted/actual demand above 90th percentile
    - Success rate within historical average
    """
    rationale = []
    
    # High demand check (using update_backlog as proxy for unmet demand)
    if row.get('bio_update_child', 0) > row.get('rolling_4w_mean_bio_update_child', 0) * 1.5:
        rationale.append("Bio updates 50%+ above 4-week average")
    
    if row.get('demo_update_child', 0) > row.get('rolling_4w_mean_demo_update_child', 0) * 1.5:
        rationale.append("Demo updates 50%+ above 4-week average")
    
    # Normal success (completion near historical)
    completion = row.get('completion_rate_child', 0)
    if pd.notna(completion) and completion > 0.8:
        rationale.append(f"Completion rate healthy ({completion:.2f})")
    
    if len(rationale) >= 2:
        return True, "DEMOGRAPHIC_SURGE", "; ".join(rationale)
    return False, None, None


def flag_capacity_strain(row: pd.Series) -> tuple:
    """
    Detect capacity strain - volume too high for infrastructure.
    
    Triggers:
    - High enrollment saturation
    - Declining completion rate trend
    """
    rationale = []
    
    saturation = row.get('saturation_proxy', 0)
    if pd.notna(saturation) and saturation > 0.5:
        rationale.append(f"High child saturation ({saturation:.2f})")
    
    # Week-over-week decline in bio updates
    wow_change = row.get('wow_change_bio_update_child', 0)
    if pd.notna(wow_change) and wow_change < -row.get('bio_update_child', 1) * 0.2:
        rationale.append("Bio updates declining week-over-week")
    
    if len(rationale) >= 2:
        return True, "CAPACITY_STRAIN", "; ".join(rationale)
    return False, None, None


def flag_anomaly_driven(row: pd.Series) -> tuple:
    """
    Flag districts with detected anomalies.
    """
    is_anomaly = row.get('is_anomaly', 0) == 1 or row.get('combined_anomaly', 0) == 1
    
    if is_anomaly:
        anomaly_score = row.get('anomaly_score', 0)
        return True, "ANOMALY_DETECTED", f"Isolation Forest score: {anomaly_score:.3f}"
    return False, None, None


def calculate_priority_score(
    row: pd.Series,
    forecasted_demand: float = None,
    w_demand: float = 0.3,
    w_incompletion: float = 0.3,
    w_backlog: float = 0.25,
    w_inequity: float = 0.15
) -> float:
    """
    Calculate composite priority score.
    
    Formula:
    Priority = w_demand * norm_demand + 
               w_incompletion * (1 - completion_rate) +
               w_backlog * norm_backlog +
               w_inequity * inequity_multiplier
    """
    # Demand component
    demand = forecasted_demand if forecasted_demand else row.get('bio_update_child', 0)
    norm_demand = min(demand / 1000, 1)  # Normalize to 0-1
    
    # Incompletion component
    completion = row.get('completion_rate_child', 1)
    if pd.isna(completion):
        completion = 1
    incompletion = 1 - min(max(completion, 0), 1)
    
    # Backlog component
    backlog = max(row.get('update_backlog_child', 0), 0)
    norm_backlog = min(backlog / 500, 1)
    
    # Inequity multiplier (based on saturation - lower saturation = potential underserved area)
    saturation = row.get('saturation_proxy', 0.5)
    if pd.isna(saturation):
        saturation = 0.5
    # Lower saturation could mean underserved, so inverse
    inequity = 1 - min(max(saturation, 0), 1)
    
    score = (
        w_demand * norm_demand +
        w_incompletion * incompletion +
        w_backlog * norm_backlog +
        w_inequity * inequity
    )
    
    return score


def apply_all_rules(row: pd.Series) -> tuple:
    """
    Apply all diagnostic rules to a row.
    
    Returns:
        (bottleneck_label, rationale) or (None, None)
    """
    # Check rules in priority order
    rules = [
        flag_operational_bottleneck,
        flag_demographic_surge,
        flag_capacity_strain,
        flag_anomaly_driven
    ]
    
    for rule_fn in rules:
        is_triggered, label, rationale = rule_fn(row)
        if is_triggered:
            return label, rationale
    
    return "NORMAL", "No bottleneck indicators detected"


def run_bottleneck_fusion(
    priority_output: str = 'outputs/priority_scores.csv',
    labels_output: str = 'outputs/bottleneck_labels.csv'
) -> tuple:
    """
    Run the complete bottleneck fusion engine.
    """
    print("=" * 60)
    print("BOTTLENECK FUSION ENGINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data sources...")
    features, forecasts, anomalies = load_data()
    print(f"  Features: {len(features):,} rows")
    print(f"  Forecasts: {len(forecasts):,} rows")
    print(f"  Anomalies: {len(anomalies):,} rows")
    
    # Get latest week per district
    print("\n[2/4] Preparing latest data...")
    latest = features.sort_values(['state', 'district', 'year', 'week_number']).groupby(
        ['state', 'district']
    ).tail(1).copy()
    print(f"  Latest week data: {len(latest):,} districts")
    
    # Merge anomalies if available
    if not anomalies.empty:
        anomaly_cols = ['state', 'district', 'is_anomaly', 'anomaly_score', 'combined_anomaly']
        anomaly_cols = [c for c in anomaly_cols if c in anomalies.columns]
        latest = latest.merge(
            anomalies[anomaly_cols].drop_duplicates(['state', 'district']),
            on=['state', 'district'],
            how='left',
            suffixes=('', '_anom')
        )
    
    # Merge forecasts if available
    if not forecasts.empty:
        forecast_cols = ['state', 'district', 'yhat']
        forecast_cols = [c for c in forecast_cols if c in forecasts.columns]
        if forecast_cols:
            latest = latest.merge(
                forecasts[forecast_cols].drop_duplicates(['state', 'district']),
                on=['state', 'district'],
                how='left'
            )
            latest['forecasted_demand'] = latest.get('yhat', latest['bio_update_child'])
    else:
        latest['forecasted_demand'] = latest['bio_update_child']
    
    # Apply rules and calculate priority
    print("\n[3/4] Applying bottleneck rules and calculating priority...")
    
    results = []
    for idx, row in latest.iterrows():
        # Calculate priority score
        priority = calculate_priority_score(row, row.get('forecasted_demand', None))
        
        # Apply diagnostic rules
        label, rationale = apply_all_rules(row)
        
        results.append({
            'state': row['state'],
            'district': row['district'],
            'year': row['year'],
            'week_number': row['week_number'],
            'priority_score': priority,
            'bottleneck_label': label,
            'rationale': rationale,
            'forecasted_demand_next_4w': row.get('forecasted_demand', row.get('bio_update_child', 0)),
            'current_bio_updates': row.get('bio_update_child', 0),
            'current_demo_updates': row.get('demo_update_child', 0),
            'update_backlog': row.get('update_backlog_child', 0),
            'completion_rate': row.get('completion_rate_child', np.nan)
        })
    
    results_df = pd.DataFrame(results)
    
    # Rank by priority
    results_df['priority_rank'] = results_df['priority_score'].rank(ascending=False, method='dense')
    results_df = results_df.sort_values('priority_rank')
    
    # Save outputs
    print("\n[4/4] Saving outputs...")
    Path(priority_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Priority scores
    priority_cols = ['state', 'district', 'priority_score', 'priority_rank', 
                     'bottleneck_label', 'forecasted_demand_next_4w']
    results_df[priority_cols].to_csv(priority_output, index=False)
    print(f"  Saved {len(results_df):,} priority scores to {priority_output}")
    
    # Bottleneck labels with rationale
    labels_cols = ['state', 'district', 'year', 'week_number', 
                   'bottleneck_label', 'rationale', 
                   'current_bio_updates', 'current_demo_updates', 
                   'update_backlog', 'completion_rate']
    results_df[labels_cols].to_csv(labels_output, index=False)
    print(f"  Saved {len(results_df):,} bottleneck labels to {labels_output}")
    
    # Summary
    print("\n" + "-" * 40)
    print("BOTTLENECK SUMMARY")
    print("-" * 40)
    label_counts = results_df['bottleneck_label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print(f"\nTop 10 priority districts:")
    print(results_df[['state', 'district', 'priority_score', 'bottleneck_label']].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("BOTTLENECK FUSION COMPLETE")
    print("=" * 60)
    
    return results_df, label_counts


if __name__ == '__main__':
    results, summary = run_bottleneck_fusion()
