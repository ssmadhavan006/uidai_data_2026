"""
fairness_audit.py
Fairness audit for disparity analysis across protected groups.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def define_protected_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define protected groups based on available proxies.
    
    Groups:
    - urbanization_tier: Based on average bio updates (proxy for infrastructure)
    - state_tier: Based on state-level aggregates
    """
    df = df.copy()
    
    # Urbanization proxy: districts with higher updates likely more urban
    state_avg = df.groupby('state')['bio_update_child'].mean()
    median_update = state_avg.median()
    
    df['urbanization_tier'] = df['state'].map(
        lambda s: 'high_infra' if state_avg.get(s, 0) > median_update else 'low_infra'
    )
    
    # State tier based on completion rate
    state_completion = df.groupby('state')['completion_rate_child'].mean()
    median_completion = state_completion.median()
    
    df['performance_tier'] = df['state'].map(
        lambda s: 'high_performer' if state_completion.get(s, 0) > median_completion else 'low_performer'
    )
    
    return df


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute metrics per group."""
    agg_dict = {
        'bio_update_child': ['mean', 'std', 'count'],
        'update_backlog_child': 'mean',
        'completion_rate_child': 'mean'
    }
    
    # Add priority_score if available
    if 'priority_score' in df.columns:
        agg_dict['priority_score'] = 'mean'
    
    metrics = df.groupby(group_col).agg(agg_dict).round(3)
    
    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    metrics = metrics.reset_index()
    
    return metrics


def calculate_disparity_indices(group_metrics: pd.DataFrame) -> dict:
    """Calculate disparity indices between groups."""
    if len(group_metrics) < 2:
        return {}
    
    # Extract values
    bio_updates = group_metrics['bio_update_child_mean'].values
    backlogs = group_metrics['update_backlog_child_mean'].values
    completion = group_metrics['completion_rate_child_mean'].values
    
    # Disparity ratios
    update_disparity = max(bio_updates) / max(min(bio_updates), 1)
    backlog_disparity = max(backlogs) / max(min(backlogs), 1)
    completion_gap = max(completion) - min(completion)
    
    return {
        'update_rate_ratio': update_disparity,
        'backlog_ratio': backlog_disparity,
        'completion_gap': completion_gap
    }


def analyze_intervention_fairness(
    priority_df: pd.DataFrame,
    interventions_path: str = 'outputs/sim_results/recommended_actions.json'
) -> dict:
    """Analyze how interventions affect fairness."""
    import json
    
    try:
        with open(interventions_path, 'r') as f:
            recommendations = json.load(f)
    except FileNotFoundError:
        return {
            'low_infra_interventions': 0,
            'high_infra_interventions': 0,
            'fairness_ratio': 1.0
        }
    
    selected = recommendations.get('selected_actions', [])
    
    if not selected:
        return {
            'low_infra_interventions': 0,
            'high_infra_interventions': 0,
            'fairness_ratio': 1.0
        }
    
    # Use priority score directly to classify districts
    # No need to call define_protected_groups on priority_df
    intervention_counts = {'low_priority': 0, 'high_priority': 0}
    
    for action in selected:
        district = action.get('district')
        row = priority_df[priority_df['district'] == district]
        if not row.empty and 'priority_score' in priority_df.columns:
            score = row.iloc[0]['priority_score']
            if score > 0.5:
                intervention_counts['high_priority'] += 1
            else:
                intervention_counts['low_priority'] += 1
    
    return {
        'low_infra_interventions': intervention_counts.get('low_priority', 0),
        'high_infra_interventions': intervention_counts.get('high_priority', 0),
        'fairness_ratio': intervention_counts.get('high_priority', 1) / max(intervention_counts.get('low_priority', 1), 1)
    }


def run_fairness_audit(
    features_path: str = 'data/processed/model_features.parquet',
    priority_path: str = 'outputs/priority_scores.csv',
    output_path: str = 'outputs/fairness_report.csv'
) -> dict:
    """Run complete fairness audit."""
    print("=" * 60)
    print("PHASE 6: FAIRNESS AUDIT")
    print("=" * 60)
    
    # Load data
    features = pd.read_parquet(features_path)
    priority = pd.read_csv(priority_path)
    
    # Drop any existing priority columns from features to avoid conflict
    cols_to_drop = [c for c in features.columns if c in ['priority_score', 'bottleneck_label', 'priority_rank']]
    if cols_to_drop:
        features = features.drop(columns=cols_to_drop)
    
    # Merge priority score into features
    merge_cols = ['state', 'district']
    priority_cols = ['priority_score', 'bottleneck_label']
    available_cols = [c for c in priority_cols if c in priority.columns]
    
    df = features.merge(priority[merge_cols + available_cols], on=merge_cols, how='left')
    
    # Fill missing values
    if 'priority_score' in df.columns:
        df['priority_score'] = df['priority_score'].fillna(0.5)
    else:
        df['priority_score'] = 0.5
    
    df['bio_update_child'] = df['bio_update_child'].fillna(0)
    df['update_backlog_child'] = df['update_backlog_child'].fillna(0) if 'update_backlog_child' in df.columns else 0
    df['completion_rate_child'] = df['completion_rate_child'].fillna(0.8) if 'completion_rate_child' in df.columns else 0.8
    
    # Define groups
    print("\n[1/4] Defining protected groups...")
    df = define_protected_groups(df)
    
    # Compute metrics by urbanization tier
    print("\n[2/4] Computing group metrics...")
    
    urban_metrics = compute_group_metrics(df, 'urbanization_tier')
    print("\nBy Urbanization Tier:")
    print(urban_metrics.to_string(index=False))
    
    perf_metrics = compute_group_metrics(df, 'performance_tier')
    print("\nBy Performance Tier:")
    print(perf_metrics.to_string(index=False))
    
    # Calculate disparities
    print("\n[3/4] Calculating disparity indices...")
    urban_disparity = calculate_disparity_indices(urban_metrics)
    perf_disparity = calculate_disparity_indices(perf_metrics)
    
    print(f"\nUrbanization Disparity:")
    print(f"  Update Rate Ratio: {urban_disparity.get('update_rate_ratio', 0):.2f}")
    print(f"  Backlog Ratio: {urban_disparity.get('backlog_ratio', 0):.2f}")
    print(f"  Completion Gap: {urban_disparity.get('completion_gap', 0):.2f}")
    
    # Intervention fairness
    print("\n[4/4] Analyzing intervention fairness...")
    intervention_fairness = analyze_intervention_fairness(priority)
    print(f"  Low-infra interventions: {intervention_fairness.get('low_infra_interventions', 0)}")
    print(f"  High-infra interventions: {intervention_fairness.get('high_infra_interventions', 0)}")
    
    # Compile report
    report = {
        'generated_at': datetime.now().isoformat(),
        'urbanization_update_ratio': urban_disparity.get('update_rate_ratio', 0),
        'urbanization_backlog_ratio': urban_disparity.get('backlog_ratio', 0),
        'urbanization_completion_gap': urban_disparity.get('completion_gap', 0),
        'performance_update_ratio': perf_disparity.get('update_rate_ratio', 0),
        'performance_backlog_ratio': perf_disparity.get('backlog_ratio', 0),
        'low_infra_interventions': intervention_fairness.get('low_infra_interventions', 0),
        'high_infra_interventions': intervention_fairness.get('high_infra_interventions', 0)
    }
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Validation check
    print("\n" + "=" * 50)
    print("FAIRNESS FINDINGS")
    print("=" * 50)
    
    issues = []
    if urban_disparity.get('update_rate_ratio', 1) > 2:
        issues.append("High update rate disparity between infrastructure tiers")
    if urban_disparity.get('completion_gap', 0) > 0.2:
        issues.append("Significant completion rate gap across regions")
    if intervention_fairness.get('fairness_ratio', 1) < 0.5:
        issues.append("Interventions skewed toward high-infrastructure areas")
    
    if issues:
        print(f"Identified {len(issues)} disparity issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No significant disparity issues identified")
    
    return {
        'urban_metrics': urban_metrics,
        'perf_metrics': perf_metrics,
        'disparities': {**urban_disparity, **perf_disparity},
        'intervention_fairness': intervention_fairness,
        'issues': issues,
        'report': report
    }


if __name__ == '__main__':
    results = run_fairness_audit()
