"""
synthetic_pilot.py
Generate synthetic post-intervention data for pilot evaluation.

Creates treatment effect in treatment districts while control districts
follow the baseline trend.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_pilot_data(
    features_path: str = 'data/processed/model_features.parquet',
    regions_path: str = 'outputs/pilot_regions.csv',
    output_path: str = 'data/processed/pilot_synthetic.parquet',
    treatment_effect: float = 0.30,  # 30% increase in treatment
    intervention_start_week: int = 25,
    n_post_weeks: int = 12
) -> pd.DataFrame:
    """
    Generate synthetic post-intervention data.
    
    Args:
        features_path: Path to feature data
        regions_path: Path to pilot regions CSV
        output_path: Output path for synthetic data
        treatment_effect: Relative increase for treatment group
        intervention_start_week: Week number when intervention starts
        n_post_weeks: Number of weeks of post-intervention data
    
    Returns:
        DataFrame with synthetic pilot data
    """
    print("=" * 60)
    print("SYNTHETIC PILOT DATA GENERATOR")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    features = pd.read_parquet(features_path)
    regions = pd.read_csv(regions_path)
    
    treatment_districts = regions[regions['group'] == 'treatment']['district'].tolist()
    control_districts = regions[regions['group'] == 'control']['district'].tolist()
    
    print(f"  Treatment districts: {len(treatment_districts)}")
    print(f"  Control districts: {len(control_districts)}")
    
    # Filter to pilot districts
    pilot_features = features[
        features['district'].isin(treatment_districts + control_districts)
    ].copy()
    
    # Add group column
    pilot_features['group'] = pilot_features['district'].apply(
        lambda x: 'treatment' if x in treatment_districts else 'control'
    )
    
    # Generate synthetic post-intervention weeks
    print("\n[2/4] Generating post-intervention data...")
    
    synthetic_rows = []
    max_week = features['week_number'].max()
    
    for district in treatment_districts + control_districts:
        district_data = pilot_features[pilot_features['district'] == district]
        
        if district_data.empty:
            continue
            
        # Get baseline values (mean of last 4 weeks)
        baseline = district_data.nlargest(4, 'week_number')
        base_bio = baseline['bio_update_child'].mean()
        base_demo = baseline['demo_update_child'].mean() if 'demo_update_child' in baseline else 0
        base_backlog = baseline['update_backlog_child'].mean() if 'update_backlog_child' in baseline else 0
        state = district_data['state'].iloc[0]
        
        is_treatment = district in treatment_districts
        
        for week_offset in range(n_post_weeks):
            week = intervention_start_week + week_offset
            
            # Natural variation
            noise = np.random.normal(0, 0.1)
            
            if is_treatment:
                # Treatment effect: gradual ramp up over 4 weeks
                ramp = min(week_offset / 4, 1.0)
                effect = 1 + (treatment_effect * ramp) + noise
                backlog_effect = 1 - (0.5 * ramp)  # Backlog decreases
            else:
                # Control: small natural variation only
                effect = 1 + noise * 0.5
                backlog_effect = 1 + noise * 0.3
            
            synthetic_rows.append({
                'state': state,
                'district': district,
                'year': 2026,
                'week_number': week,
                'bio_update_child': max(0, base_bio * effect),
                'demo_update_child': max(0, base_demo * (1 + noise * 0.3)),
                'update_backlog_child': max(0, base_backlog * backlog_effect),
                'completion_rate_child': min(0.95, 0.7 * effect),
                'group': 'treatment' if is_treatment else 'control',
                'post_intervention': 1,
                'in_treatment_group': 1 if is_treatment else 0
            })
    
    synthetic_df = pd.DataFrame(synthetic_rows)
    
    # Add pre-intervention data with flags
    print("\n[3/4] Combining with pre-intervention data...")
    pilot_features['post_intervention'] = 0
    pilot_features['in_treatment_group'] = pilot_features['group'].apply(
        lambda x: 1 if x == 'treatment' else 0
    )
    
    # Combine
    combined = pd.concat([pilot_features, synthetic_df], ignore_index=True)
    
    # Save
    print("\n[4/4] Saving synthetic data...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    
    print(f"  Saved {len(combined)} rows to {output_path}")
    print(f"  Pre-intervention: {len(pilot_features)} rows")
    print(f"  Post-intervention: {len(synthetic_df)} rows")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    pre_treatment = pilot_features[pilot_features['group'] == 'treatment']['bio_update_child'].mean()
    post_treatment = synthetic_df[synthetic_df['group'] == 'treatment']['bio_update_child'].mean()
    pre_control = pilot_features[pilot_features['group'] == 'control']['bio_update_child'].mean()
    post_control = synthetic_df[synthetic_df['group'] == 'control']['bio_update_child'].mean()
    
    print(f"\nTreatment Group:")
    print(f"  Pre:  {pre_treatment:,.0f}")
    print(f"  Post: {post_treatment:,.0f}")
    print(f"  Change: {(post_treatment/pre_treatment - 1)*100:+.1f}%")
    
    print(f"\nControl Group:")
    print(f"  Pre:  {pre_control:,.0f}")
    print(f"  Post: {post_control:,.0f}")
    print(f"  Change: {(post_control/pre_control - 1)*100:+.1f}%")
    
    print(f"\nDifference-in-Differences:")
    did = (post_treatment - pre_treatment) - (post_control - pre_control)
    print(f"  DiD Estimate: {did:+,.0f} updates/week")
    
    return combined


if __name__ == '__main__':
    generate_pilot_data()
