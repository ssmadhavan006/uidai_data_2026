"""
action_recommender.py
Budget-constrained action recommendation engine.

Answers: "Given budget B, what's the optimal set of interventions across districts?"
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Import simulator
try:
    from simulator import (
        load_interventions, load_priority_data, 
        run_monte_carlo_simulation, simulate_intervention
    )
except ImportError:
    from src.simulator import (
        load_interventions, load_priority_data,
        run_monte_carlo_simulation, simulate_intervention
    )


def calculate_cost_effectiveness(
    district_data: Dict,
    intervention: Dict,
    intervention_name: str,
    priority_score: float
) -> Dict:
    """
    Calculate cost-effectiveness ratio for a district-intervention pair.
    
    Returns:
        Dict with intervention details and effectiveness metrics
    """
    # Run quick simulation (median scenario)
    result = simulate_intervention(
        district_data, intervention, intervention_name, scenario='median'
    )
    
    # Calculate priority-weighted impact
    weighted_impact = result.backlog_reduction * priority_score
    
    # Cost-effectiveness ratio
    cost = intervention.get('cost', 1)
    ratio = weighted_impact / cost
    
    return {
        'district': district_data.get('district'),
        'state': district_data.get('state'),
        'intervention': intervention_name,
        'cost': cost,
        'backlog_reduction': result.backlog_reduction,
        'reduction_pct': result.reduction_pct,
        'cost_per_update': result.cost_per_update,
        'priority_score': priority_score,
        'weighted_impact': weighted_impact,
        'cost_effectiveness_ratio': ratio
    }


def recommend_actions(
    priority_list: pd.DataFrame,
    interventions: Dict,
    budget: float,
    max_per_district: int = 1
) -> Tuple[List[Dict], Dict]:
    """
    Greedy budget-constrained optimization.
    
    Args:
        priority_list: DataFrame with priority scores
        interventions: Dict of intervention configs
        budget: Total available budget
        max_per_district: Max interventions per district
    
    Returns:
        (selected_actions, summary_stats)
    """
    print("\n[1/3] Generating candidate actions...")
    candidates = []
    
    for _, row in priority_list.iterrows():
        district_data = {
            'state': row['state'],
            'district': row['district'],
            'backlog': row.get('forecasted_demand_next_4w', 500),
            'weekly_demand': row.get('yhat', 100),
            'baseline_capacity': row.get('yhat', 100) * 0.7
        }
        
        for interv_name, interv_config in interventions.items():
            candidate = calculate_cost_effectiveness(
                district_data, interv_config, interv_name, row.get('priority_score', 0.5)
            )
            candidate['bottleneck_label'] = row.get('bottleneck_label', 'UNKNOWN')
            candidates.append(candidate)
    
    print(f"  Generated {len(candidates)} candidate actions")
    
    # Sort by cost-effectiveness ratio (descending)
    print("\n[2/3] Selecting optimal actions (greedy)...")
    candidates.sort(key=lambda x: x['cost_effectiveness_ratio'], reverse=True)
    
    selected = []
    remaining_budget = budget
    districts_selected = {}
    
    for candidate in candidates:
        district = candidate['district']
        cost = candidate['cost']
        
        # Check constraints
        if cost > remaining_budget:
            continue
        if districts_selected.get(district, 0) >= max_per_district:
            continue
        
        # Select this action
        selected.append(candidate)
        remaining_budget -= cost
        districts_selected[district] = districts_selected.get(district, 0) + 1
    
    # Calculate summary stats
    summary = {
        'total_budget': budget,
        'budget_used': budget - remaining_budget,
        'budget_remaining': remaining_budget,
        'actions_selected': len(selected),
        'districts_covered': len(districts_selected),
        'total_backlog_reduction': sum(a['backlog_reduction'] for a in selected),
        'total_weighted_impact': sum(a['weighted_impact'] for a in selected),
        'avg_cost_per_update': np.mean([a['cost_per_update'] for a in selected]) if selected else 0
    }
    
    print(f"  Selected {len(selected)} actions covering {len(districts_selected)} districts")
    print(f"  Budget used: ₹{summary['budget_used']:,.0f} / ₹{budget:,.0f}")
    
    return selected, summary


def generate_action_pack(
    action: Dict,
    intervention_config: Dict,
    output_dir: str = 'outputs/sim_results'
) -> str:
    """
    Generate detailed action pack JSON for a selected intervention.
    """
    # Run Monte Carlo for uncertainty
    district_data = {
        'state': action['state'],
        'district': action['district'],
        'backlog': action['backlog_reduction'] / 0.35,  # Approximate from reduction
        'weekly_demand': action.get('weekly_demand', 100),
        'baseline_capacity': action.get('weekly_demand', 100) * 0.7
    }
    
    mc_result = run_monte_carlo_simulation(
        district_data, intervention_config, action['intervention'], n_runs=500
    )
    
    # Create action pack
    action_pack = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'district': action['district'],
            'state': action['state'],
            'intervention': action['intervention']
        },
        'rationale': {
            'bottleneck_type': action.get('bottleneck_label', 'UNKNOWN'),
            'priority_score': action['priority_score'],
            'why_selected': f"Highest cost-effectiveness ratio ({action['cost_effectiveness_ratio']:.4f}) for this district's {action.get('bottleneck_label', 'identified')} bottleneck"
        },
        'intervention_details': {
            'name': action['intervention'],
            'description': intervention_config.get('description', ''),
            'targets': intervention_config.get('target', []),
            'duration_weeks': intervention_config.get('duration_weeks', 4)
        },
        'cost_breakdown': {
            'total_cost': action['cost'],
            'cost_per_update': action['cost_per_update'],
            'currency': 'INR'
        },
        'expected_impact': {
            'backlog_reduction': {
                'median': mc_result['backlog_reduction']['p50'],
                'range_90pct': [
                    mc_result['backlog_reduction']['p5'],
                    mc_result['backlog_reduction']['p95']
                ]
            },
            'reduction_percentage': {
                'median': mc_result['reduction_pct']['p50'],
                'range_90pct': [
                    mc_result['reduction_pct']['p5'],
                    mc_result['reduction_pct']['p95']
                ]
            }
        },
        'confidence': '90% based on Monte Carlo simulation with 500 runs'
    }
    
    # Save
    filename = f"action_pack_{action['district'].replace(' ', '_')}.json"
    filepath = Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(action_pack, f, indent=2)
    
    return str(filepath)


def run_recommendation_engine(
    budget: float = 1000000,
    n_districts: int = 50,
    output_path: str = 'outputs/sim_results/recommended_actions.json'
) -> Tuple[List[Dict], Dict]:
    """
    Run the full recommendation pipeline.
    """
    print("=" * 60)
    print("ACTION RECOMMENDATION ENGINE")
    print("=" * 60)
    print(f"\nBudget: ₹{budget:,.0f}")
    
    # Load data
    print("\nLoading data...")
    interventions = load_interventions()
    priority = load_priority_data()
    
    # Get top priority districts
    priority_subset = priority.nsmallest(n_districts, 'priority_rank')
    
    # Run optimization
    selected, summary = recommend_actions(priority_subset, interventions, budget)
    
    # Generate action packs for top 10
    print("\n[3/3] Generating action packs...")
    action_packs = []
    for action in selected[:10]:
        interv_config = interventions[action['intervention']]
        pack_path = generate_action_pack(action, interv_config)
        action_packs.append(pack_path)
        print(f"  Created: {pack_path}")
    
    # Save full recommendations
    output = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'budget': budget,
            'districts_analyzed': n_districts
        },
        'summary': summary,
        'selected_actions': selected,
        'action_pack_files': action_packs
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved recommendations to {output_path}")
    
    # Print summary
    print("\n" + "-" * 40)
    print("RECOMMENDATION SUMMARY")
    print("-" * 40)
    print(f"Budget: ₹{budget:,.0f}")
    print(f"Used: ₹{summary['budget_used']:,.0f} ({summary['budget_used']/budget*100:.1f}%)")
    print(f"Actions: {summary['actions_selected']}")
    print(f"Districts: {summary['districts_covered']}")
    print(f"Est. backlog reduction: {summary['total_backlog_reduction']:,.0f} updates")
    
    print("\nTop 5 Recommended Actions:")
    for i, action in enumerate(selected[:5]):
        print(f"  {i+1}. {action['district']} ({action['state']})")
        print(f"     → {action['intervention']} @ ₹{action['cost']:,.0f}")
        print(f"     → Est. reduction: {action['backlog_reduction']:.0f} ({action['reduction_pct']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION COMPLETE")
    print("=" * 60)
    
    return selected, summary


if __name__ == '__main__':
    selected, summary = run_recommendation_engine(budget=1000000)
