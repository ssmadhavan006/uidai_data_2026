"""
simulator.py
Policy intervention simulator with Monte Carlo uncertainty propagation.

Answers: "What if we deploy intervention X in district Y?"
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Result of a single intervention simulation."""
    district: str
    state: str
    intervention: str
    initial_backlog: float
    final_backlog: float
    backlog_reduction: float
    reduction_pct: float
    total_cost: float
    cost_per_update: float
    weeks_simulated: int
    scenario: str
    fairness_index: float  # 0-1 where 1 = most equitable outcome


def load_interventions(path: str = 'config/interventions.json') -> Dict:
    """Load intervention definitions from config."""
    with open(path, 'r') as f:
        return json.load(f)


def load_priority_data(
    priority_path: str = 'outputs/priority_scores.csv',
    forecast_path: str = 'outputs/forecasts.csv'
) -> pd.DataFrame:
    """Load priority scores and forecasts."""
    priority = pd.read_csv(priority_path)
    
    try:
        forecasts = pd.read_csv(forecast_path)
        # Get average forecast per district
        forecast_agg = forecasts.groupby(['state', 'district']).agg({
            'yhat': 'mean',
            'yhat_lower': 'mean',
            'yhat_upper': 'mean'
        }).reset_index()
        
        priority = priority.merge(forecast_agg, on=['state', 'district'], how='left')
    except FileNotFoundError:
        priority['yhat'] = priority['forecasted_demand_next_4w']
        priority['yhat_lower'] = priority['yhat'] * 0.8
        priority['yhat_upper'] = priority['yhat'] * 1.2
    
    return priority


def simulate_intervention(
    district_data: Dict,
    intervention: Dict,
    intervention_name: str,
    scenario: str = 'median',
    demand_override: Optional[float] = None
) -> SimulationResult:
    """
    Simulate intervention impact on a district's backlog.
    
    Args:
        district_data: Dict with 'state', 'district', 'backlog', 'weekly_demand', 'baseline_capacity'
        intervention: Dict with intervention parameters
        intervention_name: Name of the intervention
        scenario: 'conservative', 'median', or 'optimistic'
        demand_override: Override weekly demand (for Monte Carlo)
    
    Returns:
        SimulationResult with projected outcomes
    """
    # Extract district info
    state = district_data.get('state', 'Unknown')
    district = district_data.get('district', 'Unknown')
    initial_backlog = max(district_data.get('backlog', 0), 0)
    weekly_demand = demand_override or district_data.get('weekly_demand', 100)
    baseline_capacity = district_data.get('baseline_capacity', weekly_demand * 0.8)
    
    # Intervention parameters
    duration = intervention.get('duration_weeks', 4)
    capacity_boost = intervention.get('capacity_per_week', 0)
    effectiveness = intervention.get('effectiveness', {}).get(scenario, 0.3)
    cost = intervention.get('cost', 0)
    
    # Calculate effective capacity boost
    effective_boost = capacity_boost * effectiveness
    
    # Simulate week-by-week
    backlog = initial_backlog
    total_processed = 0
    
    for week in range(duration):
        # New demand adds to backlog
        backlog += weekly_demand
        
        # Total processing capacity
        total_capacity = baseline_capacity + effective_boost
        
        # Process updates (can't exceed backlog)
        processed = min(total_capacity, backlog)
        backlog -= processed
        total_processed += processed
        
        # Ensure non-negative
        backlog = max(backlog, 0)
    
    # Calculate metrics
    backlog_reduction = initial_backlog - backlog + (weekly_demand * duration)  # Demand served
    reduction_pct = ((initial_backlog - backlog) / initial_backlog * 100) if initial_backlog > 0 else 0
    cost_per_update = cost / max(total_processed, 1)
    
    # Fairness index: higher = more equitable (considers backlog cleared and cost efficiency)
    # Range 0-1, where 1 = fully cleared backlog with low cost per update
    backlog_cleared_ratio = 1 - (backlog / max(initial_backlog + weekly_demand * duration, 1))
    cost_efficiency = 1 - min(cost_per_update / 500, 1)  # Benchmark: ₹500/update is baseline
    fairness_index = (backlog_cleared_ratio * 0.7 + cost_efficiency * 0.3)
    
    return SimulationResult(
        district=district,
        state=state,
        intervention=intervention_name,
        initial_backlog=initial_backlog,
        final_backlog=backlog,
        backlog_reduction=backlog_reduction,
        reduction_pct=reduction_pct,
        total_cost=cost,
        cost_per_update=cost_per_update,
        weeks_simulated=duration,
        scenario=scenario,
        fairness_index=fairness_index
    )


def run_monte_carlo_simulation(
    district_data: Dict,
    intervention: Dict,
    intervention_name: str,
    n_runs: int = 1000,
    forecast_lower: Optional[float] = None,
    forecast_upper: Optional[float] = None
) -> Dict:
    """
    Run Monte Carlo simulation to propagate forecast uncertainty.
    
    Args:
        district_data: District information
        intervention: Intervention configuration
        intervention_name: Name of intervention
        n_runs: Number of Monte Carlo runs
        forecast_lower: Lower bound of weekly demand forecast
        forecast_upper: Upper bound of weekly demand forecast
    
    Returns:
        Dict with percentile-based impact ranges
    """
    # Get demand bounds
    base_demand = district_data.get('weekly_demand', 100)
    lower = forecast_lower or base_demand * 0.7
    upper = forecast_upper or base_demand * 1.3
    
    results = []
    
    for _ in range(n_runs):
        # Sample demand from uniform distribution
        sampled_demand = np.random.uniform(lower, upper)
        
        # Sample effectiveness scenario
        scenario = np.random.choice(['conservative', 'median', 'optimistic'], p=[0.25, 0.5, 0.25])
        
        # Run simulation
        result = simulate_intervention(
            district_data, intervention, intervention_name,
            scenario=scenario, demand_override=sampled_demand
        )
        
        results.append({
            'backlog_reduction': result.backlog_reduction,
            'final_backlog': result.final_backlog,
            'reduction_pct': result.reduction_pct,
            'cost_per_update': result.cost_per_update
        })
    
    # Aggregate results
    df = pd.DataFrame(results)
    
    return {
        'district': district_data.get('district', 'Unknown'),
        'state': district_data.get('state', 'Unknown'),
        'intervention': intervention_name,
        'n_runs': n_runs,
        'backlog_reduction': {
            'p5': df['backlog_reduction'].quantile(0.05),
            'p50': df['backlog_reduction'].quantile(0.50),
            'p95': df['backlog_reduction'].quantile(0.95)
        },
        'reduction_pct': {
            'p5': df['reduction_pct'].quantile(0.05),
            'p50': df['reduction_pct'].quantile(0.50),
            'p95': df['reduction_pct'].quantile(0.95)
        },
        'cost_per_update': {
            'p5': df['cost_per_update'].quantile(0.05),
            'p50': df['cost_per_update'].quantile(0.50),
            'p95': df['cost_per_update'].quantile(0.95)
        },
        'total_cost': intervention.get('cost', 0),
        'confidence_interval': '90%'
    }


def simulate_all_interventions(
    district_data: Dict,
    interventions: Dict
) -> List[Dict]:
    """
    Simulate all possible interventions for a district.
    """
    results = []
    
    for name, config in interventions.items():
        mc_result = run_monte_carlo_simulation(
            district_data, config, name, n_runs=500
        )
        results.append(mc_result)
    
    return results


def run_bulk_simulation(
    n_districts: int = 20,
    output_path: str = 'outputs/sim_results/bulk_simulation.json'
) -> List[Dict]:
    """
    Run simulation for top priority districts.
    """
    print("=" * 60)
    print("POLICY INTERVENTION SIMULATOR")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading data...")
    interventions = load_interventions()
    priority = load_priority_data()
    print(f"  Loaded {len(interventions)} interventions")
    print(f"  Loaded {len(priority)} districts")
    
    # Get top priority districts
    top_districts = priority.nsmallest(n_districts, 'priority_rank')
    print(f"  Selected top {n_districts} priority districts")
    
    # Run simulations
    print("\n[2/3] Running simulations...")
    all_results = []
    
    for _, row in top_districts.iterrows():
        district_data = {
            'state': row['state'],
            'district': row['district'],
            'backlog': row.get('forecasted_demand_next_4w', 500),
            'weekly_demand': row.get('yhat', 100),
            'baseline_capacity': row.get('yhat', 100) * 0.7
        }
        
        # Simulate all interventions
        for interv_name, interv_config in interventions.items():
            mc_result = run_monte_carlo_simulation(
                district_data, interv_config, interv_name,
                n_runs=200,
                forecast_lower=row.get('yhat_lower', district_data['weekly_demand'] * 0.8),
                forecast_upper=row.get('yhat_upper', district_data['weekly_demand'] * 1.2)
            )
            mc_result['priority_score'] = row.get('priority_score', 0)
            mc_result['bottleneck_label'] = row.get('bottleneck_label', 'UNKNOWN')
            all_results.append(mc_result)
    
    # Save results
    print("\n[3/3] Saving results...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {len(all_results)} simulation results to {output_path}")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    return all_results


if __name__ == '__main__':
    results = run_bulk_simulation()
    
    # Print sample
    print("\nSample result:")
    if results:
        sample = results[0]
        print(f"District: {sample['district']} ({sample['state']})")
        print(f"Intervention: {sample['intervention']}")
        print(f"Reduction: {sample['backlog_reduction']['p50']:.0f} (90% CI: {sample['backlog_reduction']['p5']:.0f}-{sample['backlog_reduction']['p95']:.0f})")
