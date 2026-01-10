"""
data_loader.py
Cached data loading utilities for the dashboard.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path


@st.cache_data
def load_priority_scores(path: str = 'outputs/priority_scores.csv') -> pd.DataFrame:
    """Load priority scores with caching."""
    return pd.read_csv(path)


@st.cache_data
def load_bottleneck_labels(path: str = 'outputs/bottleneck_labels.csv') -> pd.DataFrame:
    """Load bottleneck labels with caching."""
    return pd.read_csv(path)


@st.cache_data
def load_forecasts(path: str = 'outputs/forecasts.csv') -> pd.DataFrame:
    """Load forecasts with caching."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_features(path: str = 'data/processed/model_features.parquet') -> pd.DataFrame:
    """Load feature data with caching."""
    return pd.read_parquet(path)


@st.cache_data
def load_interventions(path: str = 'config/interventions.json') -> dict:
    """Load intervention definitions with caching."""
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_simulation_results(path: str = 'outputs/sim_results/recommended_actions.json') -> dict:
    """Load simulation results with caching."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def mask_small_values(value, threshold: int = 10) -> str:
    """Mask values below k-anonymity threshold."""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)) and value > 0 and value < threshold:
        return "≤10"
    return str(int(value)) if isinstance(value, float) else str(value)


def get_district_list(priority_df: pd.DataFrame) -> list:
    """Get sorted list of districts."""
    return sorted(priority_df['district'].unique().tolist())


def get_district_data(district: str, state: str, priority_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    """Get combined data for a district with sensible defaults."""
    priority_row = priority_df[
        (priority_df['district'] == district) & (priority_df['state'] == state)
    ]
    labels_row = labels_df[
        (labels_df['district'] == district) & (labels_df['state'] == state)
    ]
    
    if priority_row.empty:
        return {}
    
    p = priority_row.iloc[0]
    l = labels_row.iloc[0] if not labels_row.empty else pd.Series()
    
    # Get values with sensible defaults, handling NaN
    def safe_get(row, key, default):
        val = row.get(key, default)
        return default if pd.isna(val) else val
    
    # Forecast demand - fallback to a reasonable estimate
    forecast = safe_get(p, 'forecasted_demand_next_4w', 0)
    if forecast <= 0:
        forecast = safe_get(l, 'current_bio_updates', 500) * 4  # 4 weeks
    if forecast <= 0:
        forecast = 1000  # Default minimum
    
    weekly_demand = forecast / 4
    
    # Backlog - use update_backlog or estimate from forecast
    backlog = safe_get(l, 'update_backlog', 0)
    if backlog <= 0:
        backlog = weekly_demand * 2  # Assume 2 weeks backlog if not available
    
    return {
        'district': district,
        'state': state,
        'priority_score': safe_get(p, 'priority_score', 0.5),
        'priority_rank': safe_get(p, 'priority_rank', 999),
        'bottleneck_label': safe_get(p, 'bottleneck_label', 'UNKNOWN'),
        'forecasted_demand': forecast,
        'rationale': safe_get(l, 'rationale', 'No rationale available'),
        'current_bio_updates': safe_get(l, 'current_bio_updates', weekly_demand),
        'current_demo_updates': safe_get(l, 'current_demo_updates', 0),
        'update_backlog': backlog,
        'completion_rate': safe_get(l, 'completion_rate', 0.7),
        'backlog': max(backlog, weekly_demand),  # Ensure non-zero for simulation
        'weekly_demand': max(weekly_demand, 100),  # Minimum 100/week
        'baseline_capacity': max(weekly_demand * 0.7, 70)  # 70% of demand
    }
