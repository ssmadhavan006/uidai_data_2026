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
    """Get combined data for a district."""
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
    
    return {
        'district': district,
        'state': state,
        'priority_score': p.get('priority_score', 0),
        'priority_rank': p.get('priority_rank', 999),
        'bottleneck_label': p.get('bottleneck_label', 'UNKNOWN'),
        'forecasted_demand': p.get('forecasted_demand_next_4w', 0),
        'rationale': l.get('rationale', 'No rationale available'),
        'current_bio_updates': l.get('current_bio_updates', 0),
        'current_demo_updates': l.get('current_demo_updates', 0),
        'update_backlog': l.get('update_backlog', 0),
        'completion_rate': l.get('completion_rate', 0),
        'backlog': max(l.get('update_backlog', 0), 0),
        'weekly_demand': p.get('forecasted_demand_next_4w', 100) / 4,
        'baseline_capacity': p.get('forecasted_demand_next_4w', 100) / 4 * 0.7
    }
