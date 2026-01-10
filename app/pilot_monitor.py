"""
pilot_monitor.py
Streamlit dashboard for monitoring pilot treatment vs control districts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add paths
app_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(app_dir)
sys.path.insert(0, root_dir)

st.set_page_config(
    page_title="Pilot Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Pilot Monitoring Dashboard")
st.caption("Treatment vs Control District Comparison")

# Load pilot regions
@st.cache_data
def load_pilot_data():
    try:
        regions = pd.read_csv(os.path.join(root_dir, 'outputs/pilot_regions.csv'))
        features = pd.read_parquet(os.path.join(root_dir, 'data/processed/model_features.parquet'))
        
        # Filter to pilot districts
        pilot_districts = regions['district'].tolist()
        pilot_data = features[features['district'].isin(pilot_districts)].copy()
        
        # Merge group info
        pilot_data = pilot_data.merge(
            regions[['district', 'group']], 
            on='district', 
            how='left'
        )
        
        return regions, pilot_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

regions, pilot_data = load_pilot_data()

if regions.empty:
    st.warning("Pilot regions not found. Run 08_pilot_design.ipynb first.")
    st.stop()

# Sidebar
st.sidebar.header("Pilot Status")
n_treatment = len(regions[regions['group'] == 'treatment'])
n_control = len(regions[regions['group'] == 'control'])
st.sidebar.metric("Treatment Districts", n_treatment)
st.sidebar.metric("Control Districts", n_control)

# KPI Selection
kpi = st.selectbox(
    "Select KPI",
    ['bio_update_child', 'completion_rate_child', 'update_backlog_child'],
    format_func=lambda x: x.replace('_', ' ').title()
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Treatment vs Control Trends")
    
    if not pilot_data.empty and kpi in pilot_data.columns:
        # Aggregate by week and group
        trend_data = pilot_data.groupby(['week_number', 'group'])[kpi].mean().reset_index()
        
        fig = px.line(
            trend_data,
            x='week_number',
            y=kpi,
            color='group',
            markers=True,
            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'},
            title=f'{kpi.replace("_", " ").title()} by Week'
        )
        fig.update_layout(
            xaxis_title="Week",
            yaxis_title=kpi.replace("_", " ").title(),
            legend_title="Group"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data available")

with col2:
    st.subheader("District Comparison")
    
    if not pilot_data.empty and kpi in pilot_data.columns:
        # Latest week comparison
        latest = pilot_data.groupby(['district', 'group'])[kpi].mean().reset_index()
        
        fig = px.bar(
            latest.sort_values('group'),
            x='district',
            y=kpi,
            color='group',
            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'},
            title=f'Current {kpi.replace("_", " ").title()} by District'
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# Action Tracker
st.subheader("📋 Intervention Action Tracker")

# Simulated action tracker data - match actual treatment districts
treatment_districts = regions[regions['group'] == 'treatment']['district'].tolist()
n_treatment = len(treatment_districts)

if n_treatment > 0:
    interventions = ['Mobile Camp', 'Device Upgrade', 'Staff Training', 'Extended Hours', 'School Partnership'][:n_treatment]
    statuses = ['✅ Completed', '🔄 In Progress', '📅 Scheduled', '✅ Completed', '🔄 In Progress'][:n_treatment]
    weeks = [3, 4, 5, 3, 4][:n_treatment]
    effects = ['High', 'Medium', 'Pending', 'High', 'Medium'][:n_treatment]
    
    actions = pd.DataFrame({
        'District': treatment_districts,
        'Intervention': interventions,
        'Status': statuses,
        'Start Week': weeks,
        'Effectiveness': effects
    })
    st.dataframe(actions, use_container_width=True, hide_index=True)
else:
    st.info("No treatment districts found")

# Summary Metrics
st.subheader("📈 Pilot Summary")
m1, m2, m3, m4 = st.columns(4)

if not pilot_data.empty and kpi in pilot_data.columns:
    treatment_mean = pilot_data[pilot_data['group'] == 'treatment'][kpi].mean()
    control_mean = pilot_data[pilot_data['group'] == 'control'][kpi].mean()
    diff = treatment_mean - control_mean
    pct_diff = (diff / control_mean * 100) if control_mean > 0 else 0
    
    with m1:
        st.metric("Treatment Mean", f"{treatment_mean:,.0f}")
    with m2:
        st.metric("Control Mean", f"{control_mean:,.0f}")
    with m3:
        st.metric("Difference", f"{diff:+,.0f}", f"{pct_diff:+.1f}%")
    with m4:
        st.metric("Pilot Week", "8 of 12")

st.divider()
st.caption("Pilot Monitor | Aadhaar Pulse Phase 8")
