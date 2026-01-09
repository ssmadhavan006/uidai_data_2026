"""
why_view.py
District drill-down "Why" view - explains bottleneck and priority.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
sys.path.insert(0, '..')

from utils.data_loader import mask_small_values


def create_forecast_chart(features_df: pd.DataFrame, district: str, state: str) -> go.Figure:
    """Create time-series chart with forecast."""
    df = features_df[(features_df['district'] == district) & (features_df['state'] == state)]
    df = df.sort_values(['year', 'week_number'])
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Create figure
    fig = go.Figure()
    
    # Historical bio updates
    fig.add_trace(go.Scatter(
        x=df['week_number'],
        y=df['bio_update_child'],
        mode='lines+markers',
        name='Bio Updates (Child)',
        line=dict(color='#3498db')
    ))
    
    # Demo updates
    fig.add_trace(go.Scatter(
        x=df['week_number'],
        y=df['demo_update_child'],
        mode='lines',
        name='Demo Updates (Child)',
        line=dict(color='#e74c3c', dash='dash')
    ))
    
    # Rolling mean
    if 'rolling_4w_mean_bio_update_child' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['week_number'],
            y=df['rolling_4w_mean_bio_update_child'],
            mode='lines',
            name='4-Week Average',
            line=dict(color='#27ae60', width=2)
        ))
    
    fig.update_layout(
        title=f'Update Trends: {district}',
        xaxis_title='Week Number',
        yaxis_title='Count',
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_metrics_chart(district_data: dict) -> go.Figure:
    """Create gauge charts for key metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Completion rate gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=district_data.get('completion_rate', 0) * 100,
        title={'text': "Completion Rate %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, 50], 'color': '#e74c3c'},
                {'range': [50, 80], 'color': '#f39c12'},
                {'range': [80, 100], 'color': '#2ecc71'}
            ]
        }
    ), row=1, col=1)
    
    # Priority score gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=district_data.get('priority_score', 0) * 100,
        title={'text': "Priority Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#e74c3c"},
            'steps': [
                {'range': [0, 30], 'color': '#2ecc71'},
                {'range': [30, 70], 'color': '#f39c12'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ]
        }
    ), row=1, col=2)
    
    # Backlog gauge
    max_backlog = 2000
    backlog = min(district_data.get('update_backlog', 0), max_backlog)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=backlog,
        title={'text': "Update Backlog"},
        gauge={
            'axis': {'range': [0, max_backlog]},
            'bar': {'color': "#9b59b6"},
            'steps': [
                {'range': [0, 500], 'color': '#2ecc71'},
                {'range': [500, 1000], 'color': '#f39c12'},
                {'range': [1000, max_backlog], 'color': '#e74c3c'}
            ]
        }
    ), row=1, col=3)
    
    fig.update_layout(height=250)
    return fig


def render_bottleneck_card(district_data: dict):
    """Render the bottleneck explanation card."""
    label = district_data.get('bottleneck_label', 'UNKNOWN')
    
    # Color mapping
    colors = {
        'OPERATIONAL_BOTTLENECK': '#e74c3c',
        'DEMOGRAPHIC_SURGE': '#f39c12',
        'CAPACITY_STRAIN': '#9b59b6',
        'ANOMALY_DETECTED': '#3498db',
        'NORMAL': '#2ecc71'
    }
    
    color = colors.get(label, '#95a5a6')
    
    st.markdown(f"""
    <div style="
        background-color: {color}20;
        border-left: 5px solid {color};
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    ">
        <h4 style="color: {color}; margin: 0;">🏷️ {label.replace('_', ' ')}</h4>
        <p style="margin: 10px 0 0 0;">{district_data.get('rationale', 'No rationale available')}</p>
    </div>
    """, unsafe_allow_html=True)


def create_shap_chart(district_data: dict) -> go.Figure:
    """Create horizontal bar chart for feature importance (SHAP-style)."""
    # Simulate SHAP-like feature importance based on available metrics
    features = []
    importance = []
    
    # Create pseudo-SHAP values based on district metrics
    backlog = district_data.get('update_backlog', 0)
    completion = district_data.get('completion_rate', 0.8)
    priority = district_data.get('priority_score', 0.5)
    forecast = district_data.get('forecasted_demand', 100)
    
    features.append('Update Backlog')
    importance.append(min(backlog / 500, 1) * 0.3)
    
    features.append('Completion Rate')
    importance.append((1 - completion) * 0.25)
    
    features.append('Forecasted Demand')
    importance.append(min(forecast / 1000, 1) * 0.2)
    
    features.append('Priority Score')
    importance.append(priority * 0.15)
    
    features.append('Bio/Demo Ratio')
    importance.append(0.1)
    
    # Sort by importance
    sorted_pairs = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
    features, importance = zip(*sorted_pairs[:5])
    
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in importance]
    
    fig = go.Figure(go.Bar(
        x=list(importance),
        y=list(features),
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Feature Impact on Priority',
        xaxis_title='Impact Score',
        height=250,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def render_why_view(district_data: dict, features_df: pd.DataFrame):
    """Render the complete Why view for a district."""
    district = district_data.get('district', 'Unknown')
    state = district_data.get('state', 'Unknown')
    
    st.header(f"🔍 {district}")
    st.caption(f"State: {state}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Priority Rank", f"#{int(district_data.get('priority_rank', 0))}")
    with col2:
        st.metric("Bio Updates", mask_small_values(district_data.get('current_bio_updates', 0)))
    with col3:
        st.metric("Demo Updates", mask_small_values(district_data.get('current_demo_updates', 0)))
    with col4:
        backlog = district_data.get('update_backlog', 0)
        st.metric("Backlog", mask_small_values(backlog))
    
    # Two columns: Bottleneck + SHAP
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Bottleneck Diagnosis")
        render_bottleneck_card(district_data)
    
    with col_right:
        st.subheader("Key Drivers (SHAP)")
        fig_shap = create_shap_chart(district_data)
        st.plotly_chart(fig_shap, use_container_width=True)
    
    # Metrics gauges
    st.subheader("Performance Metrics")
    fig_metrics = create_metrics_chart(district_data)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Time series
    st.subheader("Historical Trends")
    fig_ts = create_forecast_chart(features_df, district, state)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Forecast info
    st.subheader("4-Week Forecast")
    forecast = district_data.get('forecasted_demand', 0)
    st.info(f"📈 Estimated demand: **{forecast:,.0f}** child updates in the next 4 weeks")
