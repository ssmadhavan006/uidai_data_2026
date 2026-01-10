"""
monitor.py
Production monitoring dashboard for model drift and system health.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="System Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Production Monitoring Dashboard")
st.caption("Data Drift | Model Performance | System Health")

# Generate mock monitoring data
@st.cache_data
def generate_mock_metrics():
    """Generate simulated monitoring metrics."""
    dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
    
    # PSI scores (should stay low)
    np.random.seed(42)
    psi_bio = np.clip(np.random.normal(0.08, 0.05, 12), 0, 0.4)
    psi_bio[-2:] = [0.22, 0.28]  # Simulate drift in recent weeks
    
    psi_demo = np.clip(np.random.normal(0.05, 0.03, 12), 0, 0.3)
    
    # MAPE over time (should stay stable)
    mape = np.clip(np.random.normal(55, 10, 12), 30, 85)
    
    # Pipeline success
    success_rate = np.clip(np.random.normal(98, 2, 12), 90, 100)
    
    return pd.DataFrame({
        'date': dates,
        'bio_update_psi': psi_bio,
        'demo_update_psi': psi_demo,
        'forecast_mape': mape,
        'pipeline_success': success_rate
    })

metrics = generate_mock_metrics()

# Sidebar - Overall Status
st.sidebar.header("System Status")
latest = metrics.iloc[-1]

if latest['bio_update_psi'] > 0.25:
    st.sidebar.error("⚠️ Data Drift Detected")
else:
    st.sidebar.success("✅ Data Stable")

if latest['forecast_mape'] > 70:
    st.sidebar.warning("⚠️ Model Performance Warning")
else:
    st.sidebar.success("✅ Model Healthy")

if latest['pipeline_success'] < 95:
    st.sidebar.warning("⚠️ Pipeline Issues")
else:
    st.sidebar.success("✅ Pipeline OK")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Data Drift", "📈 Model Performance", "🔧 System Health"])

with tab1:
    st.subheader("Population Stability Index (PSI)")
    st.caption("PSI > 0.25 indicates significant data drift requiring attention")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['date'], y=metrics['bio_update_psi'],
        mode='lines+markers', name='bio_update_child',
        line=dict(color='#e74c3c', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=metrics['date'], y=metrics['demo_update_psi'],
        mode='lines+markers', name='demo_update_child',
        line=dict(color='#3498db', width=2)
    ))
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                  annotation_text="Alert Threshold")
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="PSI Score",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert status
    if latest['bio_update_psi'] > 0.25:
        st.error(f"🚨 ALERT: bio_update_child PSI = {latest['bio_update_psi']:.2f} exceeds threshold")

with tab2:
    st.subheader("Forecast MAPE Over Time")
    st.caption("Target: MAPE ≤ 70% for cross-sectional data")
    
    fig = px.line(
        metrics, x='date', y='forecast_mape',
        markers=True,
        title="Weekly Forecast Error"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="orange",
                  annotation_text="Target Threshold")
    fig.update_traces(line_color='#9b59b6', line_width=2)
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # District-level errors (mock)
    st.subheader("Prediction Error by District")
    districts = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
    errors = np.random.normal(50, 15, 5)
    
    fig = px.bar(x=districts, y=errors, color=errors,
                 color_continuous_scale=['green', 'yellow', 'red'],
                 title="MAPE by District (Latest Week)")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Pipeline Success Rate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            metrics, x='date', y='pipeline_success',
            markers=True,
            title="Weekly Pipeline Success Rate (%)"
        )
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                      annotation_text="SLA Target (95%)")
        fig.update_traces(line_color='#2ecc71', line_width=2)
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Latest Pipeline Runs")
        runs = pd.DataFrame({
            'Pipeline': ['ETL', 'Model Training', 'Priority Scoring', 'Forecast'],
            'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success'],
            'Duration': ['45 min', '12 min', '3 min', '8 min'],
            'Last Run': ['2026-01-10 06:00', '2026-01-10 06:45', '2026-01-10 06:48', '2026-01-10 06:56']
        })
        st.dataframe(runs, use_container_width=True, hide_index=True)

# Alert Log
st.subheader("📋 Alert History")
alerts = pd.DataFrame({
    'Timestamp': ['2026-01-10 07:15', '2026-01-08 12:30', '2026-01-03 09:00'],
    'Severity': ['🔴 Critical', '🟡 Warning', '🟢 Info'],
    'Alert': [
        'High data drift detected in bio_update_child (PSI=0.28)',
        'MAPE slightly elevated for District_X (72%)',
        'Weekly retraining completed successfully'
    ],
    'Status': ['Open', 'Resolved', 'Closed']
})
st.dataframe(alerts, use_container_width=True, hide_index=True)

st.divider()
st.caption("Aadhaar Pulse | Production Monitoring | Phase 9")
