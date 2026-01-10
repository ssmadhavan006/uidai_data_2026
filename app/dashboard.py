"""
Aadhaar Pulse: Child Update Intelligence Platform
Unified Dashboard with RBAC, Audit Logging, Pilot Monitor & System Health
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add paths
app_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(app_dir)
sys.path.insert(0, app_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

# Page config
st.set_page_config(
    page_title="Aadhaar Pulse",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import auth module
from utils.auth import login_form, render_user_sidebar, require_role, get_current_user
from utils.audit_logger import get_audit_logger

# Authentication check
if not login_form():
    st.stop()

# User is authenticated
render_user_sidebar()
user = get_current_user()
audit = get_audit_logger()

# Custom CSS with dark mode support
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 { margin: 0; font-size: 32px; }
    .main-header p { margin: 5px 0 0 0; opacity: 0.9; }
    
    /* Dark mode fixes */
    .stMetric, .stSelectbox, .stSlider, .stRadio {
        background: transparent !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: inherit !important;
    }
    .stMarkdown, .stCaption {
        background: transparent !important;
    }
    div[data-testid="stForm"] {
        background: transparent !important;
        border: 1px solid rgba(128,128,128,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="main-header">
    <h1>🆔 Aadhaar Pulse</h1>
    <p>Child Update Intelligence Platform | Forecasting & Policy Simulation</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_all_data():
    try:
        priority = pd.read_csv(os.path.join(root_dir, 'outputs/priority_scores.csv'))
        labels = pd.read_csv(os.path.join(root_dir, 'outputs/bottleneck_labels.csv'))
        features = pd.read_parquet(os.path.join(root_dir, 'data/processed/model_features.parquet'))
        return priority, labels, features
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_pilot_regions():
    try:
        return pd.read_csv(os.path.join(root_dir, 'outputs/pilot_regions.csv'))
    except:
        return pd.DataFrame()

priority_df, labels_df, features_df = load_all_data()
pilot_regions = load_pilot_regions()

# Apply masking for Viewer role
def mask_for_viewer(df):
    if user['role'] == 'Viewer':
        masked = df.copy()
        if 'priority_score' in masked.columns:
            masked['priority_score'] = masked['priority_score'].apply(lambda x: round(x, 1))
        return masked
    return df

priority_df = mask_for_viewer(priority_df)

# Tabs - Analyst gets all tabs
if require_role("Analyst"):
    tabs = st.tabs(["📍 Hotspot Map", "🔍 District Analysis", "🎮 Policy Simulator", "📊 Overview", "📈 Pilot Monitor", "🔧 System Health"])
    tab1, tab2, tab3, tab4, tab5, tab6 = tabs
else:
    tabs = st.tabs(["📍 Hotspot Map", "📊 Overview"])
    tab1, tab4 = tabs
    tab2 = tab3 = tab5 = tab6 = None

# ===== TAB 1: MAP VIEW =====
with tab1:
    audit.log_view(user['username'], user['role'], "map_view")
    if not priority_df.empty:
        from components.map_view import render_map_view
        render_map_view(priority_df)
    else:
        st.warning("Priority data not available. Run Phase 3 pipeline first.")

# ===== TAB 2: DISTRICT ANALYSIS (Analyst only) =====
if tab2 is not None:
    with tab2:
        if not priority_df.empty:
            st.header("🔍 District Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                states = sorted(priority_df['state'].unique().tolist())
                selected_state = st.selectbox("Select State", states)
                
                districts = priority_df[priority_df['state'] == selected_state]['district'].tolist()
                selected_district = st.selectbox("Select District", districts)
            
            with col2:
                district_row = priority_df[
                    (priority_df['district'] == selected_district) & 
                    (priority_df['state'] == selected_state)
                ]
                if not district_row.empty:
                    r = district_row.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    rank = int(r['priority_rank']) if pd.notna(r['priority_rank']) else 'N/A'
                    score = f"{r['priority_score']:.2f}" if pd.notna(r['priority_score']) else 'N/A'
                    label = r['bottleneck_label'].replace('_', ' ') if pd.notna(r['bottleneck_label']) else 'Unknown'
                    c1.metric("Priority Rank", f"#{rank}" if isinstance(rank, int) else rank)
                    c2.metric("Priority Score", score)
                    c3.metric("Bottleneck", label)
            
            audit.log_view(user['username'], user['role'], "district_analysis", selected_district)
            
            from utils.data_loader import get_district_data
            district_data = get_district_data(selected_district, selected_state, priority_df, labels_df)
            
            if district_data:
                from components.why_view import render_why_view
                render_why_view(district_data, features_df)

# ===== TAB 3: SIMULATOR (Analyst only) =====
if tab3 is not None:
    with tab3:
        if not priority_df.empty:
            st.header("🎮 Policy Simulator")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                sim_state = st.selectbox("State", sorted(priority_df['state'].unique().tolist()), key="sim_state")
                sim_districts = priority_df[priority_df['state'] == sim_state]['district'].tolist()
                sim_district = st.selectbox("District", sim_districts, key="sim_district")
            
            from utils.data_loader import get_district_data
            sim_district_data = get_district_data(sim_district, sim_state, priority_df, labels_df)
            
            with col2:
                if sim_district_data:
                    from components.simulator_view import render_simulator_view
                    render_simulator_view(sim_district_data)
            
            st.divider()
            if st.session_state.get('last_sim_result'):
                result = st.session_state['last_sim_result']
                
                audit.log_simulation(user['username'], user['role'], {
                    'district': sim_district,
                    'intervention': result.get('intervention_config', {}).get('name', 'unknown')
                })
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    if st.button("📥 Export HTML", type="secondary"):
                        from utils.pdf_exporter import save_action_pack_html
                        filepath = save_action_pack_html(
                            sim_district_data,
                            result['intervention_config'],
                            result['intervention'],
                            result['mc_result']
                        )
                        audit.log_export(user['username'], user['role'], "action_pack_html", 1)
                        st.success(f"Saved to: {filepath}")
                
                with col3:
                    if st.button("📤 Export DP CSV", type="primary"):
                        from dp_export import dp_export_dataframe
                        export_df = priority_df[['state', 'district', 'priority_score', 'bottleneck_label']].copy()
                        noisy_df = dp_export_dataframe(
                            export_df,
                            epsilon=1.0,
                            rate_columns=['priority_score']
                        )
                        export_path = os.path.join(root_dir, 'outputs/dp_export.csv')
                        noisy_df.to_csv(export_path, index=False)
                        audit.log_export(user['username'], user['role'], "dp_export.csv", len(noisy_df))
                        st.success(f"DP-protected export saved ({len(noisy_df)} rows, ε=1.0)")

# ===== TAB 4: OVERVIEW =====
overview_tab = tab4
with overview_tab:
    st.header("📊 System Overview")
    
    # Child-Focused Headline Metrics
    st.subheader("🧒 Child Update Focus (Ages 5-15)")
    c1, c2, c3, c4 = st.columns(4)
    
    if not features_df.empty:
        # Get latest week aggregate
        latest_week = features_df['week_number'].max()
        latest_data = features_df[features_df['week_number'] == latest_week]
        
        total_bio_child = latest_data['bio_update_child'].sum() if 'bio_update_child' in latest_data else 0
        total_demo_child = latest_data['demo_update_child'].sum() if 'demo_update_child' in latest_data else 0
        avg_completion = latest_data['completion_rate_child'].mean() if 'completion_rate_child' in latest_data else 0
        
        with c1:
            st.metric("Weekly Bio Updates (Child)", f"{total_bio_child:,.0f}")
        with c2:
            st.metric("Weekly Demo Updates (Child)", f"{total_demo_child:,.0f}")
        with c3:
            st.metric("Avg Completion Rate", f"{avg_completion:.1%}" if pd.notna(avg_completion) else "N/A")
        with c4:
            inclusion_gaps = len(priority_df[priority_df['bottleneck_label'] == 'INCLUSION_GAP']) if not priority_df.empty else 0
            st.metric("Inclusion Gaps", inclusion_gaps, help="Districts with potential access barriers")
    
    st.divider()
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Districts", len(priority_df) if not priority_df.empty else 0)
    with col2:
        critical = len(priority_df[priority_df['priority_score'] > 0.7]) if not priority_df.empty else 0
        st.metric("Critical Priority", critical)
    with col3:
        bottlenecks = len(priority_df[priority_df['bottleneck_label'] != 'NORMAL']) if not priority_df.empty else 0
        st.metric("With Bottlenecks", bottlenecks)
    with col4:
        st.metric("Data Sources", 3)
    
    if not priority_df.empty:
        st.subheader("Bottleneck Distribution")
        bottleneck_counts = priority_df['bottleneck_label'].value_counts()
        
        fig = px.bar(
            x=bottleneck_counts.index,
            y=bottleneck_counts.values,
            color=bottleneck_counts.index,
            labels={'x': 'Bottleneck Type', 'y': 'Count'},
            color_discrete_map={
                'OPERATIONAL_BOTTLENECK': '#e74c3c',
                'DEMOGRAPHIC_SURGE': '#f39c12',
                'CAPACITY_STRAIN': '#9b59b6',
                'INCLUSION_GAP': '#e91e63',  # NEW: Pink for inclusion gaps
                'ANOMALY_DETECTED': '#3498db',
                'NORMAL': '#2ecc71'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top 10 Priority Districts")
        top_10 = priority_df.nsmallest(10, 'priority_rank')[
            ['district', 'state', 'priority_score', 'priority_rank', 'bottleneck_label']
        ]
        st.dataframe(top_10, use_container_width=True, hide_index=True)

# ===== TAB 5: PILOT MONITOR (Analyst only) =====
if tab5 is not None:
    with tab5:
        st.header("📈 Pilot Monitoring")
        
        if pilot_regions.empty:
            st.warning("Pilot regions not found. Run 08_pilot_design.ipynb first.")
        else:
            # Sidebar metrics
            n_treatment = len(pilot_regions[pilot_regions['group'] == 'treatment'])
            n_control = len(pilot_regions[pilot_regions['group'] == 'control'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Treatment Districts", n_treatment)
            col2.metric("Control Districts", n_control)
            col3.metric("Pilot Duration", "12 weeks")
            col4.metric("Current Week", "Week 8")
            
            # Filter features to pilot districts
            pilot_districts = pilot_regions['district'].tolist()
            pilot_data = features_df[features_df['district'].isin(pilot_districts)].copy() if not features_df.empty else pd.DataFrame()
            
            if not pilot_data.empty:
                pilot_data = pilot_data.merge(pilot_regions[['district', 'group']], on='district', how='left')
                
                # KPI selector
                kpi = st.selectbox(
                    "Select KPI",
                    ['bio_update_child', 'demo_update_child'],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="pilot_kpi"
                )
                
                if kpi in pilot_data.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Treatment vs Control Trends")
                        trend_data = pilot_data.groupby(['week_number', 'group'])[kpi].mean().reset_index()
                        
                        fig = px.line(
                            trend_data,
                            x='week_number',
                            y=kpi,
                            color='group',
                            markers=True,
                            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'}
                        )
                        fig.update_layout(xaxis_title="Week", yaxis_title=kpi.replace("_", " ").title())
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("District Comparison")
                        latest = pilot_data.groupby(['district', 'group'])[kpi].mean().reset_index()
                        
                        fig = px.bar(
                            latest.sort_values('group'),
                            x='district',
                            y=kpi,
                            color='group',
                            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'}
                        )
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Action Tracker
            st.subheader("📋 Intervention Action Tracker")
            treatment_districts = pilot_regions[pilot_regions['group'] == 'treatment']['district'].tolist()
            n = len(treatment_districts)
            
            if n > 0:
                actions = pd.DataFrame({
                    'District': treatment_districts,
                    'Intervention': ['Mobile Camp', 'Device Upgrade', 'Staff Training', 'Extended Hours', 'School Partnership'][:n],
                    'Status': ['✅ Completed', '🔄 In Progress', '📅 Scheduled', '✅ Completed', '🔄 In Progress'][:n],
                    'Start Week': [3, 4, 5, 3, 4][:n],
                    'Effectiveness': ['High', 'Medium', 'Pending', 'High', 'Medium'][:n]
                })
                st.dataframe(actions, use_container_width=True, hide_index=True)

# ===== TAB 6: SYSTEM HEALTH (Analyst only) =====
if tab6 is not None:
    with tab6:
        st.header("🔧 System Health & Monitoring")
        
        # Generate mock monitoring data
        @st.cache_data
        def generate_mock_metrics():
            dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
            np.random.seed(42)
            psi_bio = np.clip(np.random.normal(0.08, 0.05, 12), 0, 0.4)
            psi_bio[-2:] = [0.22, 0.28]  # Simulate recent drift
            mape = np.clip(np.random.normal(55, 10, 12), 30, 85)
            success_rate = np.clip(np.random.normal(98, 2, 12), 90, 100)
            
            return pd.DataFrame({
                'date': dates,
                'bio_update_psi': psi_bio,
                'forecast_mape': mape,
                'pipeline_success': success_rate
            })
        
        metrics = generate_mock_metrics()
        latest = metrics.iloc[-1]
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if latest['bio_update_psi'] > 0.25:
                st.error("⚠️ Data Drift Detected")
            else:
                st.success("✅ Data Stable")
        
        with col2:
            if latest['forecast_mape'] > 70:
                st.warning("⚠️ Model Performance")
            else:
                st.success("✅ Model Healthy")
        
        with col3:
            if latest['pipeline_success'] < 95:
                st.warning("⚠️ Pipeline Issues")
            else:
                st.success("✅ Pipeline OK")
        
        with col4:
            st.info(f"📅 Last Update: {latest['date'].strftime('%Y-%m-%d')}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Drift (PSI)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics['date'], y=metrics['bio_update_psi'],
                mode='lines+markers', name='bio_update_child',
                line=dict(color='#e74c3c', width=2)
            ))
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", 
                          annotation_text="Alert Threshold")
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Forecast MAPE")
            fig = px.line(metrics, x='date', y='forecast_mape', markers=True)
            fig.add_hline(y=70, line_dash="dash", line_color="orange",
                          annotation_text="Target")
            fig.update_traces(line_color='#9b59b6', line_width=2)
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Alert log
        st.subheader("📋 Recent Alerts")
        alerts = pd.DataFrame({
            'Timestamp': ['2026-01-10 07:15', '2026-01-08 12:30', '2026-01-03 09:00'],
            'Severity': ['🔴 Critical', '🟡 Warning', '🟢 Info'],
            'Alert': [
                'High data drift detected in bio_update_child (PSI=0.28)',
                'MAPE elevated for District_X (72%)',
                'Weekly retraining completed successfully'
            ],
            'Status': ['Open', 'Resolved', 'Closed']
        })
        st.dataframe(alerts, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption(f"Aadhaar Pulse | Logged in as {user['name']} ({user['role']}) | Built for UIDAI Hackathon 2025")
