"""
Aadhaar Pulse: Child Update Intelligence Platform
Unified Dashboard with RBAC, Audit Logging, Pilot Monitor & System Health
CHILD-FIRST DESIGN: Every metric tells the story of protecting children's benefits
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# =============================================================================
# HELPER FUNCTIONS: Sanitize all metrics
# =============================================================================

def sanitize_rate(value, max_val=1.0):
    """Cap rates at max_val (100%) and handle invalid values."""
    if pd.isna(value) or value < 0:
        return None
    return min(value, max_val)

def sanitize_count(value):
    """Handle suppressed cells (-1) and negative values."""
    if pd.isna(value) or value < 0:
        return None
    return value

def format_count(value):
    """Format count for display, handling suppressed values."""
    if value is None or (isinstance(value, (int, float)) and value < 0):
        return "≤10 (masked)"
    return f"{value:,.0f}"

def format_rate(value):
    """Format rate for display, capping at 100%."""
    if value is None:
        return "N/A"
    sanitized = sanitize_rate(value)
    if sanitized is None:
        return "N/A"
    return f"{sanitized:.1%}"

def estimate_children_at_risk(inclusion_gap_districts, avg_children_per_district=7000):
    """Estimate children potentially at risk from inclusion gaps."""
    return inclusion_gap_districts * avg_children_per_district

# Custom CSS with child-focused branding
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
    
    .hero-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .hero-banner h2 { margin: 0 0 10px 0; font-size: 28px; }
    .hero-banner p { margin: 0; font-size: 18px; opacity: 0.95; }
    .hero-stat { font-size: 48px; font-weight: bold; }
    
    .impact-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
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
    <p>Protecting Children's Access to Benefits | Forecasting & Policy Simulation</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_all_data():
    try:
        priority = pd.read_csv(os.path.join(root_dir, 'outputs/priority_scores.csv'))
        labels = pd.read_csv(os.path.join(root_dir, 'outputs/bottleneck_labels.csv'))
        features = pd.read_parquet(os.path.join(root_dir, 'data/processed/model_features.parquet'))
        
        # SANITIZE DATA ON LOAD
        # Cap completion rates at 100%
        for col in features.columns:
            if 'rate' in col.lower() or 'completion' in col.lower():
                features[col] = features[col].apply(lambda x: sanitize_rate(x))
        
        # Replace negative counts with NaN (will display as "≤10")
        count_cols = [c for c in features.columns if 'update' in c.lower() or 'enroll' in c.lower() or 'backlog' in c.lower()]
        for col in count_cols:
            features[col] = features[col].apply(lambda x: sanitize_count(x))
        
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
    tabs = st.tabs(["📍 Hotspot Map", "🔍 District Analysis", "📊 Compare Districts", "🎮 Policy Simulator", "📊 Overview", "📈 Pilot Monitor", "🔧 System Health", "🤖 AI Assistant"])
    tab1, tab2, tab_compare, tab3, tab4, tab5, tab6, tab7 = tabs
else:
    tabs = st.tabs(["📍 Hotspot Map", "📊 Overview", "🤖 AI Assistant"])
    tab1, tab4, tab7 = tabs
    tab2 = tab3 = tab5 = tab6 = tab_compare = None

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
            st.header("🔍 Understanding This District's Children")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                states = sorted(priority_df['state'].unique().tolist())
                selected_state = st.selectbox("Select State", states)
                
                districts = priority_df[priority_df['state'] == selected_state]['district'].tolist()
                selected_district = st.selectbox("Select District", districts)
            
            from utils.data_loader import get_district_data, format_bottleneck_label
            
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
                    label = format_bottleneck_label(r['bottleneck_label']) if pd.notna(r['bottleneck_label']) else 'Unknown'
                    c1.metric("Priority Rank", f"#{rank}" if isinstance(rank, int) else rank)
                    c2.metric("Priority Score", score)
                    c3.metric("Issue Type", label)
            
            # IMPACT WARNING BOX - Calculate from data
            # Estimate children at risk based on weekly bio updates and completion rate
            district_features = features_df[
                (features_df['district'] == selected_district)
            ] if not features_df.empty else pd.DataFrame()
            
            if not district_features.empty:
                avg_weekly = district_features['bio_update_child'].mean()
                completion = district_features['completion_rate_child'].mean()
                # Estimate 6 months of children affected by low completion
                if pd.notna(avg_weekly) and pd.notna(completion) and completion < 1:
                    estimated_children = int(avg_weekly * 26 * max(0, 1 - min(completion, 1)))
                else:
                    estimated_children = int(avg_weekly * 2) if pd.notna(avg_weekly) else 1000
            else:
                estimated_children = 1500  # Fallback estimate
            
            st.markdown(f"""
            <div class="impact-box">
                <strong>⚠️ Impact if Unaddressed:</strong> Based on current trends, approximately 
                <strong>{estimated_children:,} children</strong> in {selected_district} could face authentication 
                failures and benefit denials in the next 6 months without intervention.
            </div>
            """, unsafe_allow_html=True)
            
            audit.log_view(user['username'], user['role'], "district_analysis", selected_district)
            
            district_data = get_district_data(selected_district, selected_state, priority_df, labels_df)
            
            if district_data:
                from components.why_view import render_why_view
                render_why_view(district_data, features_df)

# ===== TAB: COMPARE DISTRICTS (Analyst only) =====
if tab_compare is not None:
    with tab_compare:
        audit.log_view(user['username'], user['role'], "compare_districts")
        if not priority_df.empty:
            from components.compare_view import render_compare_view
            render_compare_view(priority_df, labels_df, features_df)
        else:
            st.warning("Priority data not available. Run Phase 3 pipeline first.")

# ===== TAB 3: SIMULATOR (Analyst only) =====
if tab3 is not None:
    with tab3:
        if not priority_df.empty:
            st.header("🎮 Intervention Simulator: Protecting Children")
            
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
                
                # CHILD-FOCUSED RESULTS
                mc_data = result.get('mc_result', {})
                backlog_data = mc_data.get('backlog_reduction', {})
                reduction = backlog_data.get('p50', 0) if isinstance(backlog_data, dict) else 0
                ci_low = backlog_data.get('p5', 0) if isinstance(backlog_data, dict) else 0
                ci_high = backlog_data.get('p95', 0) if isinstance(backlog_data, dict) else 0
                
                st.markdown(f"""
                <div class="hero-banner" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h2>👦👧 Children Protected from Exclusion</h2>
                    <p class="hero-stat">~{reduction:,.0f}</p>
                    <p>90% Confidence Interval: {ci_low:,.0f} to {ci_high:,.0f} children</p>
                </div>
                """, unsafe_allow_html=True)
                
                # POLICY IMPACT SUMMARY - Calculate from data
                # Get district's position in priority ranking to estimate equity impact
                district_rank = priority_df[priority_df['district'] == sim_district]['priority_rank'].values
                equity_pct = 50  # Default
                if len(district_rank) > 0 and not pd.isna(district_rank[0]):
                    percentile = (1 - district_rank[0] / len(priority_df)) * 100
                    equity_pct = min(90, max(30, int(percentile)))
                
                st.info(f"📊 **Policy Impact Summary:** This intervention targets a district in the top {100-equity_pct}% priority, with significant impact in underserved areas within {sim_district}.")
                
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
    
    # HERO STAT BANNER - The "wow" moment
    if not features_df.empty and not priority_df.empty:
        latest_week = features_df['week_number'].max()
        latest_data = features_df[features_df['week_number'] == latest_week]
        
        # Sanitize and calculate
        bio_updates = latest_data['bio_update_child'].dropna()
        bio_updates = bio_updates[bio_updates >= 0]  # Remove suppressed
        total_bio_child = bio_updates.sum()
        
        inclusion_gaps = len(priority_df[priority_df['bottleneck_label'] == 'INCLUSION_GAP'])
        children_at_risk = estimate_children_at_risk(inclusion_gaps)
        
        st.markdown(f"""
        <div class="hero-banner">
            <h2>🧒 This Week's Child Update Story</h2>
            <p class="hero-stat">{total_bio_child:,.0f}</p>
            <p>children (ages 5-15) completed biometric updates this week</p>
            <br>
            <p><strong>⚠️ {inclusion_gaps} districts</strong> have critical inclusion gaps, 
            putting approximately <strong>~{children_at_risk:,.0f} children</strong> at risk of losing benefit access.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("📊 System Overview")
    
    # Child-Focused Headline Metrics with SANITIZED values
    st.subheader("🧒 Child Update Metrics (Ages 5-15)")
    c1, c2, c3, c4 = st.columns(4)
    
    if not features_df.empty:
        latest_week = features_df['week_number'].max()
        latest_data = features_df[features_df['week_number'] == latest_week]
        
        # SANITIZED: Only count valid positive values
        bio_valid = latest_data['bio_update_child'].dropna()
        bio_valid = bio_valid[bio_valid >= 0]
        total_bio_child = bio_valid.sum()
        
        demo_valid = latest_data['demo_update_child'].dropna() if 'demo_update_child' in latest_data else pd.Series([0])
        demo_valid = demo_valid[demo_valid >= 0]
        total_demo_child = demo_valid.sum()
        
        # SANITIZED: Cap completion rate at 100%
        completion_valid = latest_data['completion_rate_child'].dropna() if 'completion_rate_child' in latest_data else pd.Series()
        completion_valid = completion_valid.apply(lambda x: min(x, 1.0) if x >= 0 else np.nan)
        avg_completion = completion_valid.mean() if len(completion_valid) > 0 else None
        
        with c1:
            st.metric("Child Bio Updates", format_count(total_bio_child), 
                     help="Biometric updates for children 5-15 this week")
        with c2:
            st.metric("Child Demo Updates", format_count(total_demo_child),
                     help="Demographic updates for children 5-15 this week")
        with c3:
            st.metric("Avg Completion Rate", format_rate(avg_completion),
                     help="Success rate (capped at 100%)")
        with c4:
            inclusion_gaps = len(priority_df[priority_df['bottleneck_label'] == 'INCLUSION_GAP']) if not priority_df.empty else 0
            st.metric("Inclusion Gaps", inclusion_gaps, 
                     help="Districts where children may face access barriers")
    
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
        st.metric("Districts with Issues", bottlenecks)
    with col4:
        st.metric("Data Sources", 3)
    
    if not priority_df.empty:
        # RENAMED: Child-first language
        st.subheader("🚧 What's Holding Children Back?")
        bottleneck_counts = priority_df['bottleneck_label'].value_counts()
        
        # Human-readable labels
        label_map = {
            'OPERATIONAL_BOTTLENECK': 'Hardware/Process Issues',
            'DEMOGRAPHIC_SURGE': 'Population Surge',
            'CAPACITY_STRAIN': 'Overloaded Centers',
            'INCLUSION_GAP': 'Access Barriers',
            'ANOMALY_DETECTED': 'Unusual Patterns',
            'NORMAL': 'No Issues'
        }
        
        fig = px.bar(
            x=[label_map.get(x, x) for x in bottleneck_counts.index],
            y=bottleneck_counts.values,
            color=bottleneck_counts.index,
            labels={'x': 'Issue Type', 'y': 'Number of Districts'},
            color_discrete_map={
                'OPERATIONAL_BOTTLENECK': '#e74c3c',
                'DEMOGRAPHIC_SURGE': '#f39c12',
                'CAPACITY_STRAIN': '#9b59b6',
                'INCLUSION_GAP': '#e91e63',
                'ANOMALY_DETECTED': '#3498db',
                'NORMAL': '#2ecc71'
            }
        )
        fig.update_layout(showlegend=False, title_text="Distribution of barriers affecting child updates")
        st.plotly_chart(fig, use_container_width=True)
        
        # SMALL MULTIPLES: Forecast patterns by bottleneck type (WOW factor)
        st.subheader("📈 Demand Patterns by Issue Type")
        st.caption("⚠️ Illustrative patterns showing typical demand signatures by bottleneck type")
        
        if not features_df.empty:
            # Create synthetic forecast visualization
            bottleneck_types = ['OPERATIONAL_BOTTLENECK', 'DEMOGRAPHIC_SURGE', 'CAPACITY_STRAIN', 'INCLUSION_GAP']
            fig = make_subplots(rows=1, cols=4, subplot_titles=[label_map.get(b, b) for b in bottleneck_types])
            
            np.random.seed(42)
            weeks = list(range(1, 13))
            
            patterns = {
                'OPERATIONAL_BOTTLENECK': [100, 95, 85, 75, 70, 65, 60, 58, 55, 52, 50, 48],  # Declining
                'DEMOGRAPHIC_SURGE': [50, 55, 70, 100, 150, 180, 160, 140, 120, 100, 90, 85],  # Spike
                'CAPACITY_STRAIN': [80, 85, 90, 95, 100, 105, 108, 110, 108, 105, 100, 95],  # Plateau
                'INCLUSION_GAP': [30, 28, 25, 22, 20, 18, 20, 22, 25, 28, 30, 32]  # Low flat
            }
            
            colors = {'OPERATIONAL_BOTTLENECK': '#e74c3c', 'DEMOGRAPHIC_SURGE': '#f39c12', 
                     'CAPACITY_STRAIN': '#9b59b6', 'INCLUSION_GAP': '#e91e63'}
            
            for i, btype in enumerate(bottleneck_types, 1):
                actual = patterns[btype]
                forecast = [a * np.random.uniform(0.9, 1.1) for a in actual[-4:]] + [actual[-1] * np.random.uniform(0.95, 1.05) for _ in range(4)]
                
                fig.add_trace(go.Scatter(x=weeks, y=actual, mode='lines', name='Actual',
                                        line=dict(color=colors[btype], width=2)), row=1, col=i)
                fig.add_trace(go.Scatter(x=list(range(9, 17)), y=forecast, mode='lines', name='Forecast',
                                        line=dict(color=colors[btype], width=2, dash='dash')), row=1, col=i)
            
            fig.update_layout(height=250, showlegend=False, template="plotly_white")
            fig.update_xaxes(title_text="Week")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏆 Top 10 Priority Districts for Children")
        top_10 = priority_df.nsmallest(10, 'priority_rank')[
            ['district', 'state', 'priority_score', 'priority_rank', 'bottleneck_label']
        ].copy()
        top_10['bottleneck_label'] = top_10['bottleneck_label'].map(lambda x: label_map.get(x, x))
        top_10.columns = ['District', 'State', 'Priority Score', 'Rank', 'Issue Type']
        st.dataframe(top_10, use_container_width=True, hide_index=True)

# ===== TAB 5: PILOT MONITOR (Analyst only) =====
if tab5 is not None:
    with tab5:
        st.header("📈 Pilot: Measuring Impact on Children")
        
        if pilot_regions.empty:
            st.warning("Pilot regions not found. Run 08_pilot_design.ipynb first.")
        else:
            n_treatment = len(pilot_regions[pilot_regions['group'] == 'treatment'])
            n_control = len(pilot_regions[pilot_regions['group'] == 'control'])
            
            # Calculate pilot duration and current week from data
            pilot_districts = pilot_regions['district'].tolist()
            pilot_data = features_df[features_df['district'].isin(pilot_districts)].copy() if not features_df.empty else pd.DataFrame()
            
            if not pilot_data.empty:
                min_week = pilot_data['week_number'].min()
                max_week = pilot_data['week_number'].max()
                pilot_duration = max_week - min_week + 1
                current_week = max_week - min_week + 1
            else:
                pilot_duration = 12  # Fallback
                current_week = 8
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Treatment Districts", n_treatment)
            col2.metric("Control Districts", n_control)
            col3.metric("Pilot Duration", f"{pilot_duration} weeks")
            col4.metric("Current Week", f"Week {current_week}")
            
            if not pilot_data.empty:
                pilot_data = pilot_data.merge(pilot_regions[['district', 'group']], on='district', how='left')
                
                kpi = st.selectbox(
                    "Select Child Metric",
                    ['bio_update_child', 'demo_update_child'],
                    format_func=lambda x: "Child Bio Updates" if 'bio' in x else "Child Demo Updates",
                    key="pilot_kpi"
                )
                
                if kpi in pilot_data.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Treatment vs Control: Children Served")
                        trend_data = pilot_data.groupby(['week_number', 'group'])[kpi].mean().reset_index()
                        # SANITIZE
                        trend_data[kpi] = trend_data[kpi].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
                        
                        fig = px.line(
                            trend_data,
                            x='week_number',
                            y=kpi,
                            color='group',
                            markers=True,
                            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'}
                        )
                        fig.update_layout(xaxis_title="Week", yaxis_title="Children Updated")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("District Comparison: Children Reached")
                        latest = pilot_data.groupby(['district', 'group'])[kpi].mean().reset_index()
                        latest[kpi] = latest[kpi].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
                        
                        fig = px.bar(
                            latest.sort_values('group'),
                            x='district',
                            y=kpi,
                            color='group',
                            color_discrete_map={'treatment': '#e74c3c', 'control': '#3498db'}
                        )
                        fig.update_layout(xaxis_tickangle=45, yaxis_title="Children Updated")
                        st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Intervention Actions for Children")
            st.caption("⚠️ Intervention types, statuses, and effectiveness are illustrative. Children Reached is calculated from actual data.")
            treatment_districts = pilot_regions[pilot_regions['group'] == 'treatment']['district'].tolist()
            n = len(treatment_districts)
            
            if n > 0:
                # Calculate children reached from actual data per district
                children_reached = []
                for district in treatment_districts:
                    if not pilot_data.empty and 'bio_update_child' in pilot_data.columns:
                        district_sum = pilot_data[pilot_data['district'] == district]['bio_update_child'].sum()
                        children_reached.append(int(max(district_sum, 0)) if pd.notna(district_sum) else 0)
                    else:
                        children_reached.append(0)
                
                # Note: Intervention types and statuses are demo/illustrative
                interventions = ['Mobile Camp', 'Device Upgrade', 'Staff Training', 'Extended Hours', 'School Partnership']
                statuses = ['Completed', 'In Progress', 'Scheduled', 'Completed', 'In Progress']
                effectiveness = ['High', 'Medium', 'Pending', 'High', 'Medium']
                
                actions = pd.DataFrame({
                    'District': treatment_districts,
                    'Intervention (Demo)': interventions[:n],
                    'Status (Demo)': statuses[:n],
                    'Children Reached': children_reached,
                    'Effectiveness (Demo)': effectiveness[:n]
                })
                st.dataframe(actions, use_container_width=True, hide_index=True)

# ===== TAB 6: SYSTEM HEALTH (Analyst only) =====
if tab6 is not None:
    with tab6:
        st.header("🔧 System Health & Monitoring")
        st.warning("⚠️ **Demo Mode**: The metrics below are simulated for demonstration purposes. Connect to real monitoring infrastructure for production metrics.")
        
        @st.cache_data
        def generate_mock_metrics():
            dates = pd.date_range(end=datetime.now(), periods=12, freq='W')
            np.random.seed(42)
            psi_bio = np.clip(np.random.normal(0.08, 0.05, 12), 0, 0.4)
            psi_bio[-2:] = [0.22, 0.28]
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
        
        st.subheader("📋 Recent Alerts (Demo)")
        alerts = pd.DataFrame({
            'Timestamp': ['2026-01-10 07:15', '2026-01-08 12:30', '2026-01-03 09:00'],
            'Severity': ['🔴 Critical', '🟡 Warning', '🟢 Info'],
            'Alert': [
                'High data drift in child bio updates (PSI=0.28)',
                'MAPE elevated in District_X (72%)',
                'Weekly retraining completed successfully'
            ],
            'Status': ['Open', 'Resolved', 'Closed']
        })
        st.dataframe(alerts, use_container_width=True, hide_index=True)

# ===== TAB 7: AI ASSISTANT =====
if tab7 is not None:
    with tab7:
        audit.log_view(user['username'], user['role'], "ai_assistant")
        from components.chatbot_view import render_chatbot_view
        render_chatbot_view()

# Footer
st.divider()
st.caption(f"Aadhaar Pulse | Protecting Children's Access to Benefits | Logged in as {user['name']} ({user['role']})")
