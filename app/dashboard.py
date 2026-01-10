"""
Aadhaar Pulse: Child Update Intelligence Platform
Main Streamlit Dashboard with RBAC and Audit Logging
"""
import streamlit as st
import pandas as pd
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

# User is authenticated - render sidebar and get user info
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

# Header with role indicator
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

priority_df, labels_df, features_df = load_all_data()

# Apply masking for Viewer role
def mask_for_viewer(df):
    """Mask sensitive data for Viewer role."""
    if user['role'] == 'Viewer':
        masked = df.copy()
        # Mask exact priority scores
        if 'priority_score' in masked.columns:
            masked['priority_score'] = masked['priority_score'].apply(
                lambda x: round(x, 1)  # Round to 1 decimal
            )
        return masked
    return df

priority_df = mask_for_viewer(priority_df)

# Tabs - Viewer gets fewer tabs
if require_role("Analyst"):
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Hotspot Map", "🔍 District Analysis", "🎮 Policy Simulator", "📊 Overview"])
else:
    tab1, tab4 = st.tabs(["📍 Hotspot Map", "📊 Overview"])
    tab2 = tab3 = None

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
            
            # District selector
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # State filter
                states = sorted(priority_df['state'].unique().tolist())
                selected_state = st.selectbox("Select State", states)
                
                # District filter
                districts = priority_df[priority_df['state'] == selected_state]['district'].tolist()
                selected_district = st.selectbox("Select District", districts)
            
            with col2:
                # Quick stats
                district_row = priority_df[
                    (priority_df['district'] == selected_district) & 
                    (priority_df['state'] == selected_state)
                ]
                if not district_row.empty:
                    r = district_row.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    # Handle NaN values gracefully
                    rank = int(r['priority_rank']) if pd.notna(r['priority_rank']) else 'N/A'
                    score = f"{r['priority_score']:.2f}" if pd.notna(r['priority_score']) else 'N/A'
                    label = r['bottleneck_label'].replace('_', ' ') if pd.notna(r['bottleneck_label']) else 'Unknown'
                    c1.metric("Priority Rank", f"#{rank}" if isinstance(rank, int) else rank)
                    c2.metric("Priority Score", score)
                    c3.metric("Bottleneck", label)
            
            # Log the view
            audit.log_view(user['username'], user['role'], "district_analysis", selected_district)
            
            # Get district data
            from utils.data_loader import get_district_data
            district_data = get_district_data(selected_district, selected_state, priority_df, labels_df)
            
            if district_data:
                from components.why_view import render_why_view
                render_why_view(district_data, features_df)
            else:
                st.warning("No data available for selected district.")
        else:
            st.warning("Priority data not available.")

# ===== TAB 3: SIMULATOR (Analyst only) =====
if tab3 is not None:
    with tab3:
        if not priority_df.empty:
            st.header("🎮 Policy Simulator")
            
            # District selector in sidebar
            col1, col2 = st.columns([1, 3])
            
            with col1:
                sim_state = st.selectbox("State", sorted(priority_df['state'].unique().tolist()), key="sim_state")
                sim_districts = priority_df[priority_df['state'] == sim_state]['district'].tolist()
                sim_district = st.selectbox("District", sim_districts, key="sim_district")
            
            # Get district data
            from utils.data_loader import get_district_data
            sim_district_data = get_district_data(sim_district, sim_state, priority_df, labels_df)
            
            with col2:
                if sim_district_data:
                    from components.simulator_view import render_simulator_view
                    render_simulator_view(sim_district_data)
                else:
                    st.warning("No data available for simulation.")
            
            # Export button section
            st.divider()
            if st.session_state.get('last_sim_result'):
                result = st.session_state['last_sim_result']
                
                # Log the simulation
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
                        # Apply differential privacy to export
                        from dp_export import dp_export_dataframe
                        export_df = priority_df[['state', 'district', 'priority_score', 'bottleneck_label']].copy()
                        noisy_df = dp_export_dataframe(
                            export_df,
                            epsilon=1.0,
                            rate_columns=['priority_score']
                        )
                        # Save
                        export_path = os.path.join(root_dir, 'outputs/dp_export.csv')
                        noisy_df.to_csv(export_path, index=False)
                        audit.log_export(user['username'], user['role'], "dp_export.csv", len(noisy_df))
                        st.success(f"DP-protected export saved ({len(noisy_df)} rows, ε=1.0)")
        else:
            st.warning("Priority data not available.")

# ===== TAB 4: OVERVIEW =====
overview_tab = tab4 if require_role("Analyst") else tab4
with overview_tab:
    st.header("📊 System Overview")
    
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
    
    # Bottleneck summary
    if not priority_df.empty:
        st.subheader("Bottleneck Distribution")
        bottleneck_counts = priority_df['bottleneck_label'].value_counts()
        
        import plotly.express as px
        fig = px.bar(
            x=bottleneck_counts.index,
            y=bottleneck_counts.values,
            color=bottleneck_counts.index,
            labels={'x': 'Bottleneck Type', 'y': 'Count'},
            color_discrete_map={
                'OPERATIONAL_BOTTLENECK': '#e74c3c',
                'DEMOGRAPHIC_SURGE': '#f39c12',
                'CAPACITY_STRAIN': '#9b59b6',
                'ANOMALY_DETECTED': '#3498db',
                'NORMAL': '#2ecc71'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top priority table
        st.subheader("Top 10 Priority Districts")
        top_10 = priority_df.nsmallest(10, 'priority_rank')[
            ['district', 'state', 'priority_score', 'priority_rank', 'bottleneck_label']
        ]
        st.dataframe(top_10, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption(f"Aadhaar Pulse | Logged in as {user['name']} ({user['role']}) | Built for UIDAI Hackathon 2025")
