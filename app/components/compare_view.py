"""
compare_view.py
District Comparison View - Side-by-side comparison of multiple districts.
Enables analysts to compare metrics, trends, and bottleneck patterns across districts.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
sys.path.insert(0, '..')

from utils.data_loader import mask_small_values, format_bottleneck_label, BOTTLENECK_LABELS


def get_comparison_metrics(priority_df: pd.DataFrame, labels_df: pd.DataFrame, 
                           features_df: pd.DataFrame, districts: List[str]) -> pd.DataFrame:
    """Extract comparison metrics for selected districts."""
    comparison_data = []
    
    for district in districts:
        # Get priority data
        priority_row = priority_df[priority_df['district'] == district]
        if priority_row.empty:
            continue
            
        state = priority_row['state'].values[0]
        
        # Get labels data (latest)
        labels_row = labels_df[labels_df['district'] == district].sort_values(
            ['year', 'week_number'], ascending=False
        ).head(1)
        
        # Get features data (latest)
        features_row = features_df[features_df['district'] == district].sort_values(
            ['year', 'week_number'], ascending=False
        ).head(1)
        
        metrics = {
            'District': district,
            'State': state,
            'Priority Score': priority_row['priority_score'].values[0] if not priority_row.empty else 0,
            'Priority Rank': int(priority_row['priority_rank'].values[0]) if not priority_row.empty and not pd.isna(priority_row['priority_rank'].values[0]) else 999,
            'Bottleneck Type': priority_row['bottleneck_label'].values[0] if not priority_row.empty else 'UNKNOWN',
            'Forecasted Demand (4w)': priority_row['forecasted_demand_next_4w'].values[0] if not priority_row.empty else 0,
            'Bio Updates': labels_row['current_bio_updates'].values[0] if not labels_row.empty else 0,
            'Demo Updates': labels_row['current_demo_updates'].values[0] if not labels_row.empty else 0,
            'Update Backlog': labels_row['update_backlog'].values[0] if not labels_row.empty else 0,
            'Completion Rate': labels_row['completion_rate'].values[0] if not labels_row.empty else 0,
        }
        
        # Add trend data from features
        if not features_row.empty:
            metrics['Rolling 4W Mean'] = features_row.get('rolling_4w_mean_bio_update_child', pd.Series([0])).values[0]
            metrics['Week-over-Week Change'] = features_row.get('wow_change_bio_update_child', pd.Series([0])).values[0]
        else:
            metrics['Rolling 4W Mean'] = 0
            metrics['Week-over-Week Change'] = 0
            
        comparison_data.append(metrics)
    
    return pd.DataFrame(comparison_data)


def create_radar_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Create radar chart comparing normalized metrics across districts."""
    if comparison_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Metrics to compare with their display names
    metrics_config = {
        'Priority Score': {'label': 'Priority', 'higher_is_worse': True},
        'Completion Rate': {'label': 'Completion', 'higher_is_worse': False},
        'Update Backlog': {'label': 'Backlog', 'higher_is_worse': True},
        'Forecasted Demand (4w)': {'label': 'Demand', 'higher_is_worse': True}
    }
    
    metrics_to_compare = list(metrics_config.keys())
    metric_labels = [metrics_config[m]['label'] for m in metrics_to_compare]
    
    # Calculate min/max for proper normalization
    ranges = {}
    for metric in metrics_to_compare:
        col_data = comparison_df[metric].dropna()
        col_data = col_data[col_data >= 0]  # Remove negative values
        if len(col_data) > 0:
            ranges[metric] = {'min': col_data.min(), 'max': col_data.max()}
        else:
            ranges[metric] = {'min': 0, 'max': 1}
    
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid[:len(comparison_df)]
    
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        values = []
        for metric in metrics_to_compare:
            val = row.get(metric, 0)
            if pd.isna(val) or val < 0:
                val = 0
            
            # Normalize to 0-100 scale using min-max
            r = ranges[metric]
            if r['max'] > r['min']:
                normalized = ((val - r['min']) / (r['max'] - r['min'])) * 100
            else:
                normalized = 50  # Default to middle if no range
            values.append(max(0, min(100, normalized)))
        
        # Close the radar chart
        values.append(values[0])
        categories = metric_labels + [metric_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['District'],
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                ticksuffix='%',
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)'
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        title=dict(text="District Metrics Comparison", x=0.5),
        height=450,
        template="plotly_white"
    )
    
    return fig


def create_bar_comparison(comparison_df: pd.DataFrame, metric: str) -> go.Figure:
    """Create bar chart comparing a specific metric across districts."""
    if comparison_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Handle negative or suppressed values
    df = comparison_df.copy()
    df[metric] = df[metric].apply(lambda x: max(0, x) if pd.notna(x) else 0)
    
    colors = px.colors.qualitative.Set2[:len(df)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['District'],
            y=df[metric],
            marker_color=colors,
            text=df[metric].apply(lambda x: f"{x:,.0f}" if x >= 10 else f"{x:.2f}"),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Comparison: {metric}",
        xaxis_title="District",
        yaxis_title=metric,
        height=350
    )
    
    return fig


def create_trend_comparison(features_df: pd.DataFrame, districts: List[str]) -> go.Figure:
    """Create overlaid time-series comparison for selected districts."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2[:len(districts)]
    
    for idx, district in enumerate(districts):
        df = features_df[features_df['district'] == district].sort_values(['year', 'week_number'])
        
        if df.empty:
            continue
        
        # Create a proper time index
        df = df.copy()
        df['time_index'] = df['year'].astype(str) + '-W' + df['week_number'].astype(str).str.zfill(2)
        
        # Bio updates
        fig.add_trace(go.Scatter(
            x=df['time_index'],
            y=df['bio_update_child'].apply(lambda x: max(0, x) if pd.notna(x) else 0),
            mode='lines+markers',
            name=f"{district} (Bio)",
            line=dict(color=colors[idx % len(colors)]),
            legendgroup=district
        ))
    
    fig.update_layout(
        title="Child Bio Update Trends Comparison",
        xaxis_title="Week",
        yaxis_title="Bio Updates (Child)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_bottleneck_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create a visual comparison of bottleneck types."""
    if comparison_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    bottleneck_colors = {
        'OPERATIONAL_BOTTLENECK': '#e74c3c',
        'DEMOGRAPHIC_SURGE': '#f39c12',
        'CAPACITY_STRAIN': '#9b59b6',
        'INCLUSION_GAP': '#e91e63',
        'ANOMALY_DETECTED': '#3498db',
        'NORMAL': '#2ecc71',
        'UNKNOWN': '#95a5a6'
    }
    
    df = comparison_df.copy()
    # Use human-readable labels
    df['Bottleneck Type Display'] = df['Bottleneck Type'].apply(format_bottleneck_label)
    df['Color'] = df['Bottleneck Type'].map(bottleneck_colors).fillna('#95a5a6')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['District'],
            y=[1] * len(df),  # All same height
            marker_color=df['Color'],
            text=df['Bottleneck Type Display'],
            textposition='inside',
            hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Bottleneck Type Comparison",
        xaxis_title="District",
        yaxis=dict(visible=False),
        height=200,
        showlegend=False
    )
    
    return fig


def calculate_peer_similarity(priority_df: pd.DataFrame, labels_df: pd.DataFrame,
                              target_district: str, n_peers: int = 5) -> List[str]:
    """Find similar districts based on key metrics for peer benchmarking."""
    target_row = priority_df[priority_df['district'] == target_district]
    if target_row.empty:
        return []
    
    target_score = target_row['priority_score'].values[0]
    target_bottleneck = target_row['bottleneck_label'].values[0]
    
    # Filter to same bottleneck type for meaningful comparison
    similar = priority_df[
        (priority_df['bottleneck_label'] == target_bottleneck) & 
        (priority_df['district'] != target_district)
    ].copy()
    
    if similar.empty:
        # Fall back to all districts
        similar = priority_df[priority_df['district'] != target_district].copy()
    
    # Calculate distance based on priority score
    similar['distance'] = abs(similar['priority_score'] - target_score)
    similar = similar.nsmallest(n_peers, 'distance')
    
    return similar['district'].tolist()


def classify_trend(features_df: pd.DataFrame, district: str) -> str:
    """Classify district trend as Improving, Declining, or Stable."""
    df = features_df[features_df['district'] == district].sort_values(['year', 'week_number'])
    
    if len(df) < 4:
        return "Insufficient Data"
    
    # Get last 4 weeks
    recent = df.tail(4)
    bio_updates = recent['bio_update_child'].values
    
    # Filter out negative values (suppressed)
    bio_updates = [v for v in bio_updates if v > 0]
    
    if len(bio_updates) < 2:
        return "Insufficient Data"
    
    # Calculate trend using simple linear regression
    x = np.arange(len(bio_updates))
    slope = np.polyfit(x, bio_updates, 1)[0]
    
    # Determine trend category
    mean_value = np.mean(bio_updates)
    relative_slope = slope / max(mean_value, 1) * 100  # Percentage change per week
    
    if relative_slope > 5:
        return "📈 Improving"
    elif relative_slope < -5:
        return "📉 Declining"
    else:
        return "➡️ Stable"


def render_compare_view(priority_df: pd.DataFrame, labels_df: pd.DataFrame, 
                        features_df: pd.DataFrame):
    """Render the complete district comparison view."""
    st.header("📊 District Comparison")
    st.caption("Compare metrics, trends, and bottleneck patterns across multiple districts")
    
    # District selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        all_districts = sorted(priority_df['district'].unique().tolist())
        selected_districts = st.multiselect(
            "Select Districts to Compare (2-5 recommended)",
            options=all_districts,
            default=all_districts[:3] if len(all_districts) >= 3 else all_districts,
            max_selections=5,
            help="Select 2-5 districts for meaningful comparison"
        )
    
    with col2:
        if len(selected_districts) == 1:
            st.info("💡 **Find Similar Districts**")
            if st.button("Find Peers"):
                peers = calculate_peer_similarity(priority_df, labels_df, selected_districts[0])
                st.session_state['suggested_peers'] = peers
        
        if 'suggested_peers' in st.session_state and st.session_state['suggested_peers']:
            st.write("Suggested peers:")
            for peer in st.session_state['suggested_peers'][:3]:
                st.write(f"• {peer}")
    
    if len(selected_districts) < 2:
        st.warning("Please select at least 2 districts to compare.")
        return
    
    # Get comparison data
    comparison_df = get_comparison_metrics(priority_df, labels_df, features_df, selected_districts)
    
    if comparison_df.empty:
        st.error("No data available for selected districts.")
        return
    
    # Add trend classification
    comparison_df['Trend'] = comparison_df['District'].apply(
        lambda d: classify_trend(features_df, d)
    )
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    st.subheader("📋 Side-by-Side Metrics Comparison")
    
    # Format the display dataframe
    display_df = comparison_df.copy()
    display_df['Priority Score'] = display_df['Priority Score'].apply(lambda x: f"{x:.3f}")
    display_df['Forecasted Demand (4w)'] = display_df['Forecasted Demand (4w)'].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) and x > 0 else "N/A"
    )
    display_df['Bio Updates'] = display_df['Bio Updates'].apply(
        lambda x: mask_small_values(x)
    )
    display_df['Demo Updates'] = display_df['Demo Updates'].apply(
        lambda x: mask_small_values(x)
    )
    display_df['Update Backlog'] = display_df['Update Backlog'].apply(
        lambda x: mask_small_values(abs(x)) if pd.notna(x) else "N/A"
    )
    display_df['Completion Rate'] = display_df['Completion Rate'].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "N/A"
    )
    display_df['Bottleneck Type'] = display_df['Bottleneck Type'].apply(format_bottleneck_label)
    
    # Transpose for better readability
    display_cols = ['District', 'State', 'Priority Rank', 'Priority Score', 'Bottleneck Type', 
                   'Trend', 'Bio Updates', 'Demo Updates', 'Update Backlog', 'Completion Rate',
                   'Forecasted Demand (4w)']
    
    st.dataframe(
        display_df[display_cols].set_index('District').T,
        use_container_width=True
    )
    
    # =========================================================================
    # VISUAL COMPARISONS
    # =========================================================================
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        st.subheader("🎯 Metrics Radar")
        fig_radar = create_radar_chart(comparison_df)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Bottleneck comparison
        st.subheader("🏷️ Bottleneck Types")
        fig_bottleneck = create_bottleneck_comparison(comparison_df)
        st.plotly_chart(fig_bottleneck, use_container_width=True)
        
        # Metric selector for bar chart
        metric_options = ['Priority Score', 'Bio Updates', 'Demo Updates', 'Update Backlog', 'Forecasted Demand (4w)']
        selected_metric = st.selectbox("Compare Metric:", metric_options, index=0)
        fig_bar = create_bar_comparison(comparison_df, selected_metric)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # =========================================================================
    # TREND COMPARISON
    # =========================================================================
    st.divider()
    st.subheader("📈 Historical Trend Comparison")
    
    fig_trend = create_trend_comparison(features_df, selected_districts)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # =========================================================================
    # INSIGHTS SUMMARY
    # =========================================================================
    st.divider()
    st.subheader("💡 Comparison Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Best/worst performers
        if not comparison_df.empty:
            best_idx = comparison_df['Priority Rank'].idxmin()
            worst_idx = comparison_df['Priority Rank'].idxmax()
            best_district = comparison_df.loc[best_idx, 'District']
            worst_district = comparison_df.loc[worst_idx, 'District']
            best_rank = comparison_df.loc[best_idx, 'Priority Rank']
            worst_rank = comparison_df.loc[worst_idx, 'Priority Rank']
            
            st.markdown("### 🏆 Performance Summary")
            st.success(f"**Best Performing:** {best_district} (Rank #{best_rank})")
            st.warning(f"**Needs Attention:** {worst_district} (Rank #{worst_rank})")
            
            # Trend summary
            trends = comparison_df['Trend'].value_counts()
            improving = trends.get('📈 Improving', 0)
            declining = trends.get('📉 Declining', 0)
            stable = trends.get('➡️ Stable', 0)
            
            if improving > declining:
                st.info(f"📈 **Trend Analysis:** {improving} of {len(comparison_df)} districts are improving")
            elif declining > improving:
                st.error(f"📉 **Trend Analysis:** {declining} of {len(comparison_df)} districts are declining")
            else:
                st.info(f"➡️ **Trend Analysis:** Districts show mixed trends")
    
    with insights_col2:
        if not comparison_df.empty:
            st.markdown("### 📊 Key Metrics")
            
            # Show range of key metrics
            avg_score = comparison_df['Priority Score'].mean()
            avg_backlog = comparison_df['Update Backlog'].mean()
            
            st.metric("Avg Priority Score", f"{avg_score:.3f}")
            st.metric("Avg Update Backlog", f"{abs(avg_backlog):,.0f}")
            
            # Bottleneck summary  
            bottleneck_counts = comparison_df['Bottleneck Type'].value_counts()
            most_common = bottleneck_counts.idxmax() if not bottleneck_counts.empty else "N/A"
            st.markdown(f"**Most Common Issue:** {format_bottleneck_label(most_common)}")
    
    # =========================================================================
    # EXPORT OPTIONS
    # =========================================================================
    st.divider()
    st.subheader("📤 Export Comparison")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # CSV export
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="district_comparison.csv",
            mime="text/csv"
        )
    
    with export_col2:
        # Summary text
        summary_text = f"""
DISTRICT COMPARISON REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Districts Compared: {', '.join(selected_districts)}

SUMMARY:
{comparison_df[['District', 'Priority Rank', 'Bottleneck Type', 'Trend']].to_string(index=False)}

DETAILED METRICS:
{comparison_df.to_string(index=False)}
        """
        st.download_button(
            label="📄 Download Summary",
            data=summary_text,
            file_name="district_comparison_summary.txt",
            mime="text/plain"
        )
