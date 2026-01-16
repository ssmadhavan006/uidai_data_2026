"""
map_view.py
Enhanced choropleth map visualization for district priorities.
Features: Heatmap effect, better zoom controls, visual intensity indicators.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '..')

from utils.data_loader import format_bottleneck_label, BOTTLENECK_LABELS


def create_priority_map(priority_df: pd.DataFrame, map_style: str = 'heatmap') -> go.Figure:
    """
    Create an interactive map of district priorities with enhanced visualization.
    
    Args:
        priority_df: DataFrame with priority scores
        map_style: 'heatmap' for density view, 'scatter' for individual points
    """
    df = priority_df.copy()
    
    # Approximate coordinates for Indian states (centroid approximations)
    state_coords = {
        'Andhra Pradesh': (15.9129, 79.7400),
        'Arunachal Pradesh': (27.0844, 93.6053),
        'Assam': (26.2006, 92.9376),
        'Bihar': (25.0961, 85.3131),
        'Chhattisgarh': (21.2787, 81.8661),
        'Goa': (15.2993, 74.1240),
        'Gujarat': (22.2587, 71.1924),
        'Haryana': (29.0588, 76.0856),
        'Himachal Pradesh': (31.1048, 77.1734),
        'Jharkhand': (23.6102, 85.2799),
        'Karnataka': (15.3173, 75.7139),
        'Kerala': (10.8505, 76.2711),
        'Madhya Pradesh': (22.9734, 78.6569),
        'Maharashtra': (19.7515, 75.7139),
        'Manipur': (24.6637, 93.9063),
        'Meghalaya': (25.4670, 91.3662),
        'Mizoram': (23.1645, 92.9376),
        'Nagaland': (26.1584, 94.5624),
        'Odisha': (20.9517, 85.0985),
        'Punjab': (31.1471, 75.3412),
        'Rajasthan': (27.0238, 74.2179),
        'Sikkim': (27.5330, 88.5122),
        'Tamil Nadu': (11.1271, 78.6569),
        'Telangana': (18.1124, 79.0193),
        'Tripura': (23.9408, 91.9882),
        'Uttar Pradesh': (26.8467, 80.9462),
        'Uttarakhand': (30.0668, 79.0193),
        'West Bengal': (22.9868, 87.8550),
        'Delhi': (28.7041, 77.1025),
        'Jammu and Kashmir': (33.7782, 76.5762),
        'Ladakh': (34.1526, 77.5771),
    }
    
    # Add coordinates with smarter jitter based on district count per state
    df['lat'] = df['state'].map(lambda x: state_coords.get(x, (22.5, 82.5))[0])
    df['lon'] = df['state'].map(lambda x: state_coords.get(x, (22.5, 82.5))[1])
    
    # Apply circular jitter pattern for better distribution
    district_idx = df.groupby('state').cumcount()
    angle = district_idx * (2 * np.pi / 12)  # Spread in circle
    radius = (district_idx // 12 + 1) * 0.3
    df['lat'] = df['lat'] + np.sin(angle) * radius
    df['lon'] = df['lon'] + np.cos(angle) * radius
    
    # Clean data
    df['size_value'] = df['priority_score'].fillna(0.1).abs().clip(lower=0.1)
    df['intensity'] = (df['priority_score'].fillna(0) * 100).round(1)
    
    # Human-readable bottleneck labels
    df['bottleneck_display'] = df['bottleneck_label'].apply(format_bottleneck_label)
    
    if map_style == 'heatmap':
        # Create density mapbox for heatmap effect
        fig = go.Figure()
        
        # Add density layer
        fig.add_trace(go.Densitymapbox(
            lat=df['lat'],
            lon=df['lon'],
            z=df['priority_score'],
            radius=25,
            colorscale=[
                [0.0, 'rgba(0, 255, 0, 0.2)'],    # Green - low priority
                [0.3, 'rgba(255, 255, 0, 0.4)'],  # Yellow
                [0.5, 'rgba(255, 165, 0, 0.5)'],  # Orange
                [0.7, 'rgba(255, 69, 0, 0.6)'],   # Red-Orange
                [1.0, 'rgba(255, 0, 0, 0.8)']     # Red - high priority
            ],
            hoverinfo='skip',
            showscale=True,
            colorbar=dict(
                title=dict(text='Priority<br>Score', side='right'),
                tickformat='.2f',
                thickness=15,
                len=0.5,
                y=0.7
            )
        ))
        
        # Add scatter markers on top for interactivity
        fig.add_trace(go.Scattermapbox(
            lat=df['lat'],
            lon=df['lon'],
            mode='markers',
            marker=dict(
                size=df['size_value'] * 25,
                color=df['priority_score'],
                colorscale='Reds',
                opacity=0.7,
                showscale=False
            ),
            text=df['district'],
            customdata=np.stack([
                df['state'],
                df['intensity'],
                df['bottleneck_display'],
                df['priority_rank']
            ], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "State: %{customdata[0]}<br>"
                "Priority: %{customdata[1]}%<br>"
                "Issue: %{customdata[2]}<br>"
                "Rank: #%{customdata[3]}"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',
                center=dict(lat=22.5, lon=82.5),
                zoom=3.8
            ),
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=650,
            title=dict(
                text='🔥 Priority Hotspot Intensity Map',
                x=0.5,
                font=dict(size=18)
            )
        )
    else:
        # Standard scatter map
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color='priority_score',
            size='size_value',
            hover_name='district',
            hover_data={
                'state': True,
                'intensity': True,
                'bottleneck_display': True,
                'priority_rank': True,
                'lat': False,
                'lon': False,
                'size_value': False,
                'priority_score': False
            },
            color_continuous_scale='RdYlGn_r',
            size_max=25,
            zoom=3.8,
            center={'lat': 22.5, 'lon': 82.5},
            mapbox_style='carto-positron'
        )
        
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=650,
            title=dict(
                text='📍 District Priority Map',
                x=0.5,
                font=dict(size=18)
            )
        )
    
    return fig


def create_state_summary_chart(priority_df: pd.DataFrame) -> go.Figure:
    """Create state-level aggregated view."""
    state_agg = priority_df.groupby('state').agg({
        'priority_score': 'mean',
        'district': 'count',
        'bottleneck_label': lambda x: (x != 'NORMAL').sum()
    }).reset_index()
    state_agg.columns = ['State', 'Avg Priority', 'Districts', 'With Issues']
    state_agg = state_agg.sort_values('Avg Priority', ascending=False).head(15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=state_agg['State'],
        x=state_agg['Avg Priority'],
        orientation='h',
        marker=dict(
            color=state_agg['Avg Priority'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title='Priority')
        ),
        text=state_agg['With Issues'].apply(lambda x: f'{x} issues'),
        textposition='inside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Priority: %{x:.3f}<br>"
            "Districts: %{customdata[0]}<br>"
            "With Issues: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=np.stack([state_agg['Districts'], state_agg['With Issues']], axis=-1)
    ))
    
    fig.update_layout(
        title=dict(text='States by Average Priority Score', x=0.5),
        xaxis_title='Average Priority Score',
        yaxis=dict(categoryorder='total ascending'),
        height=450,
        template='plotly_white'
    )
    
    return fig


def create_priority_bar_chart(priority_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of top priority districts with better styling."""
    top = priority_df.nsmallest(top_n, 'priority_rank').copy()
    top['bottleneck_display'] = top['bottleneck_label'].apply(format_bottleneck_label)
    
    color_map = {
        'Hardware & Process Issues': '#e74c3c',
        'Population Surge': '#f39c12',
        'Overloaded Centers': '#9b59b6',
        'Access Barriers': '#e91e63',
        'Unusual Patterns': '#3498db',
        'No Issues': '#2ecc71',
        'Unknown': '#95a5a6'
    }
    
    fig = px.bar(
        top,
        y='district',
        x='priority_score',
        color='bottleneck_display',
        orientation='h',
        hover_data=['state', 'priority_rank'],
        title=f'Top {top_n} Priority Districts',
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=True,
        legend_title='Issue Type',
        template='plotly_white'
    )
    
    return fig


def create_bottleneck_donut(priority_df: pd.DataFrame) -> go.Figure:
    """Create donut chart for bottleneck distribution."""
    # Map to human-readable labels
    df = priority_df.copy()
    df['bottleneck_display'] = df['bottleneck_label'].apply(format_bottleneck_label)
    bottleneck_counts = df['bottleneck_display'].value_counts()
    
    colors = {
        'Hardware & Process Issues': '#e74c3c',
        'Population Surge': '#f39c12',
        'Overloaded Centers': '#9b59b6',
        'Access Barriers': '#e91e63',
        'Unusual Patterns': '#3498db',
        'No Issues': '#2ecc71',
        'Unknown': '#95a5a6'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=bottleneck_counts.index,
        values=bottleneck_counts.values,
        hole=0.5,
        marker_colors=[colors.get(x, '#95a5a6') for x in bottleneck_counts.index],
        textinfo='label+percent',
        textposition='outside',
        pull=[0.05 if c == bottleneck_counts.idxmax() else 0 for c in bottleneck_counts.index]
    )])
    
    fig.update_layout(
        title=dict(text='Issue Distribution', x=0.5),
        showlegend=False,
        height=400,
        annotations=[dict(text=f'{len(priority_df)}<br>Districts', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig


def render_map_view(priority_df: pd.DataFrame):
    """Render the enhanced map view with multiple visualization options."""
    st.header("📍 District Priority Hotspots")
    st.caption("Visualize where children need the most support across India")
    
    # View mode selector
    col_mode1, col_mode2 = st.columns([1, 3])
    with col_mode1:
        view_mode = st.radio(
            "Map Style",
            ["🔥 Heatmap", "📍 Markers"],
            horizontal=True,
            help="Heatmap shows intensity, Markers show individual districts"
        )
    
    # Filters
    st.subheader("🔍 Filters")
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        min_score = float(priority_df['priority_score'].min())
        max_score = float(priority_df['priority_score'].max())
        score_range = st.slider(
            "Priority Score Range",
            min_value=0.0, max_value=1.0,
            value=(min_score, max_score),
            step=0.05,
            format="%.2f"
        )
    
    with col_filter2:
        bottleneck_types = ['All Issues'] + [format_bottleneck_label(b) for b in priority_df['bottleneck_label'].unique()]
        selected_bottleneck = st.selectbox("Issue Type Filter", bottleneck_types)
    
    with col_filter3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Filters"):
            st.rerun()
    
    # Apply filters
    filtered_df = priority_df[
        (priority_df['priority_score'] >= score_range[0]) &
        (priority_df['priority_score'] <= score_range[1])
    ]
    if selected_bottleneck != 'All Issues':
        # Reverse lookup to original label
        reverse_map = {v: k for k, v in BOTTLENECK_LABELS.items()}
        original_label = reverse_map.get(selected_bottleneck, selected_bottleneck)
        filtered_df = filtered_df[filtered_df['bottleneck_label'] == original_label]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.metric("📊 Showing", f"{len(filtered_df)} districts")
    with col2:
        critical = len(filtered_df[filtered_df['priority_score'] > 0.7])
        st.metric("🔴 Critical", f"{critical}", help="Score > 0.7")
    with col3:
        moderate = len(filtered_df[(filtered_df['priority_score'] > 0.4) & (filtered_df['priority_score'] <= 0.7)])
        st.metric("🟡 Moderate", f"{moderate}", help="Score 0.4-0.7")
    with col4:
        normal = len(filtered_df[filtered_df['bottleneck_label'] == 'NORMAL'])
        st.metric("🟢 Healthy", f"{normal}", help="No issues detected")
    
    # Map
    st.divider()
    map_style = 'heatmap' if '🔥' in view_mode else 'scatter'
    
    with st.spinner("Rendering map..."):
        try:
            fig = create_priority_map(filtered_df, map_style=map_style)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 **Tip**: Use scroll to zoom, click+drag to pan. Hover over markers for details.")
        except Exception as e:
            st.error(f"Map rendering failed: {e}")
    
    # Bottom section: Charts
    st.divider()
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader("🏆 Top Priority Districts")
        fig_bar = create_priority_bar_chart(priority_df, top_n=12)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        st.subheader("📊 Issue Breakdown")
        fig_donut = create_bottleneck_donut(priority_df)
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # State summary
    with st.expander("📈 State-Level Summary", expanded=False):
        fig_state = create_state_summary_chart(priority_df)
        st.plotly_chart(fig_state, use_container_width=True)
