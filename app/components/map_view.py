"""
map_view.py
Choropleth map visualization for district priorities.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_priority_map(priority_df: pd.DataFrame, use_scatter: bool = True) -> go.Figure:
    """
    Create an interactive map of district priorities.
    
    Uses scatter_mapbox as fallback when GeoJSON is unavailable.
    """
    # Prepare data
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
    
    # Add approximate coordinates based on state
    df['lat'] = df['state'].map(lambda x: state_coords.get(x, (22.5, 82.5))[0])
    df['lon'] = df['state'].map(lambda x: state_coords.get(x, (22.5, 82.5))[1])
    
    # Add jitter for districts in same state
    df['lat'] = df['lat'] + (df.groupby('state').cumcount() * 0.15)
    df['lon'] = df['lon'] + (df.groupby('state').cumcount() % 5 * 0.15)
    
    # Fix size values: ensure positive and no NaN
    df['size_value'] = df['priority_score'].fillna(0.1).abs()
    df['size_value'] = df['size_value'].clip(lower=0.05)  # Minimum size
    
    # Create scatter map
    fig = px.scatter_mapbox(
        df,
        lat='lat',
        lon='lon',
        color='priority_score',
        size='size_value',  # Use cleaned size column
        hover_name='district',
        hover_data={
            'state': True,
            'priority_score': ':.2f',
            'bottleneck_label': True,
            'priority_rank': True,
            'lat': False,
            'lon': False,
            'size_value': False
        },
        color_continuous_scale='Reds',
        size_max=20,
        zoom=4,
        center={'lat': 22.5, 'lon': 82.5},
        mapbox_style='carto-positron'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=600,
        title='District Priority Hotspots'
    )
    
    return fig


def create_priority_bar_chart(priority_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of top priority districts."""
    top = priority_df.nsmallest(top_n, 'priority_rank')
    
    fig = px.bar(
        top,
        y='district',
        x='priority_score',
        color='bottleneck_label',
        orientation='h',
        hover_data=['state', 'priority_rank'],
        title=f'Top {top_n} Priority Districts',
        color_discrete_map={
            'OPERATIONAL_BOTTLENECK': '#e74c3c',
            'DEMOGRAPHIC_SURGE': '#f39c12',
            'CAPACITY_STRAIN': '#9b59b6',
            'ANOMALY_DETECTED': '#3498db',
            'NORMAL': '#2ecc71'
        }
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=True
    )
    
    return fig


def render_map_view(priority_df: pd.DataFrame):
    """Render the complete map view."""
    st.header("📍 District Priority Hotspots")
    
    # Time filter
    st.subheader("Filters")
    col_filter1, col_filter2 = st.columns([2, 2])
    with col_filter1:
        # Week slider (if week data available)
        min_score = float(priority_df['priority_score'].min())
        max_score = float(priority_df['priority_score'].max())
        score_range = st.slider(
            "Priority Score Range",
            min_value=0.0, max_value=1.0,
            value=(min_score, max_score),
            step=0.05
        )
    with col_filter2:
        # Bottleneck filter
        bottleneck_types = ['All'] + priority_df['bottleneck_label'].unique().tolist()
        selected_bottleneck = st.selectbox("Bottleneck Type", bottleneck_types)
    
    # Apply filters
    filtered_df = priority_df[
        (priority_df['priority_score'] >= score_range[0]) &
        (priority_df['priority_score'] <= score_range[1])
    ]
    if selected_bottleneck != 'All':
        filtered_df = filtered_df[filtered_df['bottleneck_label'] == selected_bottleneck]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Filtered Districts", len(filtered_df))
    with col2:
        critical = len(filtered_df[filtered_df['priority_score'] > 0.7])
        st.metric("Critical (Score > 0.7)", critical)
    with col3:
        bottleneck_count = len(filtered_df[filtered_df['bottleneck_label'] != 'NORMAL'])
        st.metric("With Bottlenecks", bottleneck_count)
    
    # Map
    st.subheader("Interactive Map")
    with st.spinner("Loading map..."):
        try:
            fig = create_priority_map(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Map loading failed: {e}")
            st.warning("Displaying bar chart instead. Check data quality.")
    
    # Bar chart
    st.subheader("Top Priority Districts")
    fig_bar = create_priority_bar_chart(priority_df)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Bottleneck distribution
    st.subheader("Bottleneck Distribution")
    bottleneck_counts = priority_df['bottleneck_label'].value_counts()
    fig_pie = px.pie(
        values=bottleneck_counts.values,
        names=bottleneck_counts.index,
        title='Bottleneck Types',
        color=bottleneck_counts.index,
        color_discrete_map={
            'OPERATIONAL_BOTTLENECK': '#e74c3c',
            'DEMOGRAPHIC_SURGE': '#f39c12',
            'CAPACITY_STRAIN': '#9b59b6',
            'ANOMALY_DETECTED': '#3498db',
            'NORMAL': '#2ecc71'
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)
