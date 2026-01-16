"""
simulator_view.py
Policy simulator interactive view.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.simulator import simulate_intervention, run_monte_carlo_simulation, load_interventions
except ImportError:
    from simulator import simulate_intervention, run_monte_carlo_simulation, load_interventions


def create_backlog_timeline_chart(initial_backlog: float, weekly_demand: float, 
                                   baseline_capacity: float, intervention_boost: float,
                                   duration: int) -> go.Figure:
    """Create projected backlog timeline chart."""
    weeks = list(range(duration + 1))
    
    # Scenario without intervention
    backlog_no_interv = [initial_backlog]
    for w in range(duration):
        new_backlog = backlog_no_interv[-1] + weekly_demand - baseline_capacity
        backlog_no_interv.append(max(0, new_backlog))
    
    # Scenario with intervention
    backlog_with_interv = [initial_backlog]
    for w in range(duration):
        new_backlog = backlog_with_interv[-1] + weekly_demand - (baseline_capacity + intervention_boost)
        backlog_with_interv.append(max(0, new_backlog))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=weeks, y=backlog_no_interv,
        mode='lines+markers',
        name='Without Intervention',
        line=dict(color='#e74c3c', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=weeks, y=backlog_with_interv,
        mode='lines+markers',
        name='With Intervention',
        line=dict(color='#2ecc71')
    ))
    
    # Add fill between
    fig.add_trace(go.Scatter(
        x=weeks + weeks[::-1],
        y=backlog_no_interv + backlog_with_interv[::-1],
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(width=0),
        name='Reduction'
    ))
    
    fig.update_layout(
        title='Projected Children Protected Over Time',
        xaxis_title='Week',
        yaxis_title='Pending Child Updates',
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig


def render_simulator_view(district_data: dict, interventions: dict = None):
    """Render the policy simulator view."""
    district = district_data.get('district', 'Unknown')
    state = district_data.get('state', 'Unknown')
    
    st.header(f"🎮 Policy Simulator: {district}")
    st.caption(f"State: {state}")
    
    # Load interventions if not provided
    if interventions is None:
        try:
            interventions = load_interventions('config/interventions.json')
        except FileNotFoundError:
            interventions = load_interventions('../config/interventions.json')
    
    # Sidebar-style configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Intervention selector with human-readable names
        def format_intervention_name(name):
            """Convert intervention name to human-readable format."""
            return name.replace('_', ' ').replace('-', ' ').title()
        
        intervention_name = st.selectbox(
            "Select Intervention",
            list(interventions.keys()),
            format_func=format_intervention_name
        )
        
        intervention = interventions[intervention_name]
        
        # Show intervention details
        st.markdown(f"**Description:** {intervention['description']}")
        st.markdown(f"**Cost:** ₹{intervention['cost']:,}")
        st.markdown(f"**Capacity:** {intervention['capacity_per_week']}/week")
        st.markdown(f"**Duration:** {intervention['duration_weeks']} weeks")
        
        # Scenario selector
        scenario = st.radio(
            "Effectiveness Scenario",
            ['conservative', 'median', 'optimistic'],
            index=1,
            horizontal=True
        )
        
        effectiveness = intervention['effectiveness'][scenario]
        st.info(f"Effectiveness: {effectiveness*100:.0f}%")
        
        # Budget slider
        st.divider()
        budget = st.slider(
            "Allocated Budget (₹)",
            min_value=0,
            max_value=500000,
            value=intervention['cost'],
            step=25000,
            format="₹%d"
        )
        
        # Scale intervention if budget differs
        budget_factor = min(budget / intervention['cost'], 1.0) if intervention['cost'] > 0 else 1.0
        st.caption(f"Budget factor: {budget_factor:.0%} of full cost")
        
        # Run simulation button
        run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Simulation Results")
        
        if run_sim:
            with st.spinner("Running Monte Carlo simulation (500 runs)..."):
                # Run simulation
                result = simulate_intervention(
                    district_data, intervention, intervention_name, scenario
                )
                
                # Run Monte Carlo
                mc_result = run_monte_carlo_simulation(
                    district_data, intervention, intervention_name,
                    n_runs=500,
                    forecast_lower=district_data.get('weekly_demand', 100) * 0.7,
                    forecast_upper=district_data.get('weekly_demand', 100) * 1.3
                )
                
                # Display metrics
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    reduction = mc_result['backlog_reduction']['p50']
                    uncertainty = (mc_result['backlog_reduction']['p95'] - mc_result['backlog_reduction']['p5']) / 2
                    st.metric("👦👧 Children Protected", f"{reduction:,.0f}", f"±{uncertainty:,.0f}")
                
                with m2:
                    pct = mc_result['reduction_pct']['p50']
                    st.metric("Workload Processed", f"{pct:.1f}%")
                
                with m3:
                    cost_per = mc_result['cost_per_update']['p50']
                    st.metric("Cost/Update", f"₹{cost_per:.0f}")
                
                with m4:
                    st.metric("Fairness Index", f"{result.fairness_index:.2f}")
                
                # Timeline chart
                intervention_boost = intervention['capacity_per_week'] * effectiveness
                fig = create_backlog_timeline_chart(
                    district_data.get('backlog', 1000),
                    district_data.get('weekly_demand', 200),
                    district_data.get('baseline_capacity', 150),
                    intervention_boost,
                    intervention['duration_weeks']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence interval display
                st.success(f"""
                **90% Confidence Interval (Children Protected):**
                - Children Reached: {mc_result['backlog_reduction']['p5']:,.0f} to {mc_result['backlog_reduction']['p95']:,.0f}
                - Workload Processed: {mc_result['reduction_pct']['p5']:.1f}% to {mc_result['reduction_pct']['p95']:.1f}%
                """)
                
                # Save result for PDF export
                st.session_state['last_sim_result'] = {
                    'district': district,
                    'state': state,
                    'intervention': intervention_name,
                    'intervention_config': intervention,
                    'mc_result': mc_result,
                    'scenario': scenario
                }
        else:
            st.info("👆 Configure intervention and click 'Run Simulation' to see results")
            
            # Show current district stats
            st.markdown("**Current District Status:**")
            st.markdown(f"- Estimated backlog: {district_data.get('backlog', 'N/A'):,}")
            st.markdown(f"- Weekly demand: {district_data.get('weekly_demand', 'N/A'):,.0f}")
            st.markdown(f"- Baseline capacity: {district_data.get('baseline_capacity', 'N/A'):,.0f}")
