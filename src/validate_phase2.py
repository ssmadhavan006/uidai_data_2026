"""
validate_phase2.py
Validation script with time-series plots for sample districts.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

print("=" * 60)
print("PHASE 2 VALIDATION: Time-Series Plots")
print("=" * 60)

# Load data
df = pd.read_parquet('data/processed/model_features.parquet')
print(f"Loaded {len(df):,} rows")

# Get top 5 districts by enroll_child volume
top_districts = df.groupby(['state', 'district'])['enroll_child'].sum().nlargest(5).reset_index()
print(f"\nTop 5 districts by enrollment:")
print(top_districts)

# Create time-series plot
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Child Enrollments Over Time', 'Bio Updates vs Demo Updates'),
                    vertical_spacing=0.15)

colors = px.colors.qualitative.Set2

for i, (_, row) in enumerate(top_districts.iterrows()):
    state, district = row['state'], row['district']
    subset = df[(df['state'] == state) & (df['district'] == district)].sort_values('week_number')
    
    # Enrollment
    fig.add_trace(
        go.Scatter(x=subset['week_number'], y=subset['enroll_child'],
                   mode='lines+markers', name=f"{district[:15]} (Enroll)",
                   line=dict(color=colors[i % len(colors)])),
        row=1, col=1
    )
    
    # Bio updates
    fig.add_trace(
        go.Scatter(x=subset['week_number'], y=subset['bio_update_child'],
                   mode='lines', name=f"{district[:15]} (Bio)",
                   line=dict(color=colors[i % len(colors)], dash='dash')),
        row=2, col=1
    )

fig.update_layout(
    height=700, 
    title_text='Phase 2 Validation: Sample District Time-Series',
    showlegend=True
)
fig.update_xaxes(title_text='Week Number', row=2, col=1)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=2, col=1)

# Save plot
Path('outputs').mkdir(exist_ok=True)
fig.write_html('outputs/validation_timeseries.html')
print(f"\nSaved plot to outputs/validation_timeseries.html")

# Also show basic stats
print("\n" + "=" * 40)
print("VALIDATION STATISTICS")
print("=" * 40)
print(f"Total rows: {len(df):,}")
print(f"Unique districts: {df['district'].nunique()}")
print(f"Week range: {df['week_number'].min()} to {df['week_number'].max()}")
print(f"Suppressed cells (value=-1): {(df == -1).sum().sum()}")

# Check for privacy compliance
count_cols = ['enroll_child', 'demo_update_child', 'bio_update_child']
violations = 0
for col in count_cols:
    v = ((df[col] > 0) & (df[col] < 10)).sum()
    violations += v
    if v > 0:
        print(f"WARNING: {col} has {v} values between 0 and 10")

if violations == 0:
    print("\n✅ Privacy check PASSED: No counts between 0 and 10")
else:
    print(f"\n❌ Privacy check FAILED: {violations} violations found")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
