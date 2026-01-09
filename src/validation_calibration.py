"""
validation_calibration.py
Uncertainty calibration - check prediction interval coverage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


def calculate_prediction_interval_coverage(
    forecasts_path: str = 'outputs/forecasts.csv',
    features_path: str = 'data/processed/model_features.parquet'
) -> dict:
    """
    Check if 90% prediction intervals achieve ~90% empirical coverage.
    """
    print("=" * 60)
    print("UNCERTAINTY CALIBRATION")
    print("=" * 60)
    
    # Load data
    try:
        forecasts = pd.read_csv(forecasts_path)
    except FileNotFoundError:
        print("Forecasts not found")
        return {'coverage': np.nan}
    
    features = pd.read_parquet(features_path)
    
    # Get latest actuals per district
    actuals = features.sort_values(['state', 'district', 'year', 'week_number']).groupby(
        ['state', 'district']
    ).tail(1)[['state', 'district', 'bio_update_child']]
    
    # Merge with forecasts (first forecast per district)
    forecast_first = forecasts.drop_duplicates(['state', 'district'], keep='first')
    merged = actuals.merge(forecast_first, on=['state', 'district'], how='inner')
    
    if merged.empty or 'yhat_lower' not in merged.columns:
        print("Cannot compute coverage - missing forecast bounds")
        return {'coverage': np.nan}
    
    # Check coverage
    merged['in_interval'] = (
        (merged['bio_update_child'] >= merged['yhat_lower']) &
        (merged['bio_update_child'] <= merged['yhat_upper'])
    )
    
    coverage = merged['in_interval'].mean()
    
    print(f"\nTotal districts with forecasts: {len(merged)}")
    print(f"Districts within 90% CI: {merged['in_interval'].sum()}")
    print(f"Empirical coverage: {coverage:.1%}")
    
    # Check if well-calibrated (85-95%)
    if 0.85 <= coverage <= 0.95:
        print("\n✅ PASS: Coverage is well-calibrated (85-95%)")
    elif coverage < 0.85:
        print(f"\n⚠️ Under-coverage: Intervals too narrow")
    else:
        print(f"\n⚠️ Over-coverage: Intervals too wide")
    
    return {
        'coverage': coverage,
        'n_districts': len(merged),
        'in_interval': merged['in_interval'].sum(),
        'well_calibrated': 0.85 <= coverage <= 0.95
    }


def create_calibration_plot(
    forecasts_path: str = 'outputs/forecasts.csv',
    features_path: str = 'data/processed/model_features.parquet',
    output_path: str = 'outputs/calibration_plot.html'
) -> go.Figure:
    """Create actual vs predicted plot with confidence bands."""
    try:
        forecasts = pd.read_csv(forecasts_path)
        features = pd.read_parquet(features_path)
    except FileNotFoundError:
        return go.Figure().add_annotation(text="No data", showarrow=False)
    
    # Get actuals
    actuals = features.sort_values(['state', 'district', 'year', 'week_number']).groupby(
        ['state', 'district']
    ).tail(1)[['state', 'district', 'bio_update_child']]
    
    # Merge
    forecast_first = forecasts.drop_duplicates(['state', 'district'], keep='first')
    merged = actuals.merge(forecast_first, on=['state', 'district'], how='inner')
    
    if merged.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)
    
    # Sort by predicted value
    merged = merged.sort_values('yhat')
    
    # Create figure
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(range(len(merged))),
        y=merged['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(100, 149, 237, 0.3)'),
        name='Upper 95%'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(merged))),
        y=merged['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(100, 149, 237, 0.3)'),
        name='90% CI'
    ))
    
    # Predicted
    fig.add_trace(go.Scatter(
        x=list(range(len(merged))),
        y=merged['yhat'],
        mode='lines',
        name='Predicted',
        line=dict(color='blue', width=2)
    ))
    
    # Actual
    fig.add_trace(go.Scatter(
        x=list(range(len(merged))),
        y=merged['bio_update_child'],
        mode='markers',
        name='Actual',
        marker=dict(color='red', size=6)
    ))
    
    fig.update_layout(
        title='Calibration: Actual vs Predicted with 90% CI',
        xaxis_title='District (sorted by prediction)',
        yaxis_title='Bio Update Child Count',
        height=500
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"\nSaved plot to {output_path}")
    
    return fig


if __name__ == '__main__':
    results = calculate_prediction_interval_coverage()
    fig = create_calibration_plot()
