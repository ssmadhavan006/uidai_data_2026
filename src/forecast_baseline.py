"""
03_forecast_baseline.py
Baseline forecasting using Prophet for bio_update_child per district.

Outputs:
- outputs/forecasts.csv with columns: state, district, ds, yhat, yhat_lower, yhat_upper
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Using fallback ETS-style forecast.")


def load_features(path: str = 'data/processed/model_features.parquet') -> pd.DataFrame:
    """Load the feature dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")
    return df


def prepare_prophet_data(df: pd.DataFrame, state: str, district: str) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    Prophet requires columns 'ds' (date) and 'y' (value).
    """
    subset = df[(df['state'] == state) & (df['district'] == district)].copy()
    
    if subset.empty:
        return pd.DataFrame()
    
    # Create date from year and week
    subset['ds'] = subset.apply(
        lambda r: datetime.strptime(f"{int(r['year'])}-W{int(r['week_number']):02d}-1", "%Y-W%W-%w"),
        axis=1
    )
    
    # Target variable
    subset['y'] = subset['bio_update_child'].clip(lower=0)  # Handle suppressed -1 values
    
    return subset[['ds', 'y']].sort_values('ds').reset_index(drop=True)


def train_prophet_model(df_prophet: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    """
    Train Prophet model and generate forecast.
    
    Args:
        df_prophet: DataFrame with 'ds' and 'y' columns
        periods: Number of weeks to forecast
    
    Returns:
        Forecast DataFrame with yhat, yhat_lower, yhat_upper
    """
    if not PROPHET_AVAILABLE or df_prophet.empty or len(df_prophet) < 4:
        # Fallback: simple moving average forecast
        if df_prophet.empty:
            return pd.DataFrame()
        
        last_date = df_prophet['ds'].max()
        last_values = df_prophet['y'].tail(4).values
        mean_val = np.mean(last_values)
        std_val = np.std(last_values) if len(last_values) > 1 else mean_val * 0.1
        
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(periods)]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [mean_val] * periods,
            'yhat_lower': [max(0, mean_val - 1.96 * std_val)] * periods,
            'yhat_upper': [mean_val + 1.96 * std_val] * periods
        })
    
    # Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(df_prophet)
    
    # Future dataframe
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    
    # Return only future periods
    forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def forecast_all_districts(
    input_path: str = 'data/processed/model_features.parquet',
    output_path: str = 'outputs/forecasts.csv',
    forecast_weeks: int = 4,
    max_districts: int = None
) -> pd.DataFrame:
    """
    Generate forecasts for all districts.
    
    Args:
        input_path: Path to feature data
        output_path: Path for output CSV
        forecast_weeks: Number of weeks to forecast
        max_districts: Limit number of districts (for testing)
    
    Returns:
        Combined forecast DataFrame
    """
    print("=" * 60)
    print("BASELINE FORECASTING (Prophet)")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading data...")
    df = load_features(input_path)
    
    # Get unique districts
    districts = df[['state', 'district']].drop_duplicates()
    if max_districts:
        districts = districts.head(max_districts)
    print(f"Forecasting for {len(districts)} districts...")
    
    # Generate forecasts
    print("\n[2/3] Training models and generating forecasts...")
    all_forecasts = []
    
    for i, (_, row) in enumerate(districts.iterrows()):
        state, district = row['state'], row['district']
        
        # Prepare data
        df_prophet = prepare_prophet_data(df, state, district)
        
        if df_prophet.empty or len(df_prophet) < 2:
            continue
        
        # Train and forecast
        try:
            forecast = train_prophet_model(df_prophet, periods=forecast_weeks)
            if not forecast.empty:
                forecast['state'] = state
                forecast['district'] = district
                all_forecasts.append(forecast)
        except Exception as e:
            print(f"  Error for {district}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(districts)} districts...")
    
    # Combine results
    print("\n[3/3] Saving results...")
    if not all_forecasts:
        print("No forecasts generated!")
        return pd.DataFrame()
    
    result = pd.concat(all_forecasts, ignore_index=True)
    result = result[['state', 'district', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Ensure non-negative forecasts
    result['yhat'] = result['yhat'].clip(lower=0)
    result['yhat_lower'] = result['yhat_lower'].clip(lower=0)
    result['yhat_upper'] = result['yhat_upper'].clip(lower=0)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result):,} forecast rows to {output_path}")
    
    print("\n" + "=" * 60)
    print("BASELINE FORECASTING COMPLETE")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    forecasts = forecast_all_districts()
    if not forecasts.empty:
        print(f"\nSample forecasts:")
        print(forecasts.head(10))
