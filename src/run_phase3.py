"""
run_phase3.py
Run the complete Phase 3 pipeline: Forecasting, Anomaly Detection, Bottleneck Fusion.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

print("=" * 60)
print("PHASE 3: FORECASTING & BOTTLENECK DETECTION")
print("=" * 60)

# === STEP 1: Baseline Forecasting ===
print("\n" + "=" * 40)
print("STEP 1: BASELINE FORECASTING")
print("=" * 40)

from forecast_baseline import forecast_all_districts
forecasts = forecast_all_districts(max_districts=200)  # Limit for speed

# === STEP 2: LightGBM Forecasting ===
print("\n" + "=" * 40)
print("STEP 2: ADVANCED FORECASTING (LightGBM)")
print("=" * 40)

try:
    from forecast_lightgbm import train_lightgbm_model
    lgb_results = train_lightgbm_model()
except ImportError as e:
    print(f"LightGBM skipped: {e}")
    lgb_results = {}
except Exception as e:
    print(f"LightGBM error: {e}")
    lgb_results = {}

# === STEP 3: Anomaly Detection ===
print("\n" + "=" * 40)
print("STEP 3: ANOMALY DETECTION")
print("=" * 40)

from anomaly_detection import run_anomaly_detection
anomalies = run_anomaly_detection()

# === STEP 4: Bottleneck Fusion ===
print("\n" + "=" * 40)
print("STEP 4: BOTTLENECK FUSION ENGINE")
print("=" * 40)

from bottleneck_fusion import run_bottleneck_fusion
results, summary = run_bottleneck_fusion()

# === SUMMARY ===
print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)

print("\nOutputs generated:")
outputs = [
    'outputs/forecasts.csv',
    'outputs/forecasts_lgb.csv',
    'outputs/anomalies.csv',
    'outputs/anomalies_flagged.csv',
    'outputs/priority_scores.csv',
    'outputs/bottleneck_labels.csv'
]

for path in outputs:
    if Path(path).exists():
        size = Path(path).stat().st_size
        print(f"  ✅ {path} ({size:,} bytes)")
    else:
        print(f"  ⚠️ {path} (not generated)")

print("\nBottleneck distribution:")
for label, count in summary.items():
    print(f"  {label}: {count}")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
