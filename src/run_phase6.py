"""
run_phase6.py
Run the complete Phase 6 validation suite.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from datetime import datetime

print("=" * 60)
print("PHASE 6: VALIDATION & SCIENTIFIC DEFENSE")
print("=" * 60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === Step 1: Forecast Validation ===
print("\n" + "=" * 40)
print("STEP 1: FORECAST VALIDATION")
print("=" * 40)

from validation_forecast import run_forecast_validation
forecast_results = run_forecast_validation()

# === Step 2: Anomaly Validation ===
print("\n" + "=" * 40)
print("STEP 2: ANOMALY VALIDATION")
print("=" * 40)

from validation_anomaly import run_anomaly_validation
anomaly_results = run_anomaly_validation()

# === Step 3: Fairness Audit ===
print("\n" + "=" * 40)
print("STEP 3: FAIRNESS AUDIT")
print("=" * 40)

from fairness_audit import run_fairness_audit
fairness_results = run_fairness_audit()

# === Step 4: Calibration Check ===
print("\n" + "=" * 40)
print("STEP 4: UNCERTAINTY CALIBRATION")
print("=" * 40)

from validation_calibration import calculate_prediction_interval_coverage, create_calibration_plot
calibration_results = calculate_prediction_interval_coverage()
create_calibration_plot()

# === SUMMARY ===
print("\n" + "=" * 60)
print("PHASE 6 VALIDATION SUMMARY")
print("=" * 60)

print("\n📊 Forecast Validation:")
print(f"   Rolling CV MAPE: {forecast_results['summary'].get('rolling_cv_mape', 'N/A'):.2f}%" if isinstance(forecast_results['summary'].get('rolling_cv_mape'), float) else "   Rolling CV MAPE: N/A")
print(f"   State Disparity: {forecast_results['summary'].get('state_disparity', 'N/A'):.2f}%" if isinstance(forecast_results['summary'].get('state_disparity'), float) else "   State Disparity: N/A")

print("\n🔍 Anomaly Validation:")
print(f"   Mean Precision: {anomaly_results.get('mean_precision', 'N/A'):.3f}" if isinstance(anomaly_results.get('mean_precision'), float) else "   Mean Precision: N/A")
print(f"   Mean Recall: {anomaly_results.get('mean_recall', 'N/A'):.3f}" if isinstance(anomaly_results.get('mean_recall'), float) else "   Mean Recall: N/A")

print("\n⚖️ Fairness Audit:")
print(f"   Issues Identified: {len(fairness_results.get('issues', []))}")
for issue in fairness_results.get('issues', []):
    print(f"   - {issue}")

print("\n📈 Calibration:")
print(f"   Coverage: {calibration_results.get('coverage', 'N/A'):.1%}" if isinstance(calibration_results.get('coverage'), float) else "   Coverage: N/A")
print(f"   Well-calibrated: {'✅ Yes' if calibration_results.get('well_calibrated') else '⚠️ No'}")

# Output files
print("\n📁 Generated Files:")
outputs = [
    'outputs/validation_forecast.csv',
    'outputs/validation_anomaly.csv',
    'outputs/fairness_report.csv',
    'outputs/calibration_plot.html'
]
for path in outputs:
    if Path(path).exists():
        print(f"   ✅ {path}")
    else:
        print(f"   ⚠️ {path} (not generated)")

print("\n" + "=" * 60)
print("PHASE 6 COMPLETE")
print("=" * 60)
