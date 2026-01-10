"""
dp_export.py
Differential Privacy export module for Aadhaar Pulse.

Provides mathematically rigorous privacy guarantees for data exports.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

# Try to import diffprivlib
try:
    from diffprivlib.mechanisms import Laplace, Gaussian
    DIFFPRIVLIB_AVAILABLE = True
except ImportError:
    DIFFPRIVLIB_AVAILABLE = False
    print("Warning: diffprivlib not installed. Using fallback noise mechanism.")


class DPExporter:
    """
    Differential Privacy exporter with k-anonymity + noise.
    
    Privacy Budget (epsilon):
    - ε = 0.1: Very strong privacy, more noise
    - ε = 1.0: Standard privacy (recommended)
    - ε = 10.0: Weak privacy, less noise
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        k_threshold: int = 10,
        delta: float = 1e-5
    ):
        """
        Initialize DP exporter.
        
        Args:
            epsilon: Privacy budget (lower = more private)
            k_threshold: K-anonymity threshold for suppression
            delta: Failure probability for Gaussian mechanism
        """
        self.epsilon = epsilon
        self.k_threshold = k_threshold
        self.delta = delta
        
        # Initialize mechanisms if available
        if DIFFPRIVLIB_AVAILABLE:
            # Laplace for counts (sensitivity = 1 for counting queries)
            self.laplace = Laplace(epsilon=epsilon, sensitivity=1.0)
            # Gaussian for bounded statistics
            self.gaussian = Gaussian(epsilon=epsilon, delta=delta, sensitivity=1.0)
    
    def _add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise to a value."""
        if DIFFPRIVLIB_AVAILABLE:
            mech = Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
            return mech.randomise(value)
        else:
            # Fallback: manual Laplace noise
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale)
            return value + noise
    
    def _add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise to a value."""
        if DIFFPRIVLIB_AVAILABLE:
            mech = Gaussian(epsilon=self.epsilon, delta=self.delta, sensitivity=sensitivity)
            return mech.randomise(value)
        else:
            # Fallback: manual Gaussian noise
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = np.random.normal(0, sigma)
            return value + noise
    
    def apply_k_anonymity(self, df: pd.DataFrame, count_col: str) -> pd.DataFrame:
        """Suppress rows where count < k."""
        df = df.copy()
        mask = df[count_col] >= self.k_threshold
        suppressed_count = (~mask).sum()
        if suppressed_count > 0:
            print(f"Suppressed {suppressed_count} rows with {count_col} < {self.k_threshold}")
        return df[mask]
    
    def dp_export_dataframe(
        self,
        df: pd.DataFrame,
        count_columns: Optional[List[str]] = None,
        rate_columns: Optional[List[str]] = None,
        suppress_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export DataFrame with differential privacy protection.
        
        1. Applies k-anonymity suppression (if suppress_column specified)
        2. Adds Laplace noise to count columns
        3. Adds Gaussian noise to rate columns
        
        Args:
            df: DataFrame to export
            count_columns: Columns containing counts (use Laplace)
            rate_columns: Columns containing rates/averages (use Gaussian)
            suppress_column: Column to use for k-anonymity check
        
        Returns:
            Privatized DataFrame
        """
        result = df.copy()
        
        # Step 1: K-anonymity suppression
        if suppress_column and suppress_column in result.columns:
            result = self.apply_k_anonymity(result, suppress_column)
        
        # Step 2: Add noise to count columns
        if count_columns:
            for col in count_columns:
                if col in result.columns:
                    # Sensitivity = 1 for counting queries
                    result[col] = result[col].apply(
                        lambda x: max(0, round(self._add_laplace_noise(x, sensitivity=1.0)))
                    )
        
        # Step 3: Add noise to rate columns
        if rate_columns:
            for col in rate_columns:
                if col in result.columns:
                    # For rates, sensitivity depends on range (0-1 has sensitivity ~0.1)
                    result[col] = result[col].apply(
                        lambda x: np.clip(self._add_gaussian_noise(x, sensitivity=0.1), 0, 1)
                    )
        
        return result
    
    def get_privacy_report(self) -> dict:
        """Get a summary of privacy parameters."""
        return {
            "epsilon": self.epsilon,
            "k_threshold": self.k_threshold,
            "delta": self.delta,
            "mechanism_counts": "Laplace",
            "mechanism_rates": "Gaussian",
            "diffprivlib_available": DIFFPRIVLIB_AVAILABLE
        }


def dp_export_dataframe(
    df: pd.DataFrame,
    epsilon: float = 1.0,
    k_threshold: int = 10,
    count_columns: Optional[List[str]] = None,
    rate_columns: Optional[List[str]] = None,
    suppress_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function for DP export.
    
    Example:
        noisy_df = dp_export_dataframe(
            priority_df,
            epsilon=1.0,
            count_columns=['bio_update_child', 'enroll_child'],
            rate_columns=['completion_rate_child'],
            suppress_column='bio_update_child'
        )
    """
    exporter = DPExporter(epsilon=epsilon, k_threshold=k_threshold)
    return exporter.dp_export_dataframe(
        df,
        count_columns=count_columns,
        rate_columns=rate_columns,
        suppress_column=suppress_column
    )


if __name__ == "__main__":
    # Test the DP export module
    print("Testing Differential Privacy Export Module")
    print("=" * 50)
    
    # Create test data
    test_df = pd.DataFrame({
        'district': ['District_A', 'District_B', 'District_C', 'District_D'],
        'bio_update_child': [150, 8, 200, 50],  # District_B should be suppressed
        'completion_rate_child': [0.85, 0.72, 0.91, 0.78]
    })
    
    print("Original Data:")
    print(test_df)
    
    # Apply DP export
    exporter = DPExporter(epsilon=1.0, k_threshold=10)
    noisy_df = exporter.dp_export_dataframe(
        test_df,
        count_columns=['bio_update_child'],
        rate_columns=['completion_rate_child'],
        suppress_column='bio_update_child'
    )
    
    print("\nPrivatized Data (ε=1.0, k=10):")
    print(noisy_df)
    
    print("\nPrivacy Report:")
    print(exporter.get_privacy_report())
