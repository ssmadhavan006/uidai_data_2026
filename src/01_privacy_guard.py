"""
01_privacy_guard.py
Privacy enforcement module for Aadhaar data analysis.

Provides:
- SHA-256 hashing for identifiers
- K-anonymity enforcement with suppression logging
- Main sanitization function
"""
import hashlib
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_salt() -> str:
    """Get the project salt from environment variable."""
    salt = os.getenv('AADHAAR_SALT', 'default_hackathon_salt_2025')
    return salt


def hash_identifier(raw_id: str, salt: Optional[str] = None) -> str:
    """
    Hash an identifier using SHA-256 with salt.
    
    Args:
        raw_id: The raw identifier to hash
        salt: Optional salt (uses env variable if not provided)
    
    Returns:
        Hexadecimal hash string
    """
    if salt is None:
        salt = get_salt()
    
    combined = f"{salt}:{raw_id}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def apply_k_anonymity(
    df: pd.DataFrame,
    k: int = 10,
    count_columns: Optional[List[str]] = None,
    log_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply k-anonymity by suppressing counts below threshold.
    
    Args:
        df: DataFrame to sanitize
        k: Minimum count threshold (default 10)
        count_columns: List of numeric columns to check (auto-detect if None)
        log_path: Path to suppression log CSV (default: data/inventory/suppression_log.csv)
    
    Returns:
        Sanitized DataFrame with suppressed values replaced by -1
    """
    df = df.copy()
    
    # Auto-detect count columns if not provided
    if count_columns is None:
        count_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in ['year', 'week_number', 'lag1_bio_update_child', 'lag1_bio_demo_ratio']
            and 'ratio' not in col.lower()
        ]
    
    # Set default log path
    if log_path is None:
        log_path = 'data/inventory/suppression_log.csv'
    
    # Collect suppression records
    suppressions = []
    
    for col in count_columns:
        if col not in df.columns:
            continue
            
        # Find rows where count is below k but > 0
        mask = (df[col] > 0) & (df[col] < k)
        
        if mask.any():
            # Log the suppressions
            for idx in df[mask].index:
                row = df.loc[idx]
                suppressions.append({
                    'timestamp': datetime.now().isoformat(),
                    'state': row.get('state', 'N/A'),
                    'district': row.get('district', 'N/A'),
                    'year': row.get('year', 'N/A'),
                    'week_number': row.get('week_number', 'N/A'),
                    'column': col,
                    'original_count': row[col],
                    'action': 'suppressed'
                })
            
            # Replace with -1 (suppression marker)
            df.loc[mask, col] = -1
    
    # Save suppression log
    if suppressions:
        log_df = pd.DataFrame(suppressions)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Append to existing log or create new
        if Path(log_path).exists():
            existing = pd.read_csv(log_path)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        
        log_df.to_csv(log_path, index=False)
        print(f"Logged {len(suppressions)} suppressions to {log_path}")
    
    return df


def sanitize_dataframe(
    raw_df: pd.DataFrame,
    k: int = 10,
    count_columns: Optional[List[str]] = None,
    hash_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Main sanitization function - applies hashing and k-anonymity.
    
    Args:
        raw_df: Raw DataFrame to sanitize
        k: K-anonymity threshold
        count_columns: Columns to check for k-anonymity
        hash_columns: Columns to hash (e.g., identifiers)
    
    Returns:
        Sanitized DataFrame
    """
    df = raw_df.copy()
    
    # Hash any identifier columns if specified
    if hash_columns:
        for col in hash_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(hash_identifier)
    
    # Apply k-anonymity
    df = apply_k_anonymity(df, k=k, count_columns=count_columns)
    
    return df


def validate_privacy(df: pd.DataFrame, k: int = 10) -> dict:
    """
    Validate that a DataFrame meets privacy requirements.
    
    Args:
        df: DataFrame to validate
        k: K-anonymity threshold
    
    Returns:
        Validation report dictionary
    """
    report = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    count_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if 'ratio' not in col.lower() and col not in ['year', 'week_number']
    ]
    
    for col in count_cols:
        # Check for values between 0 and k (excluding suppressed -1 values)
        violations = ((df[col] > 0) & (df[col] < k)).sum()
        suppressed = (df[col] == -1).sum()
        
        report['stats'][col] = {
            'violations': int(violations),
            'suppressed': int(suppressed)
        }
        
        if violations > 0:
            report['valid'] = False
            report['issues'].append(f"{col}: {violations} values below k={k}")
    
    return report


if __name__ == '__main__':
    # Test the module
    print("Testing privacy guard module...")
    
    # Test hashing
    test_hash = hash_identifier("test_district_123")
    print(f"Hash test: {test_hash[:16]}...")
    
    # Test k-anonymity
    test_df = pd.DataFrame({
        'state': ['State1', 'State2', 'State3'],
        'district': ['D1', 'D2', 'D3'],
        'year': [2025, 2025, 2025],
        'week_number': [1, 1, 1],
        'enroll_child': [5, 15, 100],  # 5 should be suppressed
        'bio_update_child': [3, 20, 50]  # 3 should be suppressed
    })
    
    print("\nOriginal:")
    print(test_df)
    
    sanitized = sanitize_dataframe(test_df, k=10)
    print("\nSanitized:")
    print(sanitized)
    
    # Validate
    report = validate_privacy(sanitized, k=10)
    print(f"\nValidation: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"Stats: {report['stats']}")
