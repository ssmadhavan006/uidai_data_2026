"""
test_privacy_guard.py
Unit tests for the privacy guard module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import privacy_guard as pg


class TestHashIdentifier:
    """Tests for hash_identifier function."""
    
    def test_hash_returns_string(self):
        result = pg.hash_identifier("test_id")
        assert isinstance(result, str)
    
    def test_hash_is_64_chars(self):
        """SHA-256 produces 64 hex characters."""
        result = pg.hash_identifier("test_id")
        assert len(result) == 64
    
    def test_same_input_same_output(self):
        """Same input with same salt = same hash."""
        result1 = pg.hash_identifier("test_id", salt="salt1")
        result2 = pg.hash_identifier("test_id", salt="salt1")
        assert result1 == result2
    
    def test_different_salt_different_output(self):
        """Different salt = different hash."""
        result1 = pg.hash_identifier("test_id", salt="salt1")
        result2 = pg.hash_identifier("test_id", salt="salt2")
        assert result1 != result2


class TestKAnonymity:
    """Tests for apply_k_anonymity function."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'state': ['S1', 'S2', 'S3', 'S4'],
            'district': ['D1', 'D2', 'D3', 'D4'],
            'year': [2025, 2025, 2025, 2025],
            'week_number': [1, 1, 1, 1],
            'enroll_child': [5, 15, 100, 0],  # 5 should be suppressed
            'bio_update_child': [3, 20, 50, 8]  # 3 and 8 should be suppressed
        })
    
    def test_suppresses_values_below_k(self, sample_df, tmp_path):
        log_path = str(tmp_path / "suppression_log.csv")
        result = pg.apply_k_anonymity(
            sample_df, k=10, 
            count_columns=['enroll_child', 'bio_update_child'],
            log_path=log_path
        )
        
        # Check suppressed values are -1
        assert result.loc[0, 'enroll_child'] == -1  # Was 5
        assert result.loc[0, 'bio_update_child'] == -1  # Was 3
        assert result.loc[3, 'bio_update_child'] == -1  # Was 8
    
    def test_does_not_suppress_values_at_k(self, sample_df, tmp_path):
        log_path = str(tmp_path / "suppression_log.csv")
        result = pg.apply_k_anonymity(
            sample_df, k=10,
            count_columns=['enroll_child', 'bio_update_child'],
            log_path=log_path
        )
        
        # Values >= k should remain unchanged
        assert result.loc[1, 'enroll_child'] == 15
        assert result.loc[2, 'enroll_child'] == 100
    
    def test_does_not_suppress_zero(self, sample_df, tmp_path):
        log_path = str(tmp_path / "suppression_log.csv")
        result = pg.apply_k_anonymity(
            sample_df, k=10,
            count_columns=['enroll_child'],
            log_path=log_path
        )
        
        # Zero should remain zero (not suppressed)
        assert result.loc[3, 'enroll_child'] == 0
    
    def test_creates_suppression_log(self, sample_df, tmp_path):
        log_path = str(tmp_path / "suppression_log.csv")
        pg.apply_k_anonymity(
            sample_df, k=10,
            count_columns=['enroll_child', 'bio_update_child'],
            log_path=log_path
        )
        
        # Log should exist and have entries
        assert os.path.exists(log_path)
        log_df = pd.read_csv(log_path)
        assert len(log_df) > 0
        assert 'original_count' in log_df.columns


class TestSanitizeDataframe:
    """Tests for the main sanitize_dataframe function."""
    
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            'state': ['S1'],
            'district': ['D1'],
            'count': [100]
        })
        result = pg.sanitize_dataframe(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'state': ['S1'],
            'district': ['D1'],
            'count': [5]
        })
        original_value = df.loc[0, 'count']
        pg.sanitize_dataframe(df, k=10, count_columns=['count'])
        assert df.loc[0, 'count'] == original_value


class TestValidatePrivacy:
    """Tests for validate_privacy function."""
    
    def test_passes_for_compliant_data(self):
        df = pd.DataFrame({
            'state': ['S1', 'S2'],
            'district': ['D1', 'D2'],
            'count': [100, 200]
        })
        report = pg.validate_privacy(df, k=10)
        assert report['valid'] == True
    
    def test_fails_for_non_compliant_data(self):
        df = pd.DataFrame({
            'state': ['S1', 'S2'],
            'district': ['D1', 'D2'],
            'count': [5, 200]  # 5 is below k=10
        })
        report = pg.validate_privacy(df, k=10)
        assert report['valid'] == False
        assert len(report['issues']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
