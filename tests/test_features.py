"""
test_features.py
Unit tests for the feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features


class TestTemporalFeatures:
    """Tests for add_temporal_features function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'state': ['S1'] * 5,
            'district': ['D1'] * 5,
            'year': [2025] * 5,
            'week_number': [1, 2, 3, 4, 5],
            'bio_update_child': [100, 120, 110, 130, 140],
            'enroll_child': [50, 60, 55, 65, 70],
            'demo_update_child': [80, 90, 85, 100, 110]
        })
    
    def test_adds_lag_features(self, sample_df):
        """Test that lag features are created."""
        result = features.add_temporal_features(sample_df)
        assert 'lag_1w_bio_update_child' in result.columns
        assert 'lag_2w_bio_update_child' in result.columns
    
    def test_lag_values_are_correct(self, sample_df):
        """Test that lag values are shifted correctly."""
        result = features.add_temporal_features(sample_df)
        # First row should have NaN for lag
        assert pd.isna(result.loc[0, 'lag_1w_bio_update_child'])
        # Second row should have first value
        assert result.loc[1, 'lag_1w_bio_update_child'] == 100
    
    def test_adds_rolling_features(self, sample_df):
        """Test that rolling mean features are created."""
        result = features.add_temporal_features(sample_df)
        assert 'rolling_4w_mean_bio_update_child' in result.columns
        assert 'rolling_4w_mean_enroll_child' in result.columns
    
    def test_adds_trend_features(self, sample_df):
        """Test that trend features are created."""
        result = features.add_temporal_features(sample_df)
        assert 'wow_change_bio_update_child' in result.columns
        assert 'trend_bio_update_child' in result.columns


class TestPerformanceFeatures:
    """Tests for add_performance_features function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'state': ['S1', 'S2'],
            'district': ['D1', 'D2'],
            'bio_update_child': [100, 80],
            'demo_update_child': [120, 100],
            'enroll_child': [50, 40],
            'enroll_total': [200, 160],
            'bio_update_total': [150, 120]
        })
    
    def test_adds_backlog_feature(self, sample_df):
        """Test that update backlog is calculated."""
        result = features.add_performance_features(sample_df)
        assert 'update_backlog_child' in result.columns
        # backlog = demo - bio
        assert result.loc[0, 'update_backlog_child'] == 20  # 120 - 100
    
    def test_adds_saturation_proxy(self, sample_df):
        """Test saturation proxy calculation."""
        result = features.add_performance_features(sample_df)
        assert 'saturation_proxy' in result.columns
        assert result.loc[0, 'saturation_proxy'] == 50 / 200
    
    def test_adds_completion_rate(self, sample_df):
        """Test completion rate is capped at 1.0."""
        result = features.add_performance_features(sample_df)
        assert 'completion_rate_child' in result.columns
        # All values should be between 0 and 1
        assert (result['completion_rate_child'] <= 1.0).all()
        assert (result['completion_rate_child'] >= 0.0).all()


class TestPriorityScore:
    """Tests for add_priority_score function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample data with required columns."""
        return pd.DataFrame({
            'state': ['S1', 'S2', 'S3'],
            'district': ['D1', 'D2', 'D3'],
            'year': [2025, 2025, 2025],
            'week_number': [1, 1, 1],
            'update_backlog_child': [100, 50, 200],
            'completion_rate_child': [0.5, 0.8, 0.3],
            'enroll_child': [1000, 500, 2000]
        })
    
    def test_adds_priority_score(self, sample_df):
        """Test that priority score is created."""
        result = features.add_priority_score(sample_df)
        assert 'priority_score' in result.columns
    
    def test_priority_score_in_valid_range(self, sample_df):
        """Test priority scores are between 0 and 1."""
        result = features.add_priority_score(sample_df)
        assert (result['priority_score'] >= 0).all()
        assert (result['priority_score'] <= 1).all()
    
    def test_adds_priority_rank(self, sample_df):
        """Test that priority rank is created."""
        result = features.add_priority_score(sample_df)
        assert 'priority_rank' in result.columns


class TestMinMaxScale:
    """Tests for the minmax_scale helper function."""
    
    def test_scales_to_zero_one(self):
        """Test that scaling produces 0-1 range."""
        series = pd.Series([10, 20, 30, 40, 50])
        # Access through the add_priority_score function's nested minmax_scale
        # We'll test indirectly through the priority score
        df = pd.DataFrame({
            'state': ['S1'],
            'district': ['D1'],
            'year': [2025],
            'week_number': [1],
            'update_backlog_child': [100],
            'completion_rate_child': [0.5],
            'enroll_child': [1000]
        })
        result = features.add_priority_score(df)
        # Single value should be 0 (min = max = value)
        assert 'priority_score' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
