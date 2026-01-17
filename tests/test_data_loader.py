"""
test_data_loader.py
Unit tests for the data loading utilities.
"""
import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))


class TestMaskSmallValues:
    """Tests for mask_small_values function."""
    
    def test_masks_values_below_threshold(self):
        """Test values below threshold are masked."""
        from app.utils.data_loader import mask_small_values
        
        result = mask_small_values(5, threshold=10)
        assert result == "≤10"
    
    def test_does_not_mask_values_at_threshold(self):
        """Test values at threshold are not masked."""
        from app.utils.data_loader import mask_small_values
        
        result = mask_small_values(10, threshold=10)
        assert result == "10"
    
    def test_does_not_mask_values_above_threshold(self):
        """Test values above threshold are not masked."""
        from app.utils.data_loader import mask_small_values
        
        result = mask_small_values(100, threshold=10)
        assert result == "100"
    
    def test_handles_nan_values(self):
        """Test NaN values return N/A."""
        from app.utils.data_loader import mask_small_values
        import numpy as np
        
        result = mask_small_values(np.nan)
        assert result == "N/A"
    
    def test_handles_zero(self):
        """Test zero is not masked."""
        from app.utils.data_loader import mask_small_values
        
        result = mask_small_values(0)
        assert result == "0"


class TestFormatBottleneckLabel:
    """Tests for format_bottleneck_label function."""
    
    def test_formats_operational_bottleneck(self):
        """Test formatting of OPERATIONAL_BOTTLENECK."""
        from app.utils.data_loader import format_bottleneck_label
        
        result = format_bottleneck_label('OPERATIONAL_BOTTLENECK')
        assert result == 'Hardware & Process Issues'
    
    def test_formats_demographic_surge(self):
        """Test formatting of DEMOGRAPHIC_SURGE."""
        from app.utils.data_loader import format_bottleneck_label
        
        result = format_bottleneck_label('DEMOGRAPHIC_SURGE')
        assert result == 'Population Surge'
    
    def test_formats_normal(self):
        """Test formatting of NORMAL."""
        from app.utils.data_loader import format_bottleneck_label
        
        result = format_bottleneck_label('NORMAL')
        assert result == 'No Issues'
    
    def test_handles_unknown_label(self):
        """Test unknown labels are formatted with title case."""
        from app.utils.data_loader import format_bottleneck_label
        
        result = format_bottleneck_label('SOME_NEW_TYPE')
        assert result == 'Some New Type'
    
    def test_handles_nan(self):
        """Test NaN returns Unknown."""
        from app.utils.data_loader import format_bottleneck_label
        import numpy as np
        
        result = format_bottleneck_label(np.nan)
        assert result == "Unknown"


class TestFormatTextLabel:
    """Tests for format_text_label function."""
    
    def test_formats_underscores(self):
        """Test underscores are replaced with spaces."""
        from app.utils.data_loader import format_text_label
        
        result = format_text_label('some_text_here')
        assert result == 'Some Text Here'
    
    def test_formats_dashes(self):
        """Test dashes are replaced with spaces."""
        from app.utils.data_loader import format_text_label
        
        result = format_text_label('some-text-here')
        assert result == 'Some Text Here'
    
    def test_handles_empty_string(self):
        """Test empty string returns N/A."""
        from app.utils.data_loader import format_text_label
        
        result = format_text_label('')
        assert result == 'N/A'
    
    def test_handles_nan(self):
        """Test NaN returns N/A."""
        from app.utils.data_loader import format_text_label
        import numpy as np
        
        result = format_text_label(np.nan)
        assert result == 'N/A'


class TestGetDistrictData:
    """Tests for get_district_data function."""
    
    @pytest.fixture
    def sample_priority_df(self):
        return pd.DataFrame({
            'state': ['State1', 'State2'],
            'district': ['District1', 'District2'],
            'priority_score': [0.75, 0.45],
            'priority_rank': [1, 2],
            'bottleneck_label': ['OPERATIONAL_BOTTLENECK', 'NORMAL'],
            'forecasted_demand_next_4w': [1000, 500]
        })
    
    @pytest.fixture
    def sample_labels_df(self):
        return pd.DataFrame({
            'state': ['State1', 'State2'],
            'district': ['District1', 'District2'],
            'current_bio_updates': [200, 150],
            'current_demo_updates': [250, 180],
            'update_backlog': [100, 50],
            'completion_rate': [0.7, 0.85],
            'rationale': ['Test rationale 1', 'Test rationale 2']
        })
    
    def test_returns_dict_for_valid_district(self, sample_priority_df, sample_labels_df):
        """Test valid district returns dict."""
        from app.utils.data_loader import get_district_data
        
        result = get_district_data('District1', 'State1', sample_priority_df, sample_labels_df)
        assert isinstance(result, dict)
        assert result['district'] == 'District1'
        assert result['state'] == 'State1'
    
    def test_returns_empty_for_invalid_district(self, sample_priority_df, sample_labels_df):
        """Test invalid district returns empty dict."""
        from app.utils.data_loader import get_district_data
        
        result = get_district_data('FakeDistrict', 'FakeState', sample_priority_df, sample_labels_df)
        assert result == {}
    
    def test_includes_required_fields(self, sample_priority_df, sample_labels_df):
        """Test result includes all required fields."""
        from app.utils.data_loader import get_district_data
        
        result = get_district_data('District1', 'State1', sample_priority_df, sample_labels_df)
        
        required_fields = [
            'district', 'state', 'priority_score', 'priority_rank',
            'bottleneck_label', 'forecasted_demand', 'rationale'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"


class TestGetDistrictList:
    """Tests for get_district_list function."""
    
    def test_returns_sorted_list(self):
        """Test district list is sorted."""
        from app.utils.data_loader import get_district_list
        
        df = pd.DataFrame({
            'district': ['Zebra', 'Apple', 'Mango', 'Banana']
        })
        
        result = get_district_list(df)
        assert result == ['Apple', 'Banana', 'Mango', 'Zebra']
    
    def test_returns_unique_values(self):
        """Test only unique districts returned."""
        from app.utils.data_loader import get_district_list
        
        df = pd.DataFrame({
            'district': ['A', 'B', 'A', 'B', 'C']
        })
        
        result = get_district_list(df)
        assert len(result) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
