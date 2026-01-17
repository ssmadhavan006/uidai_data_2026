"""
test_simulator.py
Unit tests for the policy simulation module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_create_simulation_result(self):
        """Test creating a SimulationResult instance."""
        result = simulator.SimulationResult(
            district='Test District',
            state='Test State',
            intervention='mobile_camp',
            initial_backlog=1000,
            final_backlog=500,
            backlog_reduction=500,
            reduction_pct=50.0,
            total_cost=150000,
            cost_per_update=300,
            weeks_simulated=4,
            scenario='median',
            fairness_index=0.7
        )
        assert result.district == 'Test District'
        assert result.reduction_pct == 50.0


class TestSimulateIntervention:
    """Tests for simulate_intervention function."""
    
    @pytest.fixture
    def sample_district_data(self):
        """Create sample district data."""
        return {
            'state': 'Test State',
            'district': 'Test District',
            'backlog': 1000,
            'weekly_demand': 100,
            'baseline_capacity': 80
        }
    
    @pytest.fixture
    def sample_intervention(self):
        """Create sample intervention config."""
        return {
            'description': 'Test intervention',
            'cost': 100000,
            'capacity_per_week': 500,
            'effectiveness': {
                'conservative': 0.2,
                'median': 0.3,
                'optimistic': 0.4
            },
            'duration_weeks': 4
        }
    
    def test_returns_simulation_result(self, sample_district_data, sample_intervention):
        """Test that function returns SimulationResult."""
        result = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='median'
        )
        assert isinstance(result, simulator.SimulationResult)
    
    def test_backlog_decreases_with_intervention(self, sample_district_data, sample_intervention):
        """Test that intervention reduces backlog."""
        result = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='median'
        )
        # With intervention, should process more than baseline
        assert result.final_backlog < (sample_district_data['backlog'] + 
                                       sample_district_data['weekly_demand'] * 4)
    
    def test_scenario_affects_effectiveness(self, sample_district_data, sample_intervention):
        """Test that different scenarios produce different results."""
        result_conservative = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='conservative'
        )
        result_optimistic = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='optimistic'
        )
        # Optimistic should have better reduction
        assert result_optimistic.final_backlog <= result_conservative.final_backlog
    
    def test_cost_per_update_calculated(self, sample_district_data, sample_intervention):
        """Test cost per update is calculated."""
        result = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='median'
        )
        assert result.cost_per_update > 0
        assert result.total_cost == sample_intervention['cost']
    
    def test_fairness_index_in_range(self, sample_district_data, sample_intervention):
        """Test fairness index is between 0 and 1."""
        result = simulator.simulate_intervention(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            scenario='median'
        )
        assert 0 <= result.fairness_index <= 1


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""
    
    @pytest.fixture
    def sample_district_data(self):
        return {
            'state': 'Test State',
            'district': 'Test District',
            'backlog': 500,
            'weekly_demand': 50,
            'baseline_capacity': 40
        }
    
    @pytest.fixture
    def sample_intervention(self):
        return {
            'description': 'Test intervention',
            'cost': 50000,
            'capacity_per_week': 200,
            'effectiveness': {
                'conservative': 0.15,
                'median': 0.25,
                'optimistic': 0.35
            },
            'duration_weeks': 4
        }
    
    def test_returns_dict_with_percentiles(self, sample_district_data, sample_intervention):
        """Test Monte Carlo returns dict with percentile data."""
        result = simulator.run_monte_carlo_simulation(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            n_runs=100  # Small for speed
        )
        assert isinstance(result, dict)
        assert 'backlog_reduction' in result
        assert 'p5' in result['backlog_reduction']
        assert 'p50' in result['backlog_reduction']
        assert 'p95' in result['backlog_reduction']
    
    def test_percentiles_ordered_correctly(self, sample_district_data, sample_intervention):
        """Test that p5 <= p50 <= p95."""
        result = simulator.run_monte_carlo_simulation(
            sample_district_data,
            sample_intervention,
            'test_intervention',
            n_runs=100
        )
        br = result['backlog_reduction']
        assert br['p5'] <= br['p50'] <= br['p95']


class TestLoadInterventions:
    """Tests for loading intervention configs."""
    
    def test_load_interventions_returns_dict(self):
        """Test interventions are loaded as dict."""
        interventions = simulator.load_interventions()
        assert isinstance(interventions, dict)
        assert len(interventions) > 0
    
    def test_interventions_have_required_fields(self):
        """Test each intervention has required fields."""
        interventions = simulator.load_interventions()
        required_fields = ['description', 'cost', 'capacity_per_week', 'effectiveness', 'duration_weeks']
        
        for name, config in interventions.items():
            for field in required_fields:
                assert field in config, f"{name} missing {field}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
