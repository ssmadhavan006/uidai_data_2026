"""
test_api.py
Unit tests for the FastAPI endpoints.
"""
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip tests if fastapi not installed
try:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    client = None

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="fastapi not installed")



class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_200(self):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_service_info(self):
        """Test root endpoint contains service info."""
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert data["service"] == "Aadhaar Pulse API"
        assert "version" in data
        assert "endpoints" in data


class TestDistrictsEndpoint:
    """Tests for the /districts endpoint."""
    
    def test_districts_returns_200(self):
        """Test districts endpoint returns 200."""
        response = client.get("/districts")
        assert response.status_code == 200
    
    def test_districts_returns_count(self):
        """Test districts endpoint includes count."""
        response = client.get("/districts")
        data = response.json()
        assert "count" in data
        assert "districts" in data
    
    def test_districts_list_has_required_fields(self):
        """Test each district has required fields."""
        response = client.get("/districts")
        data = response.json()
        if data["districts"]:
            district = data["districts"][0]
            assert "state" in district
            assert "district" in district
            assert "priority_rank" in district


class TestInterventionsEndpoint:
    """Tests for the /interventions endpoint."""
    
    def test_interventions_returns_200(self):
        """Test interventions endpoint returns 200."""
        response = client.get("/interventions")
        assert response.status_code == 200
    
    def test_interventions_returns_dict(self):
        """Test interventions endpoint returns intervention data."""
        response = client.get("/interventions")
        data = response.json()
        assert "count" in data
        assert "interventions" in data
        assert isinstance(data["interventions"], dict)


class TestBottleneckEndpoint:
    """Tests for the /bottleneck/analyze endpoint."""
    
    def test_bottleneck_returns_404_for_invalid_district(self):
        """Test 404 for non-existent district."""
        response = client.get("/bottleneck/analyze/FakeState/FakeDistrict")
        assert response.status_code == 404
    
    def test_bottleneck_returns_analysis_for_valid_district(self):
        """Test analysis is returned for valid district."""
        # First get a valid district
        districts_response = client.get("/districts")
        districts = districts_response.json()["districts"]
        
        if districts:
            district = districts[0]
            response = client.get(f"/bottleneck/analyze/{district['state']}/{district['district']}")
            assert response.status_code == 200
            
            data = response.json()
            assert "bottleneck_label" in data
            assert "priority_score" in data
            assert "explanation" in data


class TestForecastEndpoint:
    """Tests for the /forecast endpoint."""
    
    def test_forecast_returns_404_for_invalid_district(self):
        """Test 404 for non-existent district."""
        response = client.get("/forecast/FakeState/FakeDistrict")
        assert response.status_code == 404
    
    def test_forecast_returns_data_for_valid_district(self):
        """Test forecast is returned for valid district."""
        districts_response = client.get("/districts")
        districts = districts_response.json()["districts"]
        
        if districts:
            district = districts[0]
            response = client.get(f"/forecast/{district['state']}/{district['district']}")
            assert response.status_code == 200
            
            data = response.json()
            assert "predicted_demand" in data
            assert "confidence_interval" in data


class TestRecommendActionEndpoint:
    """Tests for the /recommend_action endpoint."""
    
    def test_recommend_action_returns_404_for_invalid_district(self):
        """Test 404 for non-existent district."""
        response = client.post(
            "/recommend_action",
            json={"district": "FakeDistrict", "state": "FakeState"}
        )
        assert response.status_code == 404
    
    def test_recommend_action_returns_recommendations(self):
        """Test recommendations are returned for valid district."""
        districts_response = client.get("/districts")
        districts = districts_response.json()["districts"]
        
        if districts:
            district = districts[0]
            response = client.post(
                "/recommend_action",
                json={"district": district["district"], "state": district["state"]}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "recommended_interventions" in data
            assert isinstance(data["recommended_interventions"], list)


class TestChatEndpoint:
    """Tests for the /chat endpoint."""
    
    def test_chat_returns_200(self):
        """Test chat endpoint returns 200."""
        response = client.post(
            "/chat",
            json={"message": "Hello"}
        )
        assert response.status_code == 200
    
    def test_chat_returns_response_structure(self):
        """Test chat returns expected structure."""
        response = client.post(
            "/chat",
            json={"message": "What are the top districts?"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "configured" in data
    
    def test_chat_with_clear_history(self):
        """Test chat with clear_history flag."""
        response = client.post(
            "/chat",
            json={"message": "Hello", "clear_history": True}
        )
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
