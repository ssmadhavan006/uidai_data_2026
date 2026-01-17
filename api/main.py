"""
FastAPI endpoint for Aadhaar Pulse recommendations.

Run with: uvicorn api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Aadhaar Pulse API",
    description="API for child biometric update recommendations and AI chatbot",
    version="1.0.0"
)

# Data paths (relative to project root)
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RecommendationRequest(BaseModel):
    district: str
    state: str
    budget: Optional[float] = 500000


class Intervention(BaseModel):
    name: str
    description: str
    cost: float
    expected_reduction: float
    effectiveness: float


class RecommendationResponse(BaseModel):
    district: str
    state: str
    priority_rank: int
    priority_score: float
    bottleneck_label: str
    recommended_interventions: List[Intervention]


class ChatRequest(BaseModel):
    message: str
    clear_history: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    configured: bool


# Chatbot singleton
_chatbot = None


def get_chatbot():
    """Get or create chatbot instance."""
    global _chatbot
    if _chatbot is None:
        try:
            from src.chatbot import AadhaarChatbot
            _chatbot = AadhaarChatbot()
        except Exception as e:
            print(f"Could not initialize chatbot: {e}")
            return None
    return _chatbot


def load_priority_data():
    """Load priority scores."""
    path = os.path.join(DATA_ROOT, 'outputs/priority_scores.csv')
    return pd.read_csv(path)


def load_interventions():
    """Load intervention configurations."""
    path = os.path.join(DATA_ROOT, 'config/interventions.json')
    with open(path, 'r') as f:
        return json.load(f)


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "service": "Aadhaar Pulse API",
        "version": "1.0.0",
        "endpoints": [
            "GET /districts - List all districts",
            "GET /interventions - List interventions",
            "GET /bottleneck/analyze/{state}/{district} - Analyze bottleneck",
            "GET /forecast/{state}/{district} - Get demand forecast",
            "POST /recommend_action - Get intervention recommendation",
            "POST /chat - Chat with AI assistant"
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with the AI assistant.
    
    Send a message and receive an AI-generated response about
    districts, forecasts, interventions, and more.
    """
    chatbot = get_chatbot()
    
    if chatbot is None:
        return ChatResponse(
            response="Chatbot module not available. Please check installation.",
            configured=False
        )
    
    if request.clear_history:
        chatbot.clear_history()
    
    if not chatbot.is_configured():
        return ChatResponse(
            response="Chatbot not configured. Please add GEMINI_API_KEY to .env file.",
            configured=False
        )
    
    try:
        response = chatbot.chat(request.message)
        return ChatResponse(response=response, configured=True)
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            configured=chatbot.is_configured()
        )


@app.get("/districts")
def list_districts():
    """List all available districts."""
    try:
        priority = load_priority_data()
        districts = priority[['state', 'district', 'priority_rank']].to_dict('records')
        return {"count": len(districts), "districts": districts[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/interventions")
def list_interventions():
    """List available interventions."""
    try:
        interventions = load_interventions()
        return {"count": len(interventions), "interventions": interventions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bottleneck/analyze/{state}/{district}")
def analyze_bottleneck(state: str, district: str):
    """
    Analyze bottleneck for a specific district.
    
    Returns bottleneck classification, priority score, and explanation.
    """
    try:
        priority = load_priority_data()
        
        district_data = priority[
            (priority['district'] == district) &
            (priority['state'] == state)
        ]
        
        if district_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"District '{district}' in state '{state}' not found"
            )
        
        row = district_data.iloc[0]
        
        # Generate explanation based on bottleneck type
        bottleneck = row.get('bottleneck_label', 'UNKNOWN')
        explanations = {
            'OPERATIONAL_BOTTLENECK': 'High failure rate and low throughput indicate hardware or process issues.',
            'DEMOGRAPHIC_SURGE': 'Significant population growth in 5-15 age group creating demand spike.',
            'CAPACITY_STRAIN': 'Demand exceeding current processing capacity at enrollment centers.',
            'ANOMALY_DETECTED': 'Unusual patterns detected requiring investigation.',
            'NORMAL': 'District operating within normal parameters.'
        }
        
        return {
            "state": state,
            "district": district,
            "bottleneck_label": bottleneck,
            "priority_score": float(row['priority_score']) if pd.notna(row['priority_score']) else 0,
            "priority_rank": int(row['priority_rank']) if pd.notna(row['priority_rank']) else 999,
            "explanation": explanations.get(bottleneck, 'Unknown bottleneck type.'),
            "recommended_action": "mobile_camp" if bottleneck in ['OPERATIONAL_BOTTLENECK', 'CAPACITY_STRAIN'] else "staff_training"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{state}/{district}")
def get_forecast(state: str, district: str):
    """
    Get demand forecast for a specific district.
    
    Returns 4-week forecast with confidence intervals.
    """
    try:
        priority = load_priority_data()
        
        district_data = priority[
            (priority['district'] == district) &
            (priority['state'] == state)
        ]
        
        if district_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"District '{district}' in state '{state}' not found"
            )
        
        row = district_data.iloc[0]
        forecast = row.get('forecasted_demand_next_4w', 0)
        
        # Generate confidence intervals (±20%)
        return {
            "state": state,
            "district": district,
            "forecast_horizon": "4 weeks",
            "predicted_demand": float(forecast) if pd.notna(forecast) else 0,
            "confidence_interval": {
                "lower": float(forecast * 0.8) if pd.notna(forecast) else 0,
                "upper": float(forecast * 1.2) if pd.notna(forecast) else 0
            },
            "model": "LightGBM",
            "smape": 62.1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend_action", response_model=RecommendationResponse)
def recommend_action(request: RecommendationRequest):
    """
    Get prioritized intervention recommendation for a district.
    
    Returns the top interventions based on district bottleneck type
    and budget constraints.
    """
    try:
        # Load data
        priority = load_priority_data()
        interventions = load_interventions()
        
        # Find district
        district_data = priority[
            (priority['district'] == request.district) &
            (priority['state'] == request.state)
        ]
        
        if district_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"District '{request.district}' in state '{request.state}' not found"
            )
        
        row = district_data.iloc[0]
        
        # Select interventions based on bottleneck and budget
        recommended = []
        remaining_budget = request.budget
        
        # Sort by effectiveness/cost ratio
        intervention_list = [
            (name, config) for name, config in interventions.items()
        ]
        intervention_list.sort(
            key=lambda x: x[1]['effectiveness']['median'] / x[1]['cost'],
            reverse=True
        )
        
        for name, config in intervention_list:
            if config['cost'] <= remaining_budget:
                recommended.append(Intervention(
                    name=name,
                    description=config['description'],
                    cost=config['cost'],
                    expected_reduction=config['capacity_per_week'] * config['duration_weeks'],
                    effectiveness=config['effectiveness']['median']
                ))
                remaining_budget -= config['cost']
                
                if len(recommended) >= 3:
                    break
        
        return RecommendationResponse(
            district=request.district,
            state=request.state,
            priority_rank=int(row['priority_rank']) if pd.notna(row['priority_rank']) else 999,
            priority_score=float(row['priority_score']) if pd.notna(row['priority_score']) else 0,
            bottleneck_label=row.get('bottleneck_label', 'UNKNOWN'),
            recommended_interventions=recommended
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
