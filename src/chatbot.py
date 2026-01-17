"""
chatbot.py
Gemini-powered AI chatbot for Aadhaar Pulse data interaction.

Provides:
- Natural language interface to query district data
- Context-aware responses about forecasts, priorities, and interventions
- Conversation memory for multi-turn interactions
"""
import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

# Determine project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from project root
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Gemini API
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

import pandas as pd


class AadhaarChatbot:
    """AI chatbot for Aadhaar Pulse data interaction."""
    
    MODEL_NAME = "gemini-2.5-flash"

    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the chatbot with Gemini API.
        
        Args:
            api_key: Gemini API key. Uses GEMINI_API_KEY env var if not provided.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.chat_session = None
        self.context = ""
        self.conversation_history: List[Dict[str, str]] = []
        
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self._initialize_model()
            self._load_context()
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        genai.configure(api_key=self.api_key)
        
        # Safety settings appropriate for government data context
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # System instruction for the chatbot
        system_instruction = """You are an AI assistant for Aadhaar Pulse, a child biometric update intelligence platform.
        
Your role is to help government officials and analysts understand:
- District-level priority rankings for child biometric updates
- Bottleneck types (OPERATIONAL_BOTTLENECK, DEMOGRAPHIC_SURGE, CAPACITY_STRAIN, INCLUSION_GAP)
- Forecasted demand for the next 4 weeks
- Recommended interventions (mobile camps, device upgrades, staff training, etc.)

Always be:
- Clear and concise
- Data-driven in your responses
- Focused on helping protect children's access to benefits
- Mindful of privacy - never reveal individual-level data

When presenting data:
- Use tables when comparing multiple districts
- Highlight key insights
- Explain bottleneck types in plain language
- Suggest actionable next steps when appropriate

Available data context will be provided. Use it to answer questions accurately."""

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction
        )
        
        self.chat_session = self.model.start_chat(history=[])
    
    def _load_context(self):
        """Load data context from project files."""
        context_parts = []
        
        # Load priority scores
        try:
            priority_path = PROJECT_ROOT / "outputs/priority_scores.csv"
            if priority_path.exists():
                df = pd.read_csv(priority_path)
                top_10 = df.nsmallest(10, 'priority_rank')
                context_parts.append(
                    f"TOP 10 PRIORITY DISTRICTS:\n{top_10[['state', 'district', 'priority_rank', 'priority_score', 'bottleneck_label']].to_string(index=False)}"
                )
        except Exception as e:
            print(f"Could not load priority scores: {e}")
        
        # Load bottleneck labels summary
        try:
            labels_path = PROJECT_ROOT / "outputs/bottleneck_labels.csv"
            if labels_path.exists():
                df = pd.read_csv(labels_path)
                bottleneck_summary = df['bottleneck_label'].value_counts().to_dict() if 'bottleneck_label' in df.columns else {}
                if bottleneck_summary:
                    context_parts.append(
                        f"BOTTLENECK DISTRIBUTION:\n{json.dumps(bottleneck_summary, indent=2)}"
                    )
        except Exception as e:
            print(f"Could not load bottleneck labels: {e}")
        
        # Load interventions config
        try:
            interventions_path = PROJECT_ROOT / "config/interventions.json"
            if interventions_path.exists():
                with open(interventions_path, 'r') as f:
                    interventions = json.load(f)
                context_parts.append(
                    f"AVAILABLE INTERVENTIONS:\n{json.dumps({k: v['description'] for k, v in interventions.items()}, indent=2)}"
                )
        except Exception as e:
            print(f"Could not load interventions: {e}")
        
        self.context = "\n\n".join(context_parts)
    
    def is_configured(self) -> bool:
        """Check if the chatbot is properly configured with API key."""
        return (
            self.api_key is not None 
            and self.api_key != "your_gemini_api_key_here" 
            and self.model is not None
        )
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User message
            
        Returns:
            AI response string
        """
        if not self.is_configured():
            return (
                "⚠️ Chatbot not configured. Please add your Gemini API key to the .env file:\n"
                "GEMINI_API_KEY=your_actual_api_key"
            )
        
        try:
            # Add context to first message or if context-related question
            if len(self.conversation_history) == 0 or self._needs_context(message):
                full_message = f"DATA CONTEXT:\n{self.context}\n\nUSER QUESTION: {message}"
            else:
                full_message = message
            
            # Send to Gemini
            response = self.chat_session.send_message(full_message)
            
            # Store in history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response.text})
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                return "❌ Invalid API key. Please check your GEMINI_API_KEY in the .env file."
            elif "QUOTA" in error_msg.upper():
                return "⚠️ API quota exceeded. Please try again later."
            else:
                return f"❌ Error: {error_msg}"
    
    def _needs_context(self, message: str) -> bool:
        """Check if message needs data context."""
        context_keywords = [
            'district', 'priority', 'bottleneck', 'forecast', 
            'intervention', 'backlog', 'rank', 'score',
            'top', 'worst', 'best', 'compare', 'data'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in context_keywords)
    
    def clear_history(self):
        """Clear conversation history and start fresh."""
        self.conversation_history = []
        if self.is_configured():
            self.chat_session = self.model.start_chat(history=[])
    
    def get_quick_insights(self) -> Dict[str, Any]:
        """Get quick insights from the data without using the AI."""
        insights = {}
        
        try:
            priority_path = PROJECT_ROOT / "outputs/priority_scores.csv"
            if priority_path.exists():
                df = pd.read_csv(priority_path)
                insights['total_districts'] = len(df)
                insights['top_priority'] = df.nsmallest(1, 'priority_rank')[['state', 'district']].iloc[0].to_dict()
                insights['avg_priority_score'] = round(df['priority_score'].mean(), 3)
        except Exception:
            pass
        
        try:
            labels_path = PROJECT_ROOT / "outputs/bottleneck_labels.csv"
            if labels_path.exists():
                df = pd.read_csv(labels_path)
                if 'bottleneck_label' in df.columns:
                    insights['bottleneck_distribution'] = df['bottleneck_label'].value_counts().to_dict()
        except Exception:
            pass
        
        return insights


def create_chatbot(api_key: Optional[str] = None) -> AadhaarChatbot:
    """Factory function to create a chatbot instance."""
    return AadhaarChatbot(api_key=api_key)


if __name__ == "__main__":
    # Test the chatbot
    print("Testing Aadhaar Chatbot...")
    
    bot = AadhaarChatbot()
    
    if bot.is_configured():
        print("Chatbot configured successfully!")
        print("\nTesting chat:")
        response = bot.chat("What are the top 5 priority districts?")
        print(response)
    else:
        print("Chatbot not configured. Add GEMINI_API_KEY to .env file.")
        print("\nQuick insights (no API needed):")
        insights = bot.get_quick_insights()
        print(json.dumps(insights, indent=2))
