"""
test_chatbot.py
Unit tests for the chatbot module.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatbotImport:
    """Test that chatbot module can be imported."""
    
    def test_import_chatbot_module(self):
        """Test basic import."""
        from src import chatbot
        assert hasattr(chatbot, 'AadhaarChatbot')
        assert hasattr(chatbot, 'create_chatbot')
    
    def test_create_chatbot_instance(self):
        """Test creating chatbot instance without API key."""
        from src.chatbot import AadhaarChatbot
        # Should not raise even without API key
        bot = AadhaarChatbot()
        assert bot is not None


class TestChatbotConfiguration:
    """Test chatbot configuration detection."""
    
    def test_is_not_configured_without_key(self):
        """Test that chatbot reports unconfigured without API key."""
        from src.chatbot import AadhaarChatbot
        bot = AadhaarChatbot(api_key=None)
        assert bot.is_configured() == False
    
    def test_is_not_configured_with_placeholder_key(self):
        """Test that placeholder key is detected as unconfigured."""
        from src.chatbot import AadhaarChatbot
        bot = AadhaarChatbot(api_key="your_gemini_api_key_here")
        assert bot.is_configured() == False


class TestChatbotMethods:
    """Test chatbot helper methods."""
    
    @pytest.fixture
    def bot(self):
        from src.chatbot import AadhaarChatbot
        return AadhaarChatbot(api_key=None)
    
    def test_clear_history(self, bot):
        """Test clearing conversation history."""
        bot.conversation_history = [{"role": "user", "content": "test"}]
        bot.clear_history()
        assert len(bot.conversation_history) == 0
    
    def test_get_quick_insights(self, bot):
        """Test getting quick insights without API."""
        insights = bot.get_quick_insights()
        assert isinstance(insights, dict)
    
    def test_chat_returns_warning_when_not_configured(self, bot):
        """Test that chat returns warning when not configured."""
        response = bot.chat("Hello")
        assert "not configured" in response.lower() or "api key" in response.lower()
    
    def test_needs_context_detection(self, bot):
        """Test context keyword detection."""
        assert bot._needs_context("What are the top districts?") == True
        assert bot._needs_context("What is the priority score?") == True
        assert bot._needs_context("Hello there") == False


class TestChatbotFactory:
    """Test factory function."""
    
    def test_create_chatbot_function(self):
        """Test factory function creates instance."""
        from src.chatbot import create_chatbot
        bot = create_chatbot()
        assert bot is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
