"""
chatbot_view.py
Streamlit UI component for the AI chatbot interface.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chatbot import AadhaarChatbot


def get_chatbot() -> AadhaarChatbot:
    """Get or create chatbot instance in session state."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AadhaarChatbot()
    return st.session_state.chatbot


def render_chatbot_view():
    """Render the AI chatbot interface."""
    st.header("🤖 AI Assistant")
    st.caption("Ask questions about districts, forecasts, interventions, and more")
    
    chatbot = get_chatbot()
    
    # Check configuration status
    if not chatbot.is_configured():
        st.warning(
            "⚠️ **Chatbot not configured**\n\n"
            "Please add your Gemini API key to the `.env` file:\n"
            "```\nGEMINI_API_KEY=your_actual_api_key\n```"
        )
        
        # Show quick insights without AI
        st.divider()
        st.subheader("📊 Quick Data Insights")
        insights = chatbot.get_quick_insights()
        
        if insights:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Districts", insights.get('total_districts', 'N/A'))
            with col2:
                top = insights.get('top_priority', {})
                st.metric("Top Priority", f"{top.get('district', 'N/A')}")
            with col3:
                st.metric("Avg Score", insights.get('avg_priority_score', 'N/A'))
            
            if 'bottleneck_distribution' in insights:
                st.subheader("Bottleneck Distribution")
                st.bar_chart(insights['bottleneck_distribution'])
        return
    
    # Initialize chat history in session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Quick action buttons
    st.subheader("💡 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_prompts = {
        "🎯 Top Priorities": "What are the top 5 priority districts that need immediate attention?",
        "📊 Bottleneck Summary": "Give me a summary of the bottleneck distribution across all districts.",
        "💰 Interventions": "What interventions are available and their costs?",
        "📈 Forecasts": "Tell me about the demand forecasts for the highest priority districts."
    }
    
    with col1:
        if st.button("🎯 Top Priorities", use_container_width=True):
            st.session_state.pending_message = quick_prompts["🎯 Top Priorities"]
    with col2:
        if st.button("📊 Bottlenecks", use_container_width=True):
            st.session_state.pending_message = quick_prompts["📊 Bottleneck Summary"]
    with col3:
        if st.button("💰 Interventions", use_container_width=True):
            st.session_state.pending_message = quick_prompts["💰 Interventions"]
    with col4:
        if st.button("📈 Forecasts", use_container_width=True):
            st.session_state.pending_message = quick_prompts["📈 Forecasts"]
    
    st.divider()
    
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
    
    # Handle pending message from quick action buttons
    if 'pending_message' in st.session_state and st.session_state.pending_message:
        user_message = st.session_state.pending_message
        st.session_state.pending_message = None
        
        # Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = chatbot.chat(user_message)
        
        # Add assistant message to history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Rerun to update display
        st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask about districts, forecasts, interventions...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = chatbot.chat(user_input)
        
        # Add assistant message to history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Rerun to update display
        st.rerun()
    
    # Clear chat button
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            chatbot.clear_history()
            st.rerun()
    
    with col1:
        st.caption(
            "💡 **Tips:** Ask about specific districts, compare regions, "
            "or inquire about intervention recommendations."
        )
