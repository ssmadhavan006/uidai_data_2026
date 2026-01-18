"""
chatbot_view.py
Streamlit UI component for the AI chatbot interface.
Enhanced with proper scrolling, better button handling, and improved UX.
Uses callbacks instead of st.rerun() to avoid tab reset issues.
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


def init_chat_state():
    """Initialize chat-related session state."""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []


def handle_quick_action(prompt: str, chatbot: AadhaarChatbot):
    """Callback to handle quick action button click."""
    # Add user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Get AI response
    response = chatbot.chat(prompt)
    
    # Add assistant response
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response
    })


def clear_chat(chatbot: AadhaarChatbot):
    """Callback to clear chat history."""
    st.session_state.chat_messages = []
    chatbot.clear_history()


def render_chatbot_view():
    """Render the AI chatbot interface."""
    
    # Custom CSS for better chat appearance
    st.markdown("""
    <style>
    .chat-welcome {
        text-align: center;
        padding: 2rem;
        color: #666;
    }
    .chat-welcome h3 {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.header("🤖 AI Assistant")
    st.caption("Ask questions about districts, forecasts, interventions, and more")
    
    chatbot = get_chatbot()
    init_chat_state()
    
    # Check configuration status
    if not chatbot.is_configured():
        st.error(
            "⚠️ **Chatbot not configured**\n\n"
            "Add your Gemini API key to `.env`:\n"
            "```\nGEMINI_API_KEY=your_api_key\n```"
        )
        
        # Show quick insights without AI
        st.divider()
        st.subheader("📊 Quick Data Insights (No API Required)")
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
    
    # ===== MAIN CHAT INTERFACE =====
    
    # Layout: Chat area on left, actions on right
    chat_col, action_col = st.columns([3, 1])
    
    with action_col:
        st.markdown("#### 💡 Quick Actions")
        
        # Quick action prompts
        quick_actions = [
            ("🎯 Top Priorities", "What are the top 5 priority districts that need immediate attention?"),
            ("📊 Bottlenecks", "Give me a summary of the bottleneck distribution across all districts."),
            ("💰 Interventions", "What interventions are available and their costs?"),
            ("📈 Forecasts", "Tell me about the demand forecasts for the highest priority districts."),
            ("🏥 Capacity", "Which districts have capacity strain issues?"),
        ]
        
        # Use on_click callbacks instead of checking return value
        for label, prompt in quick_actions:
            st.button(
                label,
                key=f"qa_{label}",
                use_container_width=True,
                on_click=handle_quick_action,
                args=(prompt, chatbot)
            )
        
        st.divider()
        
        # Clear chat button with callback
        st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            type="secondary",
            on_click=clear_chat,
            args=(chatbot,)
        )
        
        # Stats
        st.markdown("---")
        st.caption(f"💬 Messages: {len(st.session_state.chat_messages)}")
        
        # Tips
        st.markdown("---")
        st.markdown("""
        **💡 Try asking:**
        - "Compare District A vs B"
        - "Why is X district high priority?"
        - "Recommend interventions for Y"
        """)
    
    with chat_col:
        # Chat messages container with scrolling
        chat_container = st.container(height=450)
        
        with chat_container:
            if not st.session_state.chat_messages:
                # Welcome message
                st.markdown("""
                <div class="chat-welcome">
                    <h3>👋 Welcome!</h3>
                    <p>I'm your AI assistant for Aadhaar Pulse data.</p>
                    <p>Ask me about districts, priorities, bottlenecks, or interventions.</p>
                    <p>Try the <strong>Quick Actions</strong> on the right to get started!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display all messages
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(msg["content"])
        
        # Chat input at the bottom - processes immediately
        if prompt := st.chat_input("Type your question here...", key="chat_input"):
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message immediately
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
            
            # Get AI response with spinner
            with st.spinner("🤔 Thinking..."):
                response = chatbot.chat(prompt)
            
            # Add and display assistant response
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response
            })
            
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(response)
