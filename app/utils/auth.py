"""
auth.py
Authentication and Role-Based Access Control for Aadhaar Pulse Dashboard.
"""
import streamlit as st
import hashlib
from typing import Optional, Tuple
from datetime import datetime

# Try to use streamlit-authenticator if available
try:
    import streamlit_authenticator as stauth
    STAUTH_AVAILABLE = True
except ImportError:
    STAUTH_AVAILABLE = False


def hash_password(password: str) -> str:
    """Simple password hashing for hackathon demo."""
    return hashlib.sha256(password.encode()).hexdigest()


# Demo credentials (in production, load from secrets.toml)
DEMO_USERS = {
    "analyst": {
        "password": hash_password("analyst123"),
        "name": "Analyst User",
        "role": "Analyst"
    },
    "viewer": {
        "password": hash_password("viewer123"),
        "name": "Viewer User", 
        "role": "Viewer"
    },
    "admin": {
        "password": hash_password("admin123"),
        "name": "Admin User",
        "role": "Admin"
    }
}


def init_session_state():
    """Initialize session state for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "role" not in st.session_state:
        st.session_state.role = None
    if "name" not in st.session_state:
        st.session_state.name = None


def authenticate(username: str, password: str) -> Tuple[bool, Optional[str]]:
    """
    Authenticate a user.
    
    Returns:
        (success, role) tuple
    """
    if username in DEMO_USERS:
        stored = DEMO_USERS[username]
        if stored["password"] == hash_password(password):
            return True, stored["role"]
    return False, None


def login_form() -> bool:
    """
    Render login form and handle authentication.
    
    Returns:
        True if authenticated, False otherwise
    """
    init_session_state()
    
    if st.session_state.authenticated:
        return True
    
    st.markdown("""
    <div style="max-width: 400px; margin: 50px auto; padding: 30px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; color: white;">
        <h2 style="text-align: center; margin-bottom: 20px;">🔐 Aadhaar Pulse Login</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown("### Enter Credentials")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submit:
            success, role = authenticate(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.session_state.name = DEMO_USERS[username]["name"]
                
                # Log the login
                from utils.audit_logger import get_audit_logger
                get_audit_logger().log_login(username, role, True)
                
                st.rerun()
            else:
                st.error("Invalid username or password")
                
                # Log failed attempt
                try:
                    from utils.audit_logger import get_audit_logger
                    get_audit_logger().log_login(username, "unknown", False)
                except:
                    pass
    
    # Demo credentials hint
    st.markdown("""
    ---
    **Demo Credentials:**
    - Analyst: `analyst` / `analyst123` (full access)
    - Viewer: `viewer` / `viewer123` (read-only, masked data)
    """)
    
    return False


def logout():
    """Log out the current user."""
    if st.session_state.get("authenticated"):
        # Log the logout
        try:
            from utils.audit_logger import get_audit_logger
            get_audit_logger().log_logout(
                st.session_state.get("username", "unknown"),
                st.session_state.get("role", "unknown")
            )
        except:
            pass
    
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.name = None
    st.rerun()


def require_role(required_role: str) -> bool:
    """
    Check if current user has required role.
    
    Role hierarchy: Admin > Analyst > Viewer
    """
    role_hierarchy = {"Admin": 3, "Analyst": 2, "Viewer": 1}
    current = role_hierarchy.get(st.session_state.role, 0)
    required = role_hierarchy.get(required_role, 0)
    return current >= required


def get_current_user() -> dict:
    """Get current user info."""
    return {
        "username": st.session_state.get("username"),
        "role": st.session_state.get("role"),
        "name": st.session_state.get("name"),
        "authenticated": st.session_state.get("authenticated", False)
    }


def render_user_sidebar():
    """Render user info in sidebar."""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**👤 {st.session_state.name}**")
            st.markdown(f"Role: `{st.session_state.role}`")
            if st.button("🚪 Logout"):
                logout()
