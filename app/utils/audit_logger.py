"""
audit_logger.py
JSON-formatted audit logging for Aadhaar Pulse dashboard.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging


class AuditLogger:
    """
    Audit logger for tracking user actions in the dashboard.
    Writes JSON-formatted logs for security and compliance.
    """
    
    def __init__(self, log_dir: str = "outputs/audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Python logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Create daily log file handler
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y-%m-%d')}.json"
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def log(
        self,
        user_id: str,
        role: str,
        action: str,
        target: Optional[str] = None,
        status: str = "success",
        details: Optional[dict] = None
    ):
        """
        Log an audit event.
        
        Args:
            user_id: Username or identifier
            role: User role (Analyst, Viewer, Admin)
            action: Action performed (login, export, simulation, etc.)
            target: Target of action (district name, file path, etc.)
            status: success/failure/denied
            details: Additional context dictionary
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "role": role,
            "action": action,
            "target": target,
            "status": status,
            "details": details or {}
        }
        
        self.logger.info(json.dumps(entry))
        return entry
    
    def log_login(self, user_id: str, role: str, success: bool = True):
        """Log a login attempt."""
        return self.log(
            user_id=user_id,
            role=role,
            action="login",
            status="success" if success else "failed"
        )
    
    def log_logout(self, user_id: str, role: str):
        """Log a logout."""
        return self.log(
            user_id=user_id,
            role=role,
            action="logout"
        )
    
    def log_export(self, user_id: str, role: str, dataset: str, row_count: int):
        """Log a data export action."""
        return self.log(
            user_id=user_id,
            role=role,
            action="export",
            target=dataset,
            details={"row_count": row_count}
        )
    
    def log_simulation(self, user_id: str, role: str, params: dict):
        """Log a simulation run."""
        return self.log(
            user_id=user_id,
            role=role,
            action="simulation",
            details=params
        )
    
    def log_view(self, user_id: str, role: str, view_name: str, district: Optional[str] = None):
        """Log a dashboard view access."""
        return self.log(
            user_id=user_id,
            role=role,
            action="view",
            target=view_name,
            details={"district": district} if district else {}
        )
    
    def log_access_denied(self, user_id: str, role: str, resource: str):
        """Log an access denied event."""
        return self.log(
            user_id=user_id,
            role=role,
            action="access_attempt",
            target=resource,
            status="denied"
        )


# Global logger instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get the singleton audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


if __name__ == "__main__":
    # Test the audit logger
    logger = get_audit_logger()
    
    print("Testing audit logger...")
    logger.log_login("analyst_user", "Analyst", True)
    logger.log_view("analyst_user", "Analyst", "map_view", "Mumbai")
    logger.log_simulation("analyst_user", "Analyst", {"budget": 100000, "interventions": 3})
    logger.log_export("analyst_user", "Analyst", "priority_scores.csv", 742)
    logger.log_logout("analyst_user", "Analyst")
    
    print(f"Logs written to {logger.log_dir}")
