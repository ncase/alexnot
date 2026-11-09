"""
A simple object to collect Reasoning Trace messages by category (e.g. parse, transform, etc)
"""

from typing import Dict, List
from collections import defaultdict

class ReasoningTrace:
    """
    Collects debug messages organized by category.
    
    Usage:
        trace = ReasoningTrace()
        trace.log("parse", "Trying parser settings...")
        trace.log("transform", "Best program so far is...")
        
        # Get messages for a category
        perception_logs = trace.get_logs("perception")

        # Clear all logs
        trace.clear()
    """
    def __init__(self):
        # Auto-creates empty lists for new categories
        self.logs: Dict[str, List[str]] = defaultdict(list)
    
    def log(self, category:str, message:str) -> None:
        """Add a message to the specified category."""
        self.logs[category].append(message)
    
    def get_log(self, category:str) -> str:
        """Get all messages for a specific category. Concats as a big' ol string."""
        return '\n\n'.join(self.logs[category])
    
    def clear(self) -> None:
        """Clear all logged messages."""
        self.logs.clear()