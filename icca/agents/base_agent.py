"""
Base Agent interface for ICCA-RAG sub-agents.
Provides standardized message passing, execution lifecycle, and logging.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")

class BaseAgent(ABC):
    """Abstract base class for all specialized sub-agents in ICCA-RAG."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a delegated task and return a standardized result dictionary.
        
        Args:
            task: Dictionary containing 'action', 'payload', and optional metadata.
            
        Returns:
            Dictionary containing 'status' ('success' or 'error'), 'data', and 'message'.
        """
        pass

    def log(self, message: str, level: int = logging.INFO):
        """Log a message with the agent's identity."""
        self.logger.log(level, f"[{self.role}] {message}")
