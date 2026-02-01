"""
Base Pipeline
=============

Abstract base class for pipelines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePipeline(ABC):
    """
    Abstract base class for pipelines.
    
    Provides common interface for all pipeline types.
    """
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate pipeline configuration."""
        pass


class PipelineStep:
    """Represents a single step in a pipeline."""
    
    def __init__(self, name: str, func: callable, required: bool = True):
        self.name = name
        self.func = func
        self.required = required
        self.completed = False
        self.result = None
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the step."""
        self.result = self.func(*args, **kwargs)
        self.completed = True
        return self.result
