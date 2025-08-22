from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd  # noqa : 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLNode(ABC):
    """Base class for all ML workflow nodes"""

    def __init__(self, node_id: str, parameters: Dict[str, Any] = None):
        self.node_id = node_id
        self.parameters = parameters or {}
        self.inputs = {}
        self.outputs = {}

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the ML node logic"""
        pass

    def set_input(self, input_name: str, input_data: Any):
        """Set input data for the node"""
        self.inputs[input_name] = input_data

    def get_output(self, output_name: str) -> Any:
        """Get output data from the node"""
        return self.outputs.get(output_name)

    def validate_inputs(self, required_inputs: List[str]) -> bool:
        """Validate that all required inputs are present"""
        for input_name in required_inputs:
            if input_name not in self.inputs:
                raise ValueError(f"Required input '{input_name}' is missing")
        return True

    def log_info(self, message: str):
        """Log information about node execution"""
        logger.info(f"[{self.node_id}] {message}")

    def log_error(self, message: str):
        """Log error about node execution"""
        logger.error(f"[{self.node_id}] {message}")
