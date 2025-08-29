# services/ml_nodes/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd  # noqa : 
import logging
from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeOutputSchema(BaseModel):
    """Standard schema for ML node execution output"""
    status: str
    message: str
    node_id: str
    outputs: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class MLNode(ABC):
    """Base class for all ML workflow nodes"""

    def __init__(self, node_id: str, parameters: Dict[str, Any] = None, inputs: Dict[str, Any] = None):
        self.node_id = node_id
        self.parameters = parameters or {}
        self.inputs = inputs or {}
        self.outputs: Dict[str, Any] = {}

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

    def make_response(
        self,
        status: str,
        message: str,
        outputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Return standardized response with optional Pydantic validation"""
        response = {
            "status": status,
            "message": message,
            "node_id": self.node_id,
            "outputs": outputs or self.outputs,
            "metadata": metadata or {},
        }

        # Optional: Validate schema to enforce consistent structure
        try:
            NodeOutputSchema(**response)
        except ValidationError as e:
            self.log_error(f"Output schema validation failed: {str(e)}")
            response = {
                "status": "error",
                "message": f"Invalid node output: {str(e)}",
                "node_id": self.node_id,
                "outputs": {},
                "metadata": {},
            }

        return response

    def execute_safe(self) -> Dict[str, Any]:
        """
        Wrapper for execute() to guarantee a standardized response even if exceptions occur.
        This allows workflow executor to never receive None.
        """
        try:
            return self.execute()
        except Exception as e:
            self.log_error(f"Node execution failed: {str(e)}")
            return self.make_response(status="error", message=str(e))
