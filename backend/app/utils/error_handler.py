"""
Error handling utilities for workflow execution
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict

# Import pandas safely (for handling CSV-related errors)
try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Base exception for workflow errors"""

    def __init__(self, message: str, node_id: str = None, error_code: str = None):  # noqa : 
        self.message = message
        self.node_id = node_id
        self.error_code = error_code
        super().__init__(self.message)


class NodeExecutionError(WorkflowError):
    """Error during node execution"""
    pass


class DataValidationError(WorkflowError):
    """Error during data validation"""
    pass


class DependencyError(WorkflowError):
    """Error in workflow dependencies"""
    pass


def handle_node_error(e: Exception, node_id: str, task_type: str) -> Dict[str, Any]:  # noqa : 
    """Handle and format node execution errors"""
    error_info = {
        "status": "error",
        "node_id": node_id,
        "task_type": task_type,
        "error_type": type(e).__name__,
        "message": str(e),
        "timestamp": datetime.utcnow().isoformat(),
        "traceback": traceback.format_exc(),
    }

    # Log the error
    logger.error(f"Node {node_id} ({task_type}) failed: {str(e)}")
    logger.debug(f"Full traceback: {traceback.format_exc()}")

    # Return user-friendly error message
    if isinstance(e, FileNotFoundError):
        error_info["user_message"] = "File not found. Please check the file path."  # noqa : 
        error_info["error_code"] = "FILE_NOT_FOUND"
    elif pd and isinstance(e, pd.errors.EmptyDataError):
        error_info["user_message"] = "The CSV file appears to be empty."
        error_info["error_code"] = "EMPTY_DATA"
    elif pd and isinstance(e, pd.errors.ParserError):
        error_info["user_message"] = "Failed to parse the CSV file. Check the format and separator."  # noqa : 
        error_info["error_code"] = "PARSE_ERROR"
    elif isinstance(e, KeyError):
        error_info["user_message"] = f"Required column not found in dataset: {str(e)}"  # noqa : 
        error_info["error_code"] = "MISSING_COLUMN"
    elif isinstance(e, ValueError):
        error_info["user_message"] = f"Invalid parameter or data: {str(e)}"
        error_info["error_code"] = "INVALID_VALUE"
    else:
        error_info["user_message"] = f"An unexpected error occurred: {str(e)}"
        error_info["error_code"] = "UNKNOWN_ERROR"

    return error_info


def validate_workflow_definition(workflow_def: Dict[str, Any]) -> Dict[str, Any]:  # noqa : 
    """Validate workflow definition before execution"""
    errors = []
    warnings = []

    nodes = workflow_def.get("nodes", [])
    connections = workflow_def.get("connections", [])

    if not nodes:
        errors.append("Workflow must have at least one node")

    # Check for duplicate node IDs
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node IDs found")

    # Validate connections
    for conn in connections:
        source = conn.get("source")
        target = conn.get("target")

        if source not in node_ids:
            errors.append(f"Connection source '{source}' not found in nodes")

        if target not in node_ids:
            errors.append(f"Connection target '{target}' not found in nodes")

    # Check for circular dependencies
    try:
        from app.services.workflow_service import WorkflowService

        ws = WorkflowService()
        dependency_graph = ws._build_dependency_graph(nodes, connections)
        ws._get_execution_order(dependency_graph)
    except ValueError as e:
        if "Circular dependency" in str(e):
            errors.append("Circular dependency detected in workflow")

    # Check for orphaned nodes (nodes with no inputs and no connections)
    connected_nodes = set()
    for conn in connections:
        connected_nodes.add(conn["source"])
        connected_nodes.add(conn["target"])

    for node in nodes:
        if node["id"] not in connected_nodes and len(connections) > 0:
            warnings.append(f"Node '{node['id']}' is not connected to any other nodes")  # noqa : 

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
