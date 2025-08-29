from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = {}


class WorkflowConnection(BaseModel):
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    nodes: List[WorkflowNode] = []
    connections: List[WorkflowConnection] = []


class WorkflowExecuteRequest(BaseModel):
    name: str
    description: Optional[str] = "Workflow execution request"
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection] = []


class TaskStatus(BaseModel):
    id: int
    node_id: str
    task_type: str
    status: str
    error_message: Optional[str] = None


class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    status: str
    progress: Optional[float] = 0.0
    created_at: datetime
    updated_at: Optional[datetime]
    execution_time: Optional[float] = None

    class Config:
        from_attributes = True


class WorkflowStatusResponse(BaseModel):
    workflow_id: int
    status: str
    name: str
    created_at: datetime
    results: Optional[Dict[str, Any]] = None
    tasks: List[TaskStatus]


class WorkflowProgressResponse(BaseModel):
    workflow_id: int
    status: str
    progress: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
