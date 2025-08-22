from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]


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


class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    status: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
