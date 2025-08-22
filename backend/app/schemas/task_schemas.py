from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class TaskCreate(BaseModel):
    node_id: str
    task_type: str
    task_name: str
    parameters: Dict[str, Any]


class TaskResponse(BaseModel):
    id: int
    node_id: str
    task_type: str
    task_name: str
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True
