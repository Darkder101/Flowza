from typing import List

from app.database.connection import get_db
from app.models.workflow import Workflow
from app.schemas.workflow_schemas import (
    WorkflowCreate,
    WorkflowExecuteRequest,
    WorkflowResponse,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(db: Session = Depends(get_db)):
    """Get all workflows"""
    workflows = db.query(Workflow).all()
    return workflows


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    workflow: WorkflowCreate, db: Session = Depends(get_db)
):  # noqa : E501
    """Create a new workflow (doesn't execute it yet)"""
    db_workflow = Workflow(
        name=workflow.name,
        description=workflow.description,
        nodes=[node.model_dump() for node in workflow.nodes],
        connections=[conn.model_dump() for conn in workflow.connections],
    )
    db.add(db_workflow)
    db.commit()
    db.refresh(db_workflow)
    return db_workflow


@router.post("/run")
async def run_workflow(
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create and immediately execute a workflow"""
    db_workflow = Workflow(
        name=request.name,
        description=request.description,
        nodes=[node.model_dump() for node in request.nodes],
        connections=[conn.model_dump() for conn in request.connections],
        status="created",
    )
    db.add(db_workflow)
    db.commit()
    db.refresh(db_workflow)

    # Start background execution
    background_tasks.add_task(execute_workflow_background, db_workflow.id, db)

    return {
        "message": "Workflow created and execution started",
        "workflow_id": db_workflow.id,
        "status": "running",
    }


@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),  # noqa : E501
):
    """Execute an existing workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow.status = "running"
    db.commit()

    background_tasks.add_task(execute_workflow_background, workflow_id, db)

    return {
        "message": "Workflow execution started",
        "workflow_id": workflow_id,
        "status": "running",
    }


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get a specific workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.get("/{workflow_id}/progress")
async def get_workflow_progress(
    workflow_id: int, db: Session = Depends(get_db)
):  # noqa : E501
    """Get workflow execution progress"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    from app.models.task import Task

    tasks = db.query(Task).filter(Task.workflow_id == workflow_id).all()
    task_statuses = []
    for task in tasks:
        task_statuses.append(
            {
                "node_id": task.node_id,
                "task_type": task.task_type,
                "status": task.status,
                "execution_time": task.execution_time,
                "error_message": task.error_message,
            }
        )

    return {
        "workflow_id": workflow_id,
        "status": workflow.status,
        "progress": workflow.progress,
        "started_at": workflow.started_at,
        "execution_time": workflow.execution_time,
        "tasks": task_statuses,
    }


@router.get("/{workflow_id}/results")
async def get_workflow_results(
    workflow_id: int, db: Session = Depends(get_db)
):  # noqa : E501
    """Get workflow execution results"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.status not in ["completed", "failed"]:
        return {
            "status": workflow.status,
            "message": f"Workflow is {workflow.status}, results not yet available",  # noqa : E501
        }

    return {
        "status": workflow.status,
        "workflow_id": workflow_id,
        "execution_time": workflow.execution_time,
        "completed_at": workflow.completed_at,
        "results": workflow.results,
    }


@router.get("/templates/")
async def get_workflow_templates():
    """Get predefined workflow templates"""
    templates = [
        {
            "id": "basic_classification",
            "name": "Basic Classification Pipeline",
            "description": "CSV → Clean → Split → Train Logistic Regression → Evaluate",  # noqa : E501
            "nodes": [
                {
                    "id": "load_csv",
                    "type": "csv_loader",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Load CSV"},
                },
                {
                    "id": "clean_data",
                    "type": "preprocess",
                    "position": {"x": 300, "y": 100},
                    "data": {"label": "Clean Data"},
                },
                {
                    "id": "train_test_split",
                    "type": "train_test_split",
                    "position": {"x": 500, "y": 100},
                    "data": {"label": "Train/Test Split"},
                },
                {
                    "id": "train_model",
                    "type": "train_logreg",
                    "position": {"x": 700, "y": 100},
                    "data": {"label": "Train Logistic Regression"},
                },
                {
                    "id": "evaluate",
                    "type": "evaluate",
                    "position": {"x": 900, "y": 100},
                    "data": {"label": "Evaluate Model"},
                },
            ],
            "connections": [
                {"source": "load_csv", "target": "clean_data"},
                {"source": "clean_data", "target": "train_test_split"},
                {"source": "train_test_split", "target": "train_model"},
                {"source": "train_model", "target": "evaluate"},
            ],
        }
    ]
    return templates


def execute_workflow_background(workflow_id: int, db: Session):
    """Background execution simulation (replace with Celery later)"""
    try:
        workflow = (
            db.query(Workflow).filter(Workflow.id == workflow_id).first()
        )  # noqa : E501
        if workflow:
            # Simulate execution
            workflow.status = "completed"
            workflow.progress = 100
            db.commit()
    except Exception as e:  # noqa : E501
        if workflow:
            workflow.status = "failed"
            db.commit()
