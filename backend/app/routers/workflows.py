from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.workflow import Workflow
from app.schemas.workflow_schemas import WorkflowCreate, WorkflowResponse
from typing import List
import json  # noqa : 

router = APIRouter()


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(db: Session = Depends(get_db)):
    """Get all workflows"""
    workflows = db.query(Workflow).all()
    return workflows


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow: WorkflowCreate, db: Session = Depends(get_db)):  # noqa : E501
    """Create a new workflow"""
    db_workflow = Workflow(
        name=workflow.name,
        description=workflow.description,
        nodes=workflow.nodes,
        connections=workflow.connections
    )
    db.add(db_workflow)
    db.commit()
    db.refresh(db_workflow)
    return db_workflow


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get a specific workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.post("/{workflow_id}/execute")
async def execute_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Execute a workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Update workflow status
    workflow.status = "running"
    db.commit()

    # TODO: Trigger Celery task for workflow execution
    # This will be implemented in Step 2

    return {"message": "Workflow execution started", "workflow_id": workflow_id}  # noqa : E501


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
                    "data": {"label": "Load CSV"}
                },
                {
                    "id": "clean_data",
                    "type": "preprocess",
                    "position": {"x": 300, "y": 100},
                    "data": {"label": "Clean Data"}
                },
                {
                    "id": "train_test_split",
                    "type": "train_test_split",
                    "position": {"x": 500, "y": 100},
                    "data": {"label": "Train/Test Split"}
                },
                {
                    "id": "train_model",
                    "type": "train_logreg",
                    "position": {"x": 700, "y": 100},
                    "data": {"label": "Train Logistic Regression"}
                },
                {
                    "id": "evaluate",
                    "type": "evaluate",
                    "position": {"x": 900, "y": 100},
                    "data": {"label": "Evaluate Model"}
                }
            ],
            "connections": [
                {"source": "load_csv", "target": "clean_data"},
                {"source": "clean_data", "target": "train_test_split"},
                {"source": "train_test_split", "target": "train_model"},
                {"source": "train_model", "target": "evaluate"}
            ]
        }
    ]
    return templates
