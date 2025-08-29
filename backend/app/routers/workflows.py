from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.workflow_service import WorkflowService

router = APIRouter()
workflow_service = WorkflowService()


@router.get("/", response_model=List)
async def list_workflows():
    """Get all workflows"""
    try:
        return workflow_service.list_workflows()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=dict)
async def create_workflow(workflow: dict):
    """Create a new workflow (doesn't execute it yet)"""
    try:
        wf = workflow_service.create_workflow_from_definition(workflow)
        return {"workflow_id": wf.id, "status": wf.status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/run")
async def run_workflow(workflow: dict, background_tasks: BackgroundTasks):
    """Create and immediately execute a workflow"""
    try:
        wf = workflow_service.create_workflow_from_definition(workflow)
        background_tasks.add_task(workflow_service.execute_workflow_async, wf.id)
        return {"workflow_id": wf.id, "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{workflow_id}/execute")
async def execute_workflow(workflow_id: int, background_tasks: BackgroundTasks):
    """Execute an existing workflow"""
    result = workflow_service.get_workflow_progress(workflow_id)
    if not result:
        raise HTTPException(status_code=404, detail="Workflow not found")
    background_tasks.add_task(workflow_service.execute_workflow_async, workflow_id)
    return {"workflow_id": workflow_id, "status": "running"}


@router.get("/{workflow_id}", response_model=dict)
async def get_workflow(workflow_id: int):
    """Get a specific workflow"""
    try:
        return workflow_service.get_workflow_status(workflow_id)
    except HTTPException as e:
        raise e


@router.get("/{workflow_id}/progress")
async def get_workflow_progress(workflow_id: int):
    """Get workflow execution progress"""
    try:
        return workflow_service.get_workflow_progress(workflow_id)
    except HTTPException as e:
        raise e


@router.get("/{workflow_id}/results")
async def get_workflow_results(workflow_id: int):
    """Get workflow execution results"""
    try:
        return workflow_service.get_workflow_results(workflow_id)
    except HTTPException as e:
        raise e


@router.get("/templates/")
async def get_workflow_templates():
    """Get predefined workflow templates"""
    templates = [
        {
            "id": "basic_classification",
            "name": "Basic Classification Pipeline",
            "description": "CSV → Clean → Split → Train Logistic Regression → Evaluate",
            "nodes": [
                {"id": "load_csv", "type": "csv_loader", "position": {"x": 100, "y": 100}, "data": {"label": "Load CSV"}},
                {"id": "clean_data", "type": "preprocess", "position": {"x": 300, "y": 100}, "data": {"label": "Clean Data"}},
                {"id": "train_test_split", "type": "train_test_split", "position": {"x": 500, "y": 100}, "data": {"label": "Train/Test Split"}},
                {"id": "train_model", "type": "train_logreg", "position": {"x": 700, "y": 100}, "data": {"label": "Train Logistic Regression"}},
                {"id": "evaluate", "type": "evaluate", "position": {"x": 900, "y": 100}, "data": {"label": "Evaluate Model"}},
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
