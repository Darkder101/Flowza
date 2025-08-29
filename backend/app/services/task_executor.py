import json
import os
import time
from datetime import datetime, timezone

from app.database.connection import SessionLocal
from app.models.task import Task
from app.models.workflow import Workflow
from app.services.dataset_service import DatasetService
from app.schemas.task_schemas import TaskResponse
from app.services.ml_nodes.csv_loader import CSVLoader
from app.services.ml_nodes.drop_nulls import DropNulls
from app.services.ml_nodes.preprocess import Preprocess
from app.services.ml_nodes.train_test_split import TrainTestSplit
from celery import Celery
from fastapi import HTTPException

# Initialize Celery
celery = Celery(
    "flowza_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6380/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6380/0"),
)

# Node type mapping - all supported ML nodes
NODE_CLASSES = {
    "csv_loader": CSVLoader,
    "preprocess": Preprocess,
    "drop_nulls": DropNulls,
    "train_test_split": TrainTestSplit,
}


def execute_node_safe(node):
    """Execute an MLNode and ensure standardized response"""
    try:
        result = node.execute()
        # Ensure it's always a dict from make_response
        if not isinstance(result, dict):
            return node.make_response("error", "Node did not return a valid dict")
        return result
    except Exception as e:
        node.log_error(f"Node execution failed: {str(e)}")
        return node.make_response("error", str(e))


@celery.task
def execute_ml_node(task_id: int):
    """Execute a single ML node task"""
    db = SessionLocal()
    task = None
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"status": "error", "message": "Task not found"}

        # Update task status to running
        task.status = "running"
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Get ML node class
        node_class = NODE_CLASSES.get(task.task_type)
        if not node_class:
            raise ValueError(f"Unknown task type: {task.task_type}")

        node = node_class(task.node_id, task.parameters or {})

        # Execute node safely
        start_time = time.time()
        result = execute_node_safe(node)
        execution_time = time.time() - start_time

        # Save dataset if output contains a dataset
        if result.get("status") == "success" and "dataset" in result.get("outputs", {}):
            dataset_service = DatasetService()
            dataset_id = dataset_service.save_dataset(
                result["outputs"]["dataset"], workflow_id=task.workflow_id, node_id=task.node_id
            )
            result["outputs"]["dataset"] = dataset_id  # replace raw data with saved dataset ID

        # Update task info
        task.result = json.dumps(result)
        task.execution_time = execution_time
        task.status = "completed" if result.get("status") == "success" else "failed"
        task.error_message = result.get("message") if result.get("status") == "error" else None
        task.updated_at = datetime.now(timezone.utc)

        db.commit()
        return result

    except Exception as e:
        if task:
            task.status = "failed"
            task.error_message = str(e)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


@celery.task
def execute_workflow(workflow_id: int):
    """Execute an entire workflow - delegates to WorkflowService"""
    db = SessionLocal()
    workflow = None
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            return {"status": "error", "message": "Workflow not found"}

        workflow.status = "running"
        workflow.started_at = datetime.now(timezone.utc)
        db.commit()

        from app.services.workflow_service import WorkflowService

        workflow_service = WorkflowService()
        result = workflow_service.execute_workflow(workflow_id)

        workflow.completed_at = datetime.now(timezone.utc)
        workflow.execution_time = (
            workflow.completed_at.timestamp() - workflow.started_at.timestamp()
        )
        workflow.status = "completed" if result.get("status") == "success" else "failed"
        workflow.updated_at = datetime.now(timezone.utc)
        db.commit()

        return result

    except Exception as e:
        if workflow:
            workflow.status = "failed"
            workflow.completed_at = datetime.now(timezone.utc)
            workflow.updated_at = datetime.now(timezone.utc)
            db.commit()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


@celery.task
def get_task_status(task_id: int):
    """Fetch the current status of a task by its ID"""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskResponse.model_validate(task).model_dump()

    finally:
        db.close()


@celery.task
def get_task_output_dataset(task_id: int):
    """Fetch the output dataset(s) of a task by its ID"""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if not task.output_data:
            return {"status": "empty", "message": "No output dataset available yet."}

        return {"status": "success", "output_data": task.output_data}

    finally:
        db.close()
