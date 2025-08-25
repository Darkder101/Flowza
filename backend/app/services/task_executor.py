import json
import os
import time
from datetime import datetime, timezone

from app.database.connection import SessionLocal
from app.models.task import Task
from app.models.workflow import Workflow
from app.services.ml_nodes.csv_loader import CSVLoader
from app.services.ml_nodes.drop_nulls import DropNulls
from app.services.ml_nodes.preprocess import Preprocess
from app.services.ml_nodes.train_test_split import TrainTestSplit
from celery import Celery

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


@celery.task
def execute_ml_node(task_id: int):
    """Execute a single ML node task"""
    db = SessionLocal()
    task = None
    try:
        # Get task from database
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"status": "error", "message": "Task not found"}

        # Update task status to running
        task.status = "running"
        task.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Get the appropriate node class
        node_class = NODE_CLASSES.get(task.task_type)
        if not node_class:
            raise ValueError(f"Unknown task type: {task.task_type}")

        # Create and configure the node
        node = node_class(task.node_id, task.parameters or {})

        # Record start time
        start_time = time.time()

        # Execute the node
        result = node.execute()

        # Record execution time
        execution_time = time.time() - start_time

        # Update task with results
        task.result = json.dumps(result)
        task.execution_time = execution_time
        task.status = "completed" if result.get("status") == "success" else "failed"  # noqa : E501
        task.error_message = result.get("message") if result.get("status") == "error" else None  # noqa : E501
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
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
        if not workflow:
            return {"status": "error", "message": "Workflow not found"}

        # Mark workflow as started
        workflow.status = "running"
        workflow.started_at = datetime.now(timezone.utc)
        db.commit()

        # Run actual workflow execution
        from app.services.workflow_service import WorkflowService

        workflow_service = WorkflowService()
        result = workflow_service.execute_workflow(workflow_id)

        # Workflow finished
        workflow.completed_at = datetime.now(timezone.utc)
        workflow.execution_time = workflow.completed_at.timestamp() - workflow.started_at.timestamp()  # noqa : E501
        workflow.status = "completed" if result.get("status") == "success" else "failed"  # noqa : E501
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
