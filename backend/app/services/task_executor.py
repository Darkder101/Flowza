from celery import Celery
from app.database.connection import SessionLocal
from app.models.task import Task
from app.services.ml_nodes.csv_loader import CSVLoader
from app.services.ml_nodes.preprocess import Preprocess
import os
import time
from datetime import datetime

# Initialize Celery
celery = Celery(
    'flowza_tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6380/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6380/0')
)

# Node type mapping
NODE_CLASSES = {
    'csv_loader': CSVLoader,
    'preprocess': Preprocess,
}


@celery.task
def execute_ml_node(task_id: int):
    """Execute a single ML node task"""
    db = SessionLocal()

    try:
        # Get task from database
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"status": "error", "message": "Task not found"}

        # Update task status to running
        task.status = "running"
        task.updated_at = datetime.utcnow()
        db.commit()

        # Get the appropriate node class
        node_class = NODE_CLASSES.get(task.task_type)
        if not node_class:
            raise ValueError(f"Unknown task type: {task.task_type}")

        # Create and configure the node
        node = node_class(task.node_id, task.parameters)

        # Set inputs if any (this will be more complex in later phases)
        # For now, we'll handle simple cases

        # Record start time
        start_time = time.time()

        # Execute the node
        result = node.execute()

        # Record execution time
        execution_time = time.time() - start_time

        # Update task with results
        task.result = result
        task.execution_time = execution_time
        task.status = "completed" if result.get("status") == "success" else "failed"  # noqa : 
        task.error_message = result.get("message") if result.get("status") == "error" else None  # noqa : 
        task.updated_at = datetime.utcnow()

        db.commit()

        return result

    except Exception as e:
        # Update task with error
        task.status = "failed"
        task.error_message = str(e)
        task.updated_at = datetime.utcnow()
        db.commit()

        return {"status": "error", "message": str(e)}

    finally:
        db.close()


@celery.task
def execute_workflow(workflow_id: int):
    """Execute an entire workflow"""
    # This will be implemented in Week 2
    # For now, just return a placeholder
    return {"status": "success", "message": f"Workflow {workflow_id} execution started"}  # noqa : 