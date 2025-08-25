import logging
from datetime import datetime, timezone
from typing import Any, Dict

from app.database.connection import SessionLocal
from app.models.task import Task
from app.models.workflow import Workflow
from app.services.task_executor import execute_ml_node

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing workflow execution"""

    def __init__(self):
        self.db = SessionLocal()

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()

    # -----------------------------
    # Workflow Creation
    # -----------------------------
    def create_workflow_from_definition(self, workflow_def: Dict[str, Any]) -> Workflow:  # noqa : E501
        """Create a workflow from JSON definition with validation"""
        try:
            from app.utils.error_handler import validate_workflow_definition, WorkflowError  # noqa : E501

            validation_result = validate_workflow_definition(workflow_def)
            if not validation_result["valid"]:
                raise WorkflowError(
                    f"Invalid workflow definition: {validation_result['errors']}"  # noqa : E501
                )

            if validation_result["warnings"]:
                logger.warning(f"Workflow warnings: {validation_result['warnings']}")  # noqa : E501

            workflow = Workflow(
                name=workflow_def.get("name", "Untitled Workflow"),
                description=workflow_def.get("description", ""),
                nodes=workflow_def.get("nodes", []),
                connections=workflow_def.get("connections", []),
                status="pending",
                progress=0.0,
            )

            self.db.add(workflow)
            self.db.commit()
            self.db.refresh(workflow)

            # create tasks
            for node in workflow_def.get("nodes", []):
                task = Task(
                    workflow_id=workflow.id,
                    node_id=node["id"],
                    task_type=node["type"],
                    task_name=node.get("data", {}).get("label", node["type"]),
                    parameters=node.get("parameters", {}),
                    status="pending",
                )
                self.db.add(task)

            self.db.commit()
            logger.info(
                f"Created workflow {workflow.id} with {len(workflow_def.get('nodes', []))} tasks"  # noqa : E501
            )
            return workflow

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create workflow: {str(e)}")
            raise

    # -----------------------------
    # Execution (Sync)
    # -----------------------------
    def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Execute a workflow by processing nodes in dependency order"""
        try:
            workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            # 🔹 Mark as started
            workflow.status = "running"
            workflow.started_at = datetime.now(timezone.utc)
            workflow.progress = 0.0
            self.db.commit()

            tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501
            dependency_graph = self._build_dependency_graph(workflow.nodes, workflow.connections)  # noqa : E501
            execution_order = self._get_execution_order(dependency_graph)

            node_outputs = {}
            total_tasks = len(execution_order)

            for i, node_id in enumerate(execution_order):
                task = next((t for t in tasks if t.node_id == node_id), None)
                if not task:
                    continue

                logger.info(f"Executing node {node_id} ({i+1}/{total_tasks})")

                # 🔹 Progress update
                progress = ((i) / total_tasks) * 100
                self.update_workflow_progress(workflow_id, progress)

                # Gather inputs
                inputs = self._get_node_inputs(node_id, workflow.connections, node_outputs)  # noqa : E501

                # Execute task
                result = self._execute_single_task(task, inputs)

                if result.get("status") == "error":
                    workflow.status = "failed"
                    workflow.completed_at = datetime.now(timezone.utc)
                    workflow.progress = self.calculate_workflow_progress(workflow_id)  # noqa : E501
                    if workflow.started_at:
                        workflow.execution_time = (
                            workflow.completed_at - workflow.started_at
                        ).total_seconds()
                    self.db.commit()
                    return result

                node_outputs[node_id] = result.get("outputs", {})

            # 🔹 Mark as completed
            workflow.status = "completed"
            workflow.progress = 100.0
            workflow.completed_at = datetime.now(timezone.utc)
            workflow.results = {"final_outputs": node_outputs}

            if workflow.started_at:
                workflow.execution_time = (
                    workflow.completed_at - workflow.started_at
                ).total_seconds()

            self.db.commit()

            return {
                "status": "success",
                "message": f"Workflow {workflow_id} completed successfully",
                "workflow_id": workflow_id,
                "execution_time": workflow.execution_time,
                "outputs": node_outputs,
            }

        except Exception as e:
            if "workflow" in locals() and workflow:
                workflow.status = "failed"
                workflow.completed_at = datetime.now(timezone.utc)
                workflow.progress = self.calculate_workflow_progress(workflow_id)  # noqa : E501
                if workflow.started_at:
                    workflow.execution_time = (
                        workflow.completed_at - workflow.started_at
                    ).total_seconds()
                self.db.commit()

            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "workflow_id": workflow_id,
            }

    # -----------------------------
    # Execution (Async w/ Celery)
    # -----------------------------
    def execute_workflow_async(self, workflow_id: int) -> Dict[str, Any]:
        """Execute workflow asynchronously using Celery"""
        try:
            workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow.status = "running"
            workflow.started_at = datetime.now(timezone.utc)
            workflow.progress = 0.0
            self.db.commit()

            tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501
            dependency_graph = self._build_dependency_graph(workflow.nodes, workflow.connections)  # noqa : E501
            execution_order = self._get_execution_order(dependency_graph)

            for node_id in execution_order:
                task = next((t for t in tasks if t.node_id == node_id), None)
                if not task:
                    continue
                execute_ml_node.delay(task.id)
                self.db.commit()

            return {
                "status": "success",
                "message": f"Workflow {workflow_id} dispatched to Celery",
                "workflow_id": workflow_id,
            }

        except Exception as e:
            if "workflow" in locals() and workflow:
                workflow.status = "failed"
                workflow.completed_at = datetime.now(timezone.utc)
                if workflow.started_at:
                    workflow.execution_time = (
                        workflow.completed_at - workflow.started_at
                    ).total_seconds()
                self.db.commit()

            return {
                "status": "error",
                "message": str(e),
                "workflow_id": workflow_id,
            }

    # -----------------------------
    # Progress + Status
    # -----------------------------
    def update_workflow_progress(
        self, workflow_id: int, progress: float, status: str = None
    ):
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
        if workflow:
            workflow.progress = min(100.0, max(0.0, progress))
            if status:
                workflow.status = status
            self.db.commit()

    def calculate_workflow_progress(self, workflow_id: int) -> float:
        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501
        if not tasks:
            return 0.0
        completed = len([t for t in tasks if t.status == "completed"])
        failed = len([t for t in tasks if t.status == "failed"])
        return ((completed + failed) / len(tasks)) * 100

    def get_workflow_status(self, workflow_id: int) -> Dict[str, Any]:
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
        if not workflow:
            return {"status": "error", "message": "Workflow not found"}

        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501
        return {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "name": workflow.name,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "execution_time": workflow.execution_time,
            "progress": workflow.progress,
            "results": workflow.results,
            "tasks": [
                {
                    "id": task.node_id,
                    "type": task.task_type,
                    "name": task.task_name,
                    "status": task.status,
                    "parameters": task.parameters,
                    "result": task.result,
                }
                for task in tasks
            ],
        }
