import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.database.connection import SessionLocal
from app.models.task import Task
from app.models.workflow import Workflow
from app.schemas.workflow_schemas import WorkflowProgressResponse
from app.services.dataset_service import DatasetService
from app.services.task_executor import execute_ml_node  # Celery task (async path)
from fastapi import HTTPException
from pydantic import BaseModel

# We use the same registry as task_executor does
# (expected to be defined in app/ml_nodes/__init__.py as NODE_CLASSES)
try:
    from app.services.ml_nodes import NODE_CLASSES
except (
    Exception
):  # keep service import-safe even if registry import fails at tooling time
    NODE_CLASSES = {}

logger = logging.getLogger(__name__)

# -----------------------------
# Pydantic Schemas
# -----------------------------


class WorkflowListItem(BaseModel):
    id: int
    name: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class WorkflowOutputResponse(BaseModel):
    workflow_id: int
    status: str
    progress: float
    execution_time: Optional[float]
    results: Optional[Dict[str, Any]]

    model_config = {"from_attributes": True}


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

    model_config = {"from_attributes": True}


class WorkflowTasksResponse(BaseModel):
    workflow_id: int
    status: str
    tasks: List[TaskResponse]

    model_config = {"from_attributes": True}


# -----------------------------
# Workflow Service
# -----------------------------
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
    def create_workflow_from_definition(self, workflow_def: Dict[str, Any]) -> Workflow:
        """Create a workflow from JSON definition with validation"""
        try:
            from app.utils.error_handler import (
                WorkflowError,
                validate_workflow_definition,
            )

            # ✅ Validate the workflow definition
            validation_result = validate_workflow_definition(workflow_def)
            if not validation_result["valid"]:
                raise WorkflowError(
                    f"Invalid workflow definition: {validation_result['errors']}"
                )

            if validation_result.get("warnings"):
                logger.warning(f"Workflow warnings: {validation_result['warnings']}")

            # ✅ Create workflow entry
            workflow = Workflow(
                name=workflow_def.get("name", "Untitled Workflow"),
                description=workflow_def.get("description", ""),
                nodes=workflow_def.get("nodes", []),
                connections=workflow_def.get("connections", []),
                status="pending",
                progress=0.0,
            )

            self.db.add(workflow)
            self.db.commit()  # commit so workflow gets an ID
            self.db.refresh(workflow)  # refresh to populate workflow.id

            # ✅ Create Task rows from nodes (after workflow.id exists)
            tasks_to_add = []
            for node in workflow_def.get("nodes", []):
                task = Task(
                    workflow_id=workflow.id,
                    node_id=node["id"],
                    task_type=node["type"],
                    task_name=node.get("data", {}).get("label", node["type"]),
                    parameters=node.get("parameters", {}),
                    status="pending",
                )
                tasks_to_add.append(task)

            if tasks_to_add:
                self.db.add_all(tasks_to_add)
                self.db.commit()

            logger.info(
                f"✅ Created workflow {workflow.id} with {len(tasks_to_add)} tasks"
            )
            return workflow

        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Failed to create workflow: {str(e)}", exc_info=True)
            raise

    # -----------------------------
    # Execution (Sync)
    # -----------------------------
    def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Execute a workflow by processing nodes in dependency order (synchronously)."""
        dataset_service = DatasetService()
        try:
            workflow = (
                self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
            )
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow.status = "running"
            workflow.started_at = datetime.now(timezone.utc)
            workflow.progress = 0.0
            self.db.commit()

            tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
            dependency_graph = self._build_dependency_graph(
                workflow.nodes, workflow.connections
            )
            execution_order = self._get_execution_order(dependency_graph)

            node_outputs: Dict[str, Dict[str, Any]] = {}
            total_tasks = len(execution_order)

            for i, node_id in enumerate(execution_order):
                task = next((t for t in tasks if t.node_id == node_id), None)
                if not task:
                    # If no task row exists for the node, skip but continue progress
                    self.update_workflow_progress(
                        workflow_id, ((i + 1) / total_tasks) * 100
                    )
                    continue

                logger.info(f"Executing node {node_id} ({i + 1}/{total_tasks})")

                # Update progress before executing current node (0..100)
                progress = (i / total_tasks) * 100
                self.update_workflow_progress(workflow_id, progress)

                # Build inputs from upstream nodes
                inputs = self._get_node_inputs(
                    node_id, workflow.connections, node_outputs
                )

                # Execute node synchronously
                result = self._execute_single_task(task, inputs)

                if result.get("status") == "error":
                    workflow.status = "failed"
                    workflow.completed_at = datetime.now(timezone.utc)
                    workflow.progress = self.calculate_workflow_progress(workflow_id)
                    if workflow.started_at:
                        workflow.execution_time = (
                            workflow.completed_at - workflow.started_at
                        ).total_seconds()
                    self.db.commit()
                    # Cleanup temporary datasets for this workflow
                    dataset_service.cleanup_temporary_datasets(workflow_id)
                    return result

                # Stash outputs for downstream nodes
                node_outputs[node_id] = result.get("outputs", {}) or {}

            # Finished all nodes successfully
            workflow.status = "completed"
            workflow.progress = 100.0
            workflow.completed_at = datetime.now(timezone.utc)
            workflow.results = {"final_outputs": node_outputs}

            if workflow.started_at:
                workflow.execution_time = (
                    workflow.completed_at - workflow.started_at
                ).total_seconds()

            self.db.commit()

            # Cleanup temporary datasets at the end
            dataset_service.cleanup_temporary_datasets(workflow_id)

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
                workflow.progress = self.calculate_workflow_progress(workflow_id)
                if workflow.started_at:
                    workflow.execution_time = (
                        workflow.completed_at - workflow.started_at
                    ).total_seconds()
                self.db.commit()
                # Best-effort cleanup
                try:
                    dataset_service.cleanup_temporary_datasets(workflow_id)
                except Exception:
                    pass

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
        """Dispatch workflow execution to Celery, task-per-node."""
        dataset_service = DatasetService()
        try:
            workflow = (
                self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
            )
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow.status = "running"
            workflow.started_at = datetime.now(timezone.utc)
            workflow.progress = 0.0
            self.db.commit()

            tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
            dependency_graph = self._build_dependency_graph(
                workflow.nodes, workflow.connections
            )
            execution_order = self._get_execution_order(dependency_graph)

            for node_id in execution_order:
                task = next((t for t in tasks if t.node_id == node_id), None)
                if not task:
                    continue
                # Enqueue Celery task
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
                # Cleanup if we failed before dispatch finished
                try:
                    dataset_service.cleanup_temporary_datasets(workflow_id)
                except Exception:
                    pass

            return {
                "status": "error",
                "message": str(e),
                "workflow_id": workflow_id,
            }

    # -----------------------------
    # Progress + Status
    # -----------------------------
    def update_workflow_progress(
        self, workflow_id: int, progress: float, status: Optional[str] = None
    ):
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if workflow:
            workflow.progress = float(min(100.0, max(0.0, progress)))
            if status:
                workflow.status = status
            self.db.commit()

    def calculate_workflow_progress(self, workflow_id: int) -> float:
        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
        if not tasks:
            return 0.0
        completed = len([t for t in tasks if t.status == "completed"])
        failed = len([t for t in tasks if t.status == "failed"])
        return ((completed + failed) / len(tasks)) * 100.0

    def get_workflow_status(self, workflow_id: int) -> Dict[str, Any]:
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            return {"status": "error", "message": "Workflow not found"}

        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
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

    # -----------------------------
    # New Methods with Pydantic Return Types
    # -----------------------------
    def list_workflows(self) -> List[WorkflowListItem]:
        workflows = self.db.query(Workflow).order_by(Workflow.created_at.desc()).all()
        return [WorkflowListItem.model_validate(wf) for wf in workflows]

    def get_workflow_progress(self, workflow_id: int):
        with SessionLocal() as db:
            workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            return WorkflowProgressResponse(
                workflow_id=workflow.id,
                status=workflow.status,
                progress=workflow.progress or 0.0,
                results=workflow.results,
            )

    def get_workflow_results(self, workflow_id: int) -> WorkflowOutputResponse:
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return WorkflowOutputResponse.model_validate(workflow)

    # -----------------------------
    # Private helpers (added)
    # -----------------------------
    def _build_dependency_graph(
        self, nodes: List[Dict[str, Any]], connections: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Build a graph mapping each node_id -> list of its direct predecessors.
        If a node has no incoming edges, it will still appear with [].
        """
        node_ids = {n["id"] for n in nodes} if nodes else set()
        graph: Dict[str, List[str]] = {nid: [] for nid in node_ids}

        for conn in connections or []:
            src = conn.get("source")
            tgt = conn.get("target")
            if src is None or tgt is None:
                continue
            if tgt not in graph:
                graph[tgt] = []
            # record predecessor
            graph[tgt].append(src)
            # ensure source appears in graph even if no incoming edges
            if src not in graph:
                graph[src] = []

        return graph

    def _get_execution_order(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Topological sort (Kahn's algorithm).
        Raises ValueError if there is a cycle.
        """
        # Build indegree
        indegree: Dict[str, int] = {n: 0 for n in graph}
        for tgt, preds in graph.items():
            for p in preds:
                indegree[tgt] += 1

        queue = [n for n, d in indegree.items() if d == 0]
        order: List[str] = []

        # Build adjacency from predecessor list
        adjacency: Dict[str, List[str]] = {n: [] for n in graph}
        for tgt, preds in graph.items():
            for p in preds:
                adjacency.setdefault(p, []).append(tgt)

        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adjacency.get(n, []):
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)

        if len(order) != len(graph):
            raise ValueError("Cycle detected in workflow graph")

        return order

    def _get_node_inputs(
        self,
        node_id: str,
        connections: List[Dict[str, Any]],
        node_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Gather inputs for a node by merging outputs of its immediate predecessors.
        Later predecessors override earlier keys if duplicated.
        """
        inputs: Dict[str, Any] = {}
        for conn in connections or []:
            if conn.get("target") == node_id:
                src = conn.get("source")
                if src and src in node_outputs:
                    # Merge outputs dict
                    inputs.update(node_outputs[src] or {})
        return inputs

    def _execute_single_task(
        self, task: Task, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronously execute a single task using the ML node registry.
        Updates the Task row with status, timing, result, etc.
        """
        import json
        import time

        try:
            # Mark running
            task.status = "running"
            task.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            node_class = NODE_CLASSES.get(task.task_type)
            if not node_class:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Instantiate node with (node_id, params, optional inputs)
            node = node_class(task.node_id, task.parameters or {}, inputs=inputs)

            start_time = time.time()
            result: Dict[str, Any] = node.execute()
            exec_time = time.time() - start_time

            # Persist outcome to DB
            # Store JSON-serializable result; if not serializable, fall back to str(message)
            safe_result: Dict[str, Any]
            try:
                json.dumps(result)  # test serializability
                safe_result = result
            except Exception:
                safe_result = {
                    "status": result.get("status", "error"),
                    "message": str(result),
                    "outputs": {},
                }

            task.result = safe_result
            task.execution_time = exec_time
            task.status = (
                "completed" if safe_result.get("status") == "success" else "failed"
            )
            task.error_message = (
                safe_result.get("message")
                if safe_result.get("status") == "error"
                else None
            )
            task.updated_at = datetime.now(timezone.utc)
            self.db.commit()

            return safe_result

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            return {"status": "error", "message": str(e)}
