# Flowza Day 2: Workflow Execution Engine

## Prerequisites Check
Before starting Day 2, ensure you have completed Day 1 setup:
- [ ] FastAPI server running on localhost:8000
- [ ] Docker services (PostgreSQL, Redis) running
- [ ] Celery worker process active
- [ ] Sample CSV uploaded and tested
- [ ] All Day 1 dependencies installed

## Step 1: Create Workflow Execution Service

### 1.1 Create the WorkflowService class
Create `backend/app/services/workflow_service.py`:

```python
import logging
from typing import Any, Dict, List
from datetime import datetime

from app.database.connection import SessionLocal
from app.models.task import Task
from app.models.workflow import Workflow
from app.services.dataset_service import DatasetService
from app.services.task_executor import NODE_CLASSES
from sqlalchemy.orm import Session  # noqa :

# 🔹 Added imports
from app.utils.error_handler import handle_node_error, validate_workflow_definition, WorkflowError  # noqa : E501

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing workflow execution"""

    def __init__(self):
        self.db = SessionLocal()

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()

    def create_workflow_from_definition(
        self, workflow_def: Dict[str, Any]
    ) -> Workflow:
        """Create a workflow from JSON definition with validation"""
        try:
            # 🔹 Validate workflow definition
            validation_result = validate_workflow_definition(workflow_def)
            if not validation_result["valid"]:
                raise WorkflowError(f"Invalid workflow definition: {validation_result['errors']}")  # noqa : E501

            # 🔹 Log warnings if any
            if validation_result["warnings"]:
                logger.warning(f"Workflow warnings: {validation_result['warnings']}")  # noqa : E501

            # Create workflow record
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

            # Create task records
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
                f"Created workflow {workflow.id} with "
                f"{len(workflow_def.get('nodes', []))} tasks"
            )

            return workflow

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create workflow: {str(e)}")
            raise

    def update_workflow_progress(self, workflow_id: int, progress: float, status: str = None):  # noqa : E501
        """Update workflow progress and status"""
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
        if workflow:
            workflow.progress = min(100.0, max(0.0, progress))  # Clamp between 0-100  # noqa : E501
            if status:
                workflow.status = status
            self.db.commit()

    def calculate_workflow_progress(self, workflow_id: int) -> float:
        """Calculate workflow progress based on task completion"""
        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501

        if not tasks:
            return 0.0

        completed_tasks = len([t for t in tasks if t.status == "completed"])
        failed_tasks = len([t for t in tasks if t.status == "failed"])
        total_tasks = len(tasks)

        # Completed + failed both count as "done"
        progress = ((completed_tasks + failed_tasks) / total_tasks) * 100
        return progress

    def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Execute a workflow by processing nodes in dependency order"""
        try:
            workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()  # noqa : E501
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")

            # Initialize workflow execution
            workflow.status = "running"
            workflow.started_at = datetime.utcnow()
            workflow.progress = 0.0
            self.db.commit()

            # Fetch tasks
            tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()  # noqa : E501

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(workflow.nodes, workflow.connections)  # noqa : E501
            execution_order = self._get_execution_order(dependency_graph)

            node_outputs = {}
            total_tasks = len(execution_order)

            for i, node_id in enumerate(execution_order):
                task = next((t for t in tasks if t.node_id == node_id), None)
                if not task:
                    continue

                logger.info(f"Executing node {node_id} ({i+1}/{total_tasks})")

                # Update workflow progress
                progress = (i / total_tasks) * 100
                self.update_workflow_progress(workflow_id, progress)

                # Gather inputs
                inputs = self._get_node_inputs(node_id, workflow.connections, node_outputs)  # noqa : E501

                # Run task
                result = self._execute_single_task(task, inputs)

                if result.get("status") == "error":
                    workflow.status = "failed"
                    workflow.completed_at = datetime.utcnow()
                    workflow.progress = self.calculate_workflow_progress(workflow_id)  # noqa : E501
                    self.db.commit()
                    return result

                # Save outputs
                node_outputs[node_id] = result.get("outputs", {})

            # Finalize workflow
            workflow.status = "completed"
            workflow.progress = 100.0
            workflow.completed_at = datetime.utcnow()
            workflow.results = {"final_outputs": node_outputs}

            if workflow.started_at:
                execution_time = (workflow.completed_at - workflow.started_at).total_seconds()  # noqa : E501
                workflow.execution_time = execution_time

            self.db.commit()

            return {
                "status": "success",
                "message": f"Workflow {workflow_id} completed successfully",
                "workflow_id": workflow_id,
                "execution_time": workflow.execution_time,
                "outputs": node_outputs,
            }

        except Exception as e:
            if "workflow" in locals():
                workflow.status = "failed"
                workflow.completed_at = datetime.utcnow()
                workflow.progress = self.calculate_workflow_progress(workflow_id)  # noqa : E501
                self.db.commit()

            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "workflow_id": workflow_id,
            }

    def _build_dependency_graph(
        self, nodes: List[Dict], connections: List[Dict]
    ) -> Dict[str, List[str]]:
        """Build a dependency graph from workflow connections"""
        graph = {node["id"]: [] for node in nodes}

        for connection in connections:
            source = connection["source"]
            target = connection["target"]
            if target in graph:
                graph[target].append(source)

        return graph

    def _get_execution_order(
        self, dependency_graph: Dict[str, List[str]]
    ) -> List[str]:
        """Topological sort of nodes"""
        visited = set()
        temp_visited = set()
        result = []

        def visit(node):
            if node in temp_visited:
                raise ValueError("Circular dependency detected")
            if node in visited:
                return

            temp_visited.add(node)
            for dependency in dependency_graph.get(node, []):
                visit(dependency)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)

        for node in dependency_graph:
            if node not in visited:
                visit(node)

        return result

    def _get_node_inputs(
        self, node_id: str, connections: List[Dict], node_outputs: Dict
    ) -> Dict[str, Any]:
        """Resolve node inputs from outputs of connected nodes"""
        inputs = {}
        dataset_service = DatasetService()

        for connection in connections:
            if connection["target"] == node_id:
                source_node = connection["source"]
                if source_node in node_outputs:
                    source_outputs = node_outputs[source_node]

                    if (
                        "dataset_ids" in source_outputs
                        and "dataset" in source_outputs["dataset_ids"]
                    ):
                        dataset_id = source_outputs["dataset_ids"]["dataset"]
                        inputs["dataset"] = dataset_service.load_dataset(dataset_id)  # noqa : E501
                    elif "dataset" in source_outputs:
                        inputs["dataset"] = source_outputs["dataset"]

        return inputs

    def _execute_single_task(
        self, task: Task, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task with given inputs and comprehensive error handling"""  # noqa : E501
        try:
            # Update task status
            task.status = "running"
            self.db.commit()

            # Get node class
            node_class = NODE_CLASSES.get(task.task_type)
            if not node_class:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Create and configure node
            node = node_class(task.node_id, task.parameters)

            # Set inputs
            for input_name, input_data in inputs.items():
                if input_data is not None:
                    node.set_input(input_name, input_data)

            # Execute node
            result = node.execute()

            if result.get("status") == "success":
                dataset_service = DatasetService()
                saved_outputs = {}

                if node.get_output("dataset") is not None:
                    dataset = dataset_service.save_dataset(
                        dataframe=node.get_output("dataset"),
                        name=f"{task.task_type}_{task.node_id}",
                        workflow_id=task.workflow_id,
                        node_id=task.node_id,
                        description=f"Output from {task.task_type} node",
                    )
                    saved_outputs["dataset"] = dataset.id

                if node.get_output("train_dataset") is not None:
                    train_dataset = dataset_service.save_dataset(
                        dataframe=node.get_output("train_dataset"),
                        name=f"train_{task.node_id}",
                        workflow_id=task.workflow_id,
                        node_id=task.node_id,
                        dataset_type="train",
                        description=f"Training set from {task.task_type} node",
                    )
                    saved_outputs["train_dataset"] = train_dataset.id

                if node.get_output("test_dataset") is not None:
                    test_dataset = dataset_service.save_dataset(
                        dataframe=node.get_output("test_dataset"),
                        name=f"test_{task.node_id}",
                        workflow_id=task.workflow_id,
                        node_id=task.node_id,
                        dataset_type="test",
                        description=f"Test set from {task.task_type} node",
                    )
                    saved_outputs["test_dataset"] = test_dataset.id

                # Update task with success results
                task.result = result
                task.output_data = saved_outputs
                task.status = "completed"
                task.error_message = None
            else:
                # Handle node execution error
                task.result = result
                task.status = "failed"
                task.error_message = result.get("message", "Unknown error")

            self.db.commit()

            return {
                "status": result.get("status"),
                "message": result.get("message"),
                "outputs": {
                    "dataset": node.get_output("dataset"),
                    "encoders": node.get_output("encoders"),
                    "scaler": node.get_output("scaler"),
                    "train_dataset": node.get_output("train_dataset"),
                    "test_dataset": node.get_output("test_dataset"),
                    "dataset_ids": saved_outputs if "saved_outputs" in locals() else {},  # noqa : E501
                },
            }

        except Exception as e:
            # 🔹 Handle execution exception using error handler
            error_info = handle_node_error(e, task.node_id, task.task_type)

            task.status = "failed"
            task.error_message = error_info["user_message"]
            task.result = error_info
            self.db.commit()

            return {
                "status": "error",
                "message": error_info["user_message"],
                "error_details": error_info,
                "outputs": {},
            }

    def get_workflow_status(self, workflow_id: int) -> Dict[str, Any]:
        """Get workflow status with tasks info"""
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

    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Return predefined workflow templates"""
        templates = [
            {
                "id": "simple_csv",
                "name": "Simple CSV Cleaning",
                "nodes": [
                    {
                        "id": "load_csv",
                        "type": "csv_loader",
                        "position": {"x": 100, "y": 100},
                        "data": {"label": "Load CSV"},
                        "parameters": {"file_path": "datasets/sample_data.csv"},  # noqa : E501
                    },
                    {
                        "id": "drop_nulls",
                        "type": "drop_nulls",
                        "position": {"x": 300, "y": 100},
                        "data": {"label": "Drop Nulls"},
                        "parameters": {"how": "any"},
                    },
                ],
                "connections": [{"source": "load_csv", "target": "drop_nulls"}],  # noqa : E501
            },
            {
                "id": "classification_pipeline",
                "name": "Complete Classification Pipeline",
                "description": "CSV → Preprocess → Train/Test Split → Train Model → Evaluate",  # noqa : E501
                "nodes": [
                    {
                        "id": "load_csv",
                        "type": "csv_loader",
                        "position": {"x": 100, "y": 100},
                        "data": {"label": "Load CSV"},
                        "parameters": {"file_path": "datasets/iris.csv"},
                    },
                    {
                        "id": "preprocess",
                        "type": "preprocess",
                        "position": {"x": 300, "y": 100},
                        "data": {"label": "Preprocess Data"},
                        "parameters": {
                            "drop_nulls": True,
                            "encode_categorical": True,
                            "normalize": False,
                        },
                    },
                    {
                        "id": "split_data",
                        "type": "train_test_split",
                        "position": {"x": 500, "y": 100},
                        "data": {"label": "Train/Test Split"},
                        "parameters": {
                            "test_size": 0.2,
                            "target_column": "species",
                            "stratify": True,
                        },
                    },
                ],
                "connections": [
                    {"source": "load_csv", "target": "preprocess"},
                    {"source": "preprocess", "target": "split_data"},
                ],
            },
        ]
        return templates
```

### 4.2 Create enhanced schemas
Update `backend/app/schemas/workflow_schemas.py`:

```python
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = {}


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


class WorkflowExecuteRequest(BaseModel):
    name: str
    description: Optional[str] = "Workflow execution request"
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection] = []


class TaskStatus(BaseModel):
    id: int
    node_id: str
    task_type: str
    status: str
    error_message: Optional[str] = None


class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    status: str
    progress: Optional[float] = 0.0
    created_at: datetime
    updated_at: Optional[datetime]
    execution_time: Optional[float] = None

    class Config:
        from_attributes = True


class WorkflowStatusResponse(BaseModel):
    workflow_id: int
    status: str
    name: str
    created_at: datetime
    results: Optional[Dict[str, Any]] = None
    tasks: List[TaskStatus]
```

## Step 5: Add Result Persistence

### 5.1 Create dataset persistence service
Create `backend/app/services/dataset_service.py`:

```python
import logging
import os
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from app.database.connection import SessionLocal
from app.models.dataset import Dataset
from sqlalchemy.orm import Session  # noqa: F401

logger = logging.getLogger(__name__)

# Dataset size limit (default: 100 MB)
MAX_DATASET_SIZE_MB = float(os.getenv("MAX_DATASET_SIZE_MB", 100))


class DatasetService:
    """Service for managing dataset persistence"""

    def __init__(self):
        self.db = SessionLocal()
        self.datasets_dir = "datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        name: str,
        workflow_id: Optional[int] = None,
        node_id: Optional[str] = None,
        dataset_type: str = "general",
        description: Optional[str] = None,
    ) -> Dataset:
        """Save a pandas DataFrame as a dataset with size validation"""
        try:
            # Check dataset size in memory
            memory_usage_mb = dataframe.memory_usage(deep=True).sum() / (1024 * 1024)  # noqa : E501

            if memory_usage_mb > MAX_DATASET_SIZE_MB:
                raise ValueError(
                    f"Dataset too large: {memory_usage_mb:.2f}MB exceeds limit of "  # noqa : E501
                    f"{MAX_DATASET_SIZE_MB}MB"
                )

            # Validate DataFrame
            if dataframe.empty:
                raise ValueError("Cannot save empty dataset")

            if len(dataframe.columns) == 0:
                raise ValueError("Dataset must have at least one column")

            # Generate unique filename
            file_id = str(uuid.uuid4())[:8]
            filename = f"{name}_{file_id}.csv"
            file_path = os.path.join(self.datasets_dir, filename)

            # Save DataFrame to CSV
            dataframe.to_csv(file_path, index=False)

            # Calculate file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Analyze dataset
            columns_info = {
                "columns": list(dataframe.columns),
                "dtypes": dataframe.dtypes.astype(str).to_dict(),
                "null_counts": dataframe.isnull().sum().to_dict(),
                "memory_usage_mb": memory_usage_mb,
                "numeric_columns": list(
                    dataframe.select_dtypes(include=["number"]).columns
                ),
                "categorical_columns": list(
                    dataframe.select_dtypes(include=["object"]).columns
                ),
                "sample_values": {
                    col: dataframe[col].head(3).tolist() for col in dataframe.columns  # noqa : E501
                },
            }

            # Create dataset record
            dataset = Dataset(
                name=name,
                file_path=file_path,
                workflow_id=workflow_id,
                node_id=node_id,
                format="csv",
                size_mb=size_mb,
                num_rows=len(dataframe),
                num_columns=len(dataframe.columns),
                columns_info=columns_info,
                description=description,
                source="processed",
                dataset_type=dataset_type,
                is_temporary=workflow_id is not None,
            )

            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)

            logger.info(
                f"Saved dataset {dataset.id}: {file_path} ({size_mb:.2f} MB)"
            )

            return dataset

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save dataset: {str(e)}")
            raise

    def load_dataset(self, dataset_id: int) -> pd.DataFrame:
        """Load a dataset by ID"""
        dataset = (
            self.db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        if not os.path.exists(dataset.file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset.file_path}"
            )

        return pd.read_csv(dataset.file_path)

    def get_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        """Get dataset metadata"""
        dataset = (
            self.db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        return {
            "id": dataset.id,
            "name": dataset.name,
            "file_path": dataset.file_path,
            "size_mb": dataset.size_mb,
            "num_rows": dataset.num_rows,
            "num_columns": dataset.num_columns,
            "columns_info": dataset.columns_info,
            "created_at": dataset.created_at,
            "dataset_type": dataset.dataset_type,
        }

    def cleanup_temporary_datasets(self, workflow_id: int):
        """Clean up temporary datasets for a workflow"""
        datasets = (
            self.db.query(Dataset)
            .filter(
                Dataset.workflow_id == workflow_id,
                Dataset.is_temporary.is_(True),
            )
            .all()
        )

        for dataset in datasets:
            try:
                if os.path.exists(dataset.file_path):
                    os.remove(dataset.file_path)
                self.db.delete(dataset)
                logger.info(f"Cleaned up temporary dataset {dataset.id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup dataset {dataset.id}: {str(e)}"
                )

        self.db.commit()
```

### 5.2 Update workflow service to use dataset persistence
Update `backend/app/services/workflow_service.py` to include dataset persistence:

```python
# Add this import at the top
from app.services.dataset_service import DatasetService

# Update the _execute_single_task method
def _execute_single_task(self, task: Task, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task with given inputs"""
    try:
        # Update task status
        task.status = 'running'
        self.db.commit()
        
        # Get node class
        node_class = NODE_CLASSES.get(task.task_type)
        if not node_class:
            raise ValueError(f"Unknown task type: {task.task_type}")
        
        # Create and configure node
        node = node_class(task.node_id, task.parameters)
        
        # Set inputs
        for input_name, input_data in inputs.items():
            if input_data is not None:
                node.set_input(input_name, input_data)
        
        # Execute node
        result = node.execute()
        
        # Save output datasets
        dataset_service = DatasetService()
        saved_outputs = {}
        
        if result.get('status') == 'success':
            # Save main dataset output
            if node.get_output('dataset') is not None:
                dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('dataset'),
                    name=f"{task.task_type}_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    description=f"Output from {task.task_type} node"
                )
                saved_outputs['dataset'] = dataset.id
            
            # Save train/test datasets if they exist
            if node.get_output('train_dataset') is not None:
                train_dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('train_dataset'),
                    name=f"train_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    dataset_type='train',
                    description=f"Training set from {task.task_type} node"
                )
                saved_outputs['train_dataset'] = train_dataset.id
            
            if node.get_output('test_dataset') is not None:
                test_dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('test_dataset'),
                    name=f"test_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    dataset_type='test',
                    description=f"Test set from {task.task_type} node"
                )
                saved_outputs['test_dataset'] = test_dataset.id
        
        # Update task with results
        task.result = result
        task.output_data = saved_outputs
        task.status = 'completed' if result.get('status') == 'success' else 'failed'
        task.error_message = result.get('message') if result.get('status') == 'error' else None
        
        self.db.commit()
        
        # Return result with dataset IDs instead of actual data
        return {
            'status': result.get('status'),
            'message': result.get('message'),
            'outputs': {
                'dataset': node.get_output('dataset'),
                'encoders': node.get_output('encoders'),
                'scaler': node.get_output('scaler'),
                'dataset_ids': saved_outputs
            }
        }
        
    except Exception as e:
        task.status = 'failed'
        task.error_message = str(e)
        self.db.commit()
        
        return {
            'status': 'error',
            'message': str(e),
            'outputs': {}
        }

# Update _get_node_inputs method to load datasets by ID
def _get_node_inputs(self, node_id: str, connections: List[Dict], node_outputs: Dict) -> Dict[str, Any]:
    """Get inputs for a node from previous node outputs"""
    inputs = {}
    dataset_service = DatasetService()
    
    for connection in connections:
        if connection['target'] == node_id:
            source_node = connection['source']
            if source_node in node_outputs:
                source_outputs = node_outputs[source_node]
                
                # Load dataset from saved ID if available
                if 'dataset_ids' in source_outputs and 'dataset' in source_outputs['dataset_ids']:
                    dataset_id = source_outputs['dataset_ids']['dataset']
                    inputs['dataset'] = dataset_service.load_dataset(dataset_id)
                # Fallback to direct dataset if still in memory
                elif 'dataset' in source_outputs:
                    inputs['dataset'] = source_outputs['dataset']
    
    return inputs
```

## Step 6: Enhanced Task Router

### 6.1 Update task router with better node definitions
Update `backend/app/routers/tasks.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.task import Task
from app.models.dataset import Dataset
from app.services.dataset_service import DatasetService
from typing import List
import pandas as pd

router = APIRouter()


@router.get("/available")
async def get_available_ml_nodes():
    """Get all available ML nodes/tasks"""
    return {
        "data_nodes": [
            {
                "type": "csv_loader",
                "name": "CSV Loader",
                "category": "data_input",
                "description": "Load data from CSV file",
                "inputs": [],
                "outputs": ["dataset"],
                "parameters": {
                    "file_path": {
                        "type": "string", 
                        "required": True,
                        "description": "Path to the CSV file"
                    },
                    "separator": {
                        "type": "string", 
                        "default": ",",
                        "description": "Column separator"
                    },
                    "encoding": {
                        "type": "string", 
                        "default": "utf-8",
                        "description": "File encoding"
                    }
                }
            }
        ],
        "preprocessing_nodes": [
            {
                "type": "drop_nulls",
                "name": "Drop Null Values",
                "category": "preprocessing",
                "description": "Remove rows with null values",
                "inputs": ["dataset"],
                "outputs": ["dataset"],
                "parameters": {
                    "subset": {
                        "type": "array",
                        "default": None,
                        "description": "Columns to check for nulls (null = check all)"
                    },
                    "how": {
                        "type": "string",
                        "default": "any",
                        "description": "Remove rows with 'any' or 'all' nulls"
                    }
                }
            },
            {
                "type": "preprocess",
                "name": "Data Preprocessing",
                "category": "preprocessing",
                "description": "Comprehensive data cleaning and preprocessing",
                "inputs": ["dataset"],
                "outputs": ["dataset"],
                "parameters": {
                    "drop_nulls": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Remove rows with null values"
                    },
                    "normalize": {
                        "type": "boolean", 
                        "default": False,
                        "description": "Normalize numerical features"
                    },
                    "encode_categorical": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Encode categorical variables"
                    },
                    "columns_to_drop": {
                        "type": "array",
                        "default": [],
                        "description": "Columns to remove from dataset"
                    }
                }
            },
            {
                "type": "train_test_split",
                "name": "Train/Test Split",
                "category": "preprocessing",
                "description": "Split dataset into training and testing sets",
                "inputs": ["dataset"],
                "outputs": ["train_dataset", "test_dataset"],
                "parameters": {
                    "test_size": {
                        "type": "float", 
                        "default": 0.2,
                        "description": "Proportion of data for testing"
                    },
                    "random_state": {
                        "type": "integer", 
                        "default": 42,
                        "description": "Random seed for reproducibility"
                    },
                    "target_column": {
                        "type": "string",
                        "default": None,
                        "description": "Target column for stratification"
                    },
                    "stratify": {
                        "type": "boolean", 
                        "default": False,
                        "description": "Stratify split by target column"
                    }
                }
            }
        ],
        "training_nodes": [
            {
                "type": "train_logreg",
                "name": "Logistic Regression",
                "category": "training",
                "description": "Train a logistic regression model",
                "inputs": ["train_dataset"],
                "outputs": ["model"],
                "parameters": {
                    "C": {
                        "type": "float", 
                        "default": 1.0,
                        "description": "Regularization strength"
                    },
                    "max_iter": {
                        "type": "integer", 
                        "default": 1000,
                        "description": "Maximum iterations"
                    },
                    "random_state": {
                        "type": "integer", 
                        "default": 42,
                        "description": "Random seed"
                    }
                }
            }
        ],
        "evaluation_nodes": [
            {
                "type": "evaluate",
                "name": "Model Evaluation",
                "category": "evaluation",
                "description": "Evaluate model performance",
                "inputs": ["model", "test_dataset"],
                "outputs": ["metrics"],
                "parameters": {
                    "metrics": {
                        "type": "array", 
                        "default": ["accuracy", "f1_score", "precision", "recall"],
                        "description": "Metrics to calculate"
                    }
                }
            }
        ]
    }


@router.get("/{task_id}")
async def get_task_status(task_id: int, db: Session = Depends(get_db)):
    """Get task execution status"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "id": task.id,
        "node_id": task.node_id,
        "task_type": task.task_type,
        "status": task.status,
        "result": task.result,
        "error_message": task.error_message,
        "execution_time": task.execution_time,
        "output_data": task.output_data
    }


@router.get("/{task_id}/dataset")
async def get_task_output_dataset(task_id: int, db: Session = Depends(get_db)):
    """Get dataset output from a task"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not task.output_data or 'dataset' not in task.output_data:
        raise HTTPException(status_code=404, detail="No dataset output found")
    
    try:
        dataset_service = DatasetService()
        dataset_id = task.output_data['dataset']
        
        # Get dataset info
        dataset_info = dataset_service.get_dataset_info(dataset_id)
        
        # Load and preview dataset
        df = dataset_service.load_dataset(dataset_id)
        preview = {
            "shape": df.shape,
            "columns": list(df.columns),
            "head": df.head(10).to_dict('records'),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        return {
            "dataset_info": dataset_info,
            "preview": preview
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/")
async def list_datasets(db: Session = Depends(get_db), workflow_id: int = None):
    """List all datasets"""
    query = db.query(Dataset)
    if workflow_id:
        query = query.filter(Dataset.workflow_id == workflow_id)
    
    datasets = query.all()
    
    return [
        {
            "id": d.id,
            "name": d.name,
            "size_mb": d.size_mb,
            "num_rows": d.num_rows,
            "num_columns": d.num_columns,
            "dataset_type": d.dataset_type,
            "created_at": d.created_at,
            "workflow_id": d.workflow_id,
            "node_id": d.node_id
        }
        for d in datasets
    ]
```
## Step 7: Testing the End-to-End Workflow

### 7.1 Start all services
```bash
# Terminal 1: Start Docker services
docker-compose up -d

# Terminal 2: Start FastAPI server
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start Celery worker
cd backend
source venv/bin/activate
celery -A app.services.task_executor.celery worker --loglevel=info
```

### 7.2 Create test datasets
```bash
# Create a sample dataset with some null values
cat > backend/datasets/test_data.csv << EOF
name,age,salary,department,target
John,25,50000,Engineering,1
Jane,,60000,Marketing,1
Bob,30,,Engineering,0
Alice,28,55000,,1
Charlie,35,70000,Engineering,1
Diana,29,58000,Marketing,0
Eve,,62000,Engineering,1
Frank,32,48000,Marketing,0
Grace,26,52000,Engineering,1
Henry,31,,Marketing,0
EOF
```

### 7.3 Test basic workflow execution
```bash
# Test 1: Simple CSV load and drop nulls
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Drop Nulls Pipeline",
    "description": "Load CSV and drop null values",
    "nodes": [
      {
        "id": "load_csv",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load CSV"},
        "parameters": {"file_path": "datasets/test_data.csv"}
      },
      {
        "id": "drop_nulls",
        "type": "drop_nulls",
        "position": {"x": 300, "y": 100},
        "data": {"label": "Drop Nulls"},
        "parameters": {"how": "any"}
      }
    ],
    "connections": [
      {"source": "load_csv", "target": "drop_nulls"}
    ]
  }'

# Save the workflow_id from the response for checking status
```

### 7.4 Check workflow status and results
```bash
# Replace {workflow_id} with actual ID from previous response
curl "http://localhost:8000/api/workflows/{workflow_id}/status"

# Get workflow results
curl "http://localhost:8000/api/workflows/{workflow_id}/results"

# List all datasets created
curl "http://localhost:8000/api/tasks/datasets/"
```

### 7.5 Test complete preprocessing pipeline
```bash
# Test 2: Complete preprocessing pipeline
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Complete Preprocessing Pipeline",
    "description": "CSV → Preprocess → Train/Test Split",
    "nodes": [
      {
        "id": "load_csv",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load CSV"},
        "parameters": {"file_path": "datasets/test_data.csv"}
      },
      {
        "id": "preprocess",
        "type": "preprocess",
        "position": {"x": 300, "y": 100},
        "data": {"label": "Preprocess"},
        "parameters": {
          "drop_nulls": true,
          "encode_categorical": true,
          "normalize": false,
          "columns_to_drop": []
        }
      },
      {
        "id": "split_data",
        "type": "train_test_split",
        "position": {"x": 500, "y": 100},
        "data": {"label": "Train/Test Split"},
        "parameters": {
          "test_size": 0.3,
          "target_column": "target",
          "stratify": true,
          "random_state": 42
        }
      }
    ],
    "connections": [
      {"source": "load_csv", "target": "preprocess"},
      {"source": "preprocess", "target": "split_data"}
    ]
  }'
```

### 7.6 Test error handling
```bash
# Test 3: Error handling with invalid file path
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Error Test",
    "description": "Test error handling",
    "nodes": [
      {
        "id": "load_csv",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load CSV"},
        "parameters": {"file_path": "datasets/nonexistent_file.csv"}
      }
    ],
    "connections": []
  }'
```

## Step 8: Create Development Scripts

### 8.1 Create comprehensive test script
Create `backend/scripts/test_workflows.py`:

```python
#!/usr/bin/env python3
"""
Test script for Flowza workflow execution
"""

import time
import requests

BASE_URL = "http://localhost:8000"


def create_and_run_workflow(workflow_def: dict):
    """Helper: create workflow, execute, and track progress"""
    # Step 1: Create workflow
    create_resp = requests.post(f"{BASE_URL}/api/workflows/", json=workflow_def)  # noqa : 
    if create_resp.status_code != 200:
        print("❌ Failed to create workflow")
        print(create_resp.text)
        return None

    workflow_id = create_resp.json().get("id")
    print(f"✅ Workflow created: {workflow_id}")

    # Step 2: Execute workflow
    exec_resp = requests.post(f"{BASE_URL}/api/workflows/{workflow_id}/execute")  # noqa : 
    if exec_resp.status_code != 200:
        print("❌ Failed to execute workflow")
        print(exec_resp.text)
        return None
    print(f"🚀 Workflow {workflow_id} execution started")

    # Step 3: Poll progress
    for i in range(10):  # wait up to 10s
        time.sleep(1)
        prog_resp = requests.get(f"{BASE_URL}/api/workflows/{workflow_id}/progress")  # noqa : 
        if prog_resp.status_code == 200:
            progress = prog_resp.json()
            print(f"📊 Progress: {progress}")
            if progress.get("status") in ["completed", "failed"]:
                break
        else:
            print("❌ Failed to fetch progress")
            break

    return workflow_id


def test_workflow_execution():
    """Run two sample workflows"""

    # Test 1: Simple CSV Load
    print("\n1️⃣ Testing CSV Loader...")
    workflow_csv = {
        "name": "Test CSV Load",
        "description": "Simple CSV loading test",
        "nodes": [
            {
                "id": "load_csv",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Load CSV"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            }
        ],
        "connections": [],
    }
    create_and_run_workflow(workflow_csv)

    # Test 2: Drop Nulls pipeline
    print("\n2️⃣ Testing Drop Nulls Pipeline...")
    workflow_pipeline = {
        "name": "Drop Nulls Pipeline",
        "description": "CSV → Drop Nulls",
        "nodes": [
            {
                "id": "load_csv",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Load CSV"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            },
            {
                "id": "drop_nulls",
                "type": "drop_nulls",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Drop Nulls"},
                "parameters": {"how": "any"},
            },
        ],
        "connections": [{"source": "load_csv", "target": "drop_nulls"}],
    }
    create_and_run_workflow(workflow_pipeline)


if __name__ == "__main__":
    test_workflow_execution()
```

### 8.2 Make test script executable and run it
```bash
# Make script executable
chmod +x backend/scripts/test_workflows.py

# Install requests if not already installed
cd backend
pip install requests

# Run the test
python scripts/test_workflows.py
```

### 8.3 Create workflow monitoring script
Create `backend/scripts/monitor_workflows.py`:

```python
#!/usr/bin/env python3
"""
Monitor running workflows in real-time
"""
import sys  # noqa : 
import time
from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"


def monitor_workflows():
    """Monitor all workflows continuously"""
    print("📊 Flowza Workflow Monitor")
    print("Press Ctrl+C to stop monitoring")

    try:
        while True:
            # Clear screen
            print("\033[2J\033[H")
            print(
                f"Flowza Workflow Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # noqa :
            )
            print("=" * 60)

            # Get all workflows
            response = requests.get(f"{BASE_URL}/api/workflows/")
            if response.status_code == 200:
                workflows = response.json()

                if not workflows:
                    print("No workflows found")
                else:
                    for workflow in workflows:
                        status_icon = {
                            "pending": "⏳",
                            "running": "🔄",
                            "completed": "✅",
                            "failed": "❌",
                            "draft": "📝",
                        }.get(workflow["status"], "❓")

                        print(
                            f"{status_icon} Workflow {workflow['id']}: {workflow['name']}"  # noqa :
                        )
                        print(f"   Status: {workflow['status']}")
                        print(f"   Created: {workflow['created_at']}")
                        print(f"   Nodes: {len(workflow.get('nodes', []))}")
                        print()

            else:
                print("Failed to fetch workflows")

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    monitor_workflows()
```

## Step 9: Add Progress Tracking

### 9.1 Update workflow service with progress tracking
Add to `backend/app/services/workflow_service.py`:

```python
def update_workflow_progress(self, workflow_id: int, progress: float, status: str = None):
    """Update workflow progress and status"""
    workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if workflow:
        workflow.progress = min(100.0, max(0.0, progress))  # Clamp between 0-100
        if status:
            workflow.status = status
        self.db.commit()

def calculate_workflow_progress(self, workflow_id: int) -> float:
    """Calculate workflow progress based on task completion"""
    tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
    
    if not tasks:
        return 0.0
    
    completed_tasks = len([t for t in tasks if t.status == 'completed'])
    failed_tasks = len([t for t in tasks if t.status == 'failed'])
    total_tasks = len(tasks)
    
    # Calculate progress (completed + failed tasks count as "done")
    progress = ((completed_tasks + failed_tasks) / total_tasks) * 100
    return progress

# Update the execute_workflow method to track progress
def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
    """Execute a workflow by processing nodes in dependency order"""
    try:
        workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Update workflow status and start time
        workflow.status = 'running'
        workflow.started_at = datetime.utcnow()
        workflow.progress = 0.0
        self.db.commit()
        
        # Get workflow tasks
        tasks = self.db.query(Task).filter(Task.workflow_id == workflow_id).all()
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(workflow.nodes, workflow.connections)
        
        # Execute tasks in dependency order
        execution_order = self._get_execution_order(dependency_graph)
        node_outputs = {}
        total_tasks = len(execution_order)
        
        for i, node_id in enumerate(execution_order):
            task = next((t for t in tasks if t.node_id == node_id), None)
            if not task:
                continue
            
            logger.info(f"Executing node {node_id} ({i+1}/{total_tasks})")
            
            # Update progress
            progress = (i / total_tasks) * 100
            self.update_workflow_progress(workflow_id, progress)
            
            # Set inputs from previous nodes
            inputs = self._get_node_inputs(node_id, workflow.connections, node_outputs)
            
            # Execute the task
            result = self._execute_single_task(task, inputs)
            
            if result.get('status') == 'error':
                workflow.status = 'failed'
                workflow.completed_at = datetime.utcnow()
                self.db.commit()
                return result
            
            # Store outputs for next nodes
            node_outputs[node_id] = result.get('outputs', {})
        
        # Update workflow status to completed
        workflow.status = 'completed'
        workflow.progress = 100.0
        workflow.completed_at = datetime.utcnow()
        workflow.results = {'final_outputs': node_outputs}
        
        # Calculate total execution time
        if workflow.started_at:
            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
            workflow.execution_time = execution_time
        
        self.db.commit()
        
        return {
            'status': 'success',
            'message': f'Workflow {workflow_id} completed successfully',
            'workflow_id': workflow_id,
            'execution_time': workflow.execution_time,
            'outputs': node_outputs
        }
        
    except Exception as e:
        # Update workflow status to failed
        if 'workflow' in locals():
            workflow.status = 'failed'
            workflow.completed_at = datetime.utcnow()
            workflow.progress = self.calculate_workflow_progress(workflow_id)
            self.db.commit()
        
        logger.error(f"Workflow execution failed: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'workflow_id': workflow_id
        }
```

### 9.2 Add progress endpoint
Add to `backend/app/routers/workflows.py`:

```python
@router.get("/{workflow_id}/progress")
async def get_workflow_progress(workflow_id: int, db: Session = Depends(get_db)):
    """Get workflow progress in real-time"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get task statuses
    from app.models.task import Task
    tasks = db.query(Task).filter(Task.workflow_id == workflow_id).all()
    
    task_statuses = []
    for task in tasks:
        task_statuses.append({
            "node_id": task.node_id,
            "task_type": task.task_type,
            "status": task.status,
            "execution_time": task.execution_time,
            "error_message": task.error_message
        })
    
    return {
        "workflow_id": workflow_id,
        "status": workflow.status,
        "progress": workflow.progress,
        "started_at": workflow.started_at,
        "execution_time": workflow.execution_time,
        "tasks": task_statuses
    }
```

## Step 10: Error Handling and Recovery

### 10.1 Create error handling utilities
Create `backend/app/utils/error_handler.py`:

```python
"""
Error handling utilities for workflow execution
"""
import logging
import traceback
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class WorkflowError(Exception):
    """Base exception for workflow errors"""
    def __init__(self, message: str, node_id: str = None, error_code: str = None):
        self.message = message
        self.node_id = node_id
        self.error_code = error_code
        super().__init__(self.message)

class NodeExecutionError(WorkflowError):
    """Error during node execution"""
    pass

class DataValidationError(WorkflowError):
    """Error during data validation"""
    pass

class DependencyError(WorkflowError):
    """Error in workflow dependencies"""
    pass

def handle_node_error(e: Exception, node_id: str, task_type: str) -> Dict[str, Any]:
    """Handle and format node execution errors"""
    error_info = {
        "status": "error",
        "node_id": node_id,
        "task_type": task_type,
        "error_type": type(e).__name__,
        "message": str(e),
        "timestamp": datetime.utcnow().isoformat(),
        "traceback": traceback.format_exc()
    }
    
    # Log the error
    logger.error(f"Node {node_id} ({task_type}) failed: {str(e)}")
    logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    # Return user-friendly error message
    if isinstance(e, FileNotFoundError):
        error_info["user_message"] = f"File not found. Please check the file path."
        error_info["error_code"] = "FILE_NOT_FOUND"
    elif isinstance(e, pd.errors.EmptyDataError):
        error_info["user_message"] = f"The CSV file appears to be empty."
        error_info["error_code"] = "EMPTY_DATA"
    elif isinstance(e, pd.errors.ParserError):
        error_info["user_message"] = f"Failed to parse the CSV file. Check the format and separator."
        error_info["error_code"] = "PARSE_ERROR"
    elif isinstance(e, KeyError):
        error_info["user_message"] = f"Required column not found in dataset: {str(e)}"
        error_info["error_code"] = "MISSING_COLUMN"
    elif isinstance(e, ValueError):
        error_info["user_message"] = f"Invalid parameter or data: {str(e)}"
        error_info["error_code"] = "INVALID_VALUE"
    else:
        error_info["user_message"] = f"An unexpected error occurred: {str(e)}"
        error_info["error_code"] = "UNKNOWN_ERROR"
    
    return error_info

def validate_workflow_definition(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """Validate workflow definition before execution"""
    errors = []
    warnings = []
    
    nodes = workflow_def.get('nodes', [])
    connections = workflow_def.get('connections', [])
    
    if not nodes:
        errors.append("Workflow must have at least one node")
    
    # Check for duplicate node IDs
    node_ids = [node['id'] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node IDs found")
    
    # Validate connections
    for conn in connections:
        source = conn.get('source')
        target = conn.get('target')
        
        if source not in node_ids:
            errors.append(f"Connection source '{source}' not found in nodes")
        
        if target not in node_ids:
            errors.append(f"Connection target '{target}' not found in nodes")
    
    # Check for circular dependencies
    try:
        from app.services.workflow_service import WorkflowService
        ws = WorkflowService()
        dependency_graph = ws._build_dependency_graph(nodes, connections)
        ws._get_execution_order(dependency_graph)
    except ValueError as e:
        if "Circular dependency" in str(e):
            errors.append("Circular dependency detected in workflow")
    
    # Check for orphaned nodes (nodes with no inputs and no connections)
    connected_nodes = set()
    for conn in connections:
        connected_nodes.add(conn['source'])
        connected_nodes.add(conn['target'])
    
    for node in nodes:
        if node['id'] not in connected_nodes and len(connections) > 0:
            warnings.append(f"Node '{node['id']}' is not connected to any other nodes")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
```

### 10.2 Update workflow service with error handling
Add error handling to `backend/app/services/workflow_service.py`:

```python
# Add import at the top
from app.utils.error_handler import handle_node_error, validate_workflow_definition, WorkflowError

# Update create_workflow_from_definition method
def create_workflow_from_definition(self, workflow_def: Dict[str, Any]) -> Workflow:
    """Create a workflow from JSON definition with validation"""
    try:
        # Validate workflow definition
        validation_result = validate_workflow_definition(workflow_def)
        if not validation_result["valid"]:
            raise WorkflowError(f"Invalid workflow definition: {validation_result['errors']}")
        
        # Log warnings if any
        if validation_result["warnings"]:
            logger.warning(f"Workflow warnings: {validation_result['warnings']}")
        
        # Create workflow record
        workflow = Workflow(
            name=workflow_def.get('name', 'Untitled Workflow'),
            description=workflow_def.get('description', ''),
            nodes=workflow_def.get('nodes', []),
            connections=workflow_def.get('connections', []),
            status='pending'
        )
        
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)
        
        # Create task records for each node
        for node in workflow_def.get('nodes', []):
            task = Task(
                workflow_id=workflow.id,
                node_id=node['id'],
                task_type=node['type'],
                task_name=node.get('data', {}).get('label', node['type']),
                parameters=node.get('parameters', {}),
                status='pending'
            )
            self.db.add(task)
        
        self.db.commit()
        logger.info(f"Created workflow {workflow.id} with {len(workflow_def.get('nodes', []))} tasks")
        
        return workflow
        
    except Exception as e:
        self.db.rollback()
        logger.error(f"Failed to create workflow: {str(e)}")
        raise

# Update _execute_single_task method with better error handling
def _execute_single_task(self, task: Task, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task with given inputs and comprehensive error handling"""
    try:
        # Update task status
        task.status = 'running'
        self.db.commit()
        
        # Get node class
        node_class = NODE_CLASSES.get(task.task_type)
        if not node_class:
            raise ValueError(f"Unknown task type: {task.task_type}")
        
        # Create and configure node
        node = node_class(task.node_id, task.parameters)
        
        # Set inputs
        for input_name, input_data in inputs.items():
            if input_data is not None:
                node.set_input(input_name, input_data)
        
        # Execute node
        result = node.execute()
        
        # Handle successful execution
        if result.get('status') == 'success':
            # Save output datasets
            dataset_service = DatasetService()
            saved_outputs = {}
            
            # Save main dataset output
            if node.get_output('dataset') is not None:
                dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('dataset'),
                    name=f"{task.task_type}_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    description=f"Output from {task.task_type} node"
                )
                saved_outputs['dataset'] = dataset.id
            
            # Save train/test datasets if they exist
            if node.get_output('train_dataset') is not None:
                train_dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('train_dataset'),
                    name=f"train_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    dataset_type='train',
                    description=f"Training set from {task.task_type} node"
                )
                saved_outputs['train_dataset'] = train_dataset.id
            
            if node.get_output('test_dataset') is not None:
                test_dataset = dataset_service.save_dataset(
                    dataframe=node.get_output('test_dataset'),
                    name=f"test_{task.node_id}",
                    workflow_id=task.workflow_id,
                    node_id=task.node_id,
                    dataset_type='test',
                    description=f"Test set from {task.task_type} node"
                )
                saved_outputs['test_dataset'] = test_dataset.id
            
            # Update task with success results
            task.result = result
            task.output_data = saved_outputs
            task.status = 'completed'
            task.error_message = None
        else:
            # Handle node execution error
            task.result = result
            task.status = 'failed'
            task.error_message = result.get('message', 'Unknown error')
        
        self.db.commit()
        
        # Return result with dataset IDs instead of actual data
        return {
            'status': result.get('status'),
            'message': result.get('message'),
            'outputs': {
                'dataset': node.get_output('dataset'),
                'encoders': node.get_output('encoders'),
                'scaler': node.get_output('scaler'),
                'train_dataset': node.get_output('train_dataset'),
                'test_dataset': node.get_output('test_dataset'),
                'dataset_ids': saved_outputs if 'saved_outputs' in locals() else {}
            }
        }
        
    except Exception as e:
        # Handle execution exception
        error_info = handle_node_error(e, task.node_id, task.task_type)
        
        task.status = 'failed'
        task.error_message = error_info['user_message']
        task.result = error_info
        self.db.commit()
        
        return {
            'status': 'error',
            'message': error_info['user_message'],
            'error_details': error_info,
            'outputs': {}
        }
```

## Step 11: Database Migrations

### 11.1 Create database initialization
Create `backend/alembic/env.py`:

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database.connection import Base
from app.models.workflow import Workflow
from app.models.task import Task  
from app.models.dataset import Dataset

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 11.2 Initialize Alembic
```bash
# In backend directory
pip install alembic

# Initialize Alembic
alembic init alembic

# Edit alembic.ini to set the database URL
# sqlalchemy.url = postgresql://flowza:password123@localhost:5433/flowza_db

# Create initial migration
alembic revision --autogenerate -m "Initial tables"

# Run migration
alembic upgrade head
```

## Step 12: Final Testing and Verification

### 12.1 Run comprehensive tests
```bash
# Start all services
docker-compose up -d
cd backend && python -m uvicorn app.main:app --reload &
cd backend && celery -A app.services.task_executor.celery worker --loglevel=info &

# Run test script
python backend/scripts/test_workflows.py
```

### 12.2 Test complex workflow
```bash
# Test a 3-node pipeline
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Complete Data Pipeline",
    "description": "Load → Clean → Split → Analyze",
    "nodes": [
      {
        "id": "load_data",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load Data"},
        "parameters": {"file_path": "datasets/test_data.csv"}
      },
      {
        "id": "clean_data",
        "type": "preprocess",
        "position": {"x": 300, "y": 100},
        "data": {"label": "Clean Data"},
        "parameters": {
          "drop_nulls": true,
          "encode_categorical": true,
          "normalize": false
        }
      },
      {
        "id": "split_data",
        "type": "train_test_split",
        "position": {"x": 500, "y": 100},
        "data": {"label": "Split Data"},
        "parameters": {
          "test_size": 0.25,
          "target_column": "target",
          "stratify": true
        }
      }
    ],
    "connections": [
      {"source": "load_data", "target": "clean_data"},
      {"source": "clean_data", "target": "split_data"}
    ]
  }'
```

### 12.3 Verify database persistence
```bash
# Check that workflows and tasks are saved
curl "http://localhost:8000/api/workflows/"

# Check that datasets are created
curl "http://localhost:8000/api/tasks/datasets/"

# Verify specific workflow results
curl "http://localhost:8000/api/workflows/1/results"
```

### 12.4 Test error scenarios
```bash
# Test with missing file
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Error Test",
    "nodes": [
      {
        "id": "load_fail",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load Fail"},
        "parameters": {"file_path": "datasets/missing.csv"}
      }
    ],
    "connections": []
  }'

# Test circular dependency
curl -X POST "http://localhost:8000/api/workflows/run" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Circular Test",
    "nodes": [
      {
        "id": "node1",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Node 1"},
        "parameters": {"file_path": "datasets/test_data.csv"}
      },
      {
        "id": "node2",
        "type": "drop_nulls",
        "position": {"x": 300, "y": 100},
        "data": {"label": "Node 2"},
        "parameters": {}
      }
    ],
    "connections": [
      {"source": "node1", "target": "node2"},
      {"source": "node2", "target": "node1"}
    ]
  }'
```

## Step 13: Performance Optimization

### 13.1 Add connection pooling and caching
Update `backend/app/database/connection.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Configure connection pool for better performance
engine = create_engine(
    DATABASE_URL, 
    echo=True if os.getenv("DEBUG") == "True" else False,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 13.2 Add dataset size limits and validation
Update `backend/app/services/dataset_service.py`:

```python
# Add at the top
MAX_DATASET_SIZE_MB = float(os.getenv('MAX_DATASET_SIZE_MB', 100))

def save_dataset(
    self,
    dataframe: pd.DataFrame,
    name: str,
    workflow_id: Optional[int] = None,
    node_id: Optional[str] = None,
    dataset_type: str = 'general',
    description: str = None
) -> Dataset:
    """Save a pandas DataFrame as a dataset with size validation"""
    try:
        # Check dataset size before saving
        memory_usage_mb = dataframe.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if memory_usage_mb > MAX_DATASET_SIZE_MB:
            raise ValueError(f"Dataset too large: {memory_usage_mb:.2f}MB exceeds limit of {MAX_DATASET_SIZE_MB}MB")
        
        # Validate DataFrame
        if dataframe.empty:
            raise ValueError("Cannot save empty dataset")
        
        if len(dataframe.columns) == 0:
            raise ValueError("Dataset must have at least one column")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        filename = f"{name}_{file_id}.csv"
        file_path = os.path.join(self.datasets_dir, filename)
        
        # Save DataFrame to CSV
        dataframe.to_csv(file_path, index=False)
        
        # Calculate file size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Analyze dataset
        columns_info = {
            "columns": list(dataframe.columns),
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
            "null_counts": dataframe.isnull().sum().to_dict(),
            "memory_usage_mb": memory_usage_mb,
            "numeric_columns": list(dataframe.select_dtypes(include=['number']).columns),
            "categorical_columns": list(dataframe.select_dtypes(include=['object']).columns),
            "sample_values": {col: dataframe[col].head(3).tolist() for col in dataframe.columns}
        }
        
        # Create dataset record
        dataset = Dataset(
            name=name,
            file_path=file_path,
            workflow_id=workflow_id,
            node_id=node_id,
            format="csv",
            size_mb=size_mb,
            num_rows=len(dataframe),
            num_columns=len(dataframe.columns),
            columns_info=columns_info,
            description=description,
            source="processed",
            dataset_type=dataset_type,
            is_temporary=workflow_id is not None
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        logger.info(f"Saved dataset {dataset.id}: {file_path} ({size_mb:.2f} MB)")
        
        return dataset
        
    except Exception as e:
        self.db.rollback()
        logger.error(f"Failed to save dataset: {str(e)}")
        raise
```

## Step 14: Documentation and Logging

### 14.1 Create API documentation
Create `backend/docs/api_documentation.md`:

```markdown
# Flowza API Documentation

## Workflow Endpoints

### POST /api/workflows/run
Execute a workflow from definition.

**Request Body:**
```json
{
  "name": "My Workflow",
  "description": "Optional description",
  "nodes": [
    {
      "id": "unique_node_id",
      "type": "node_type",
      "position": {"x": 100, "y": 100},
      "data": {"label": "Node Label"},
      "parameters": {"param": "value"}
    }
  ],
  "connections": [
    {"source": "node1", "target": "node2"}
  ]
}
```

**Response:**
```json
{
  "message": "Workflow execution started",
  "workflow_id": 1,
  "task_id": "celery-task-id",
  "status": "running"
}
```

### GET /api/workflows/{workflow_id}/status
Get workflow execution status.

**Response:**
```json
{
  "workflow_id": 1,
  "status": "running",
  "progress": 50.0,
  "tasks": [
    {
      "node_id": "load_csv",
      "status": "completed",
      "execution_time": 2.5
    }
  ]
}
```

### GET /api/workflows/{workflow_id}/results
Get workflow execution results.

**Response:**
```json
{
  "status": "completed",
  "workflow_id": 1,
  "execution_time": 15.2,
  "results": {
    "final_outputs": {
      "node1": {
        "dataset_preview": {
          "shape": [100, 5],
          "columns": ["col1", "col2"],
          "head": [{"col1": 1, "col2": "a"}]
        }
      }
    }
  }
}
```

## Available Node Types

### csv_loader
Load data from CSV file.

**Parameters:**
- `file_path` (required): Path to CSV file
- `separator`: Column separator (default: ",")
- `encoding`: File encoding (default: "utf-8")

### drop_nulls
Remove rows with null values.

**Parameters:**
- `subset`: Columns to check (default: all)
- `how`: "any" or "all" (default: "any")

### preprocess
Comprehensive data preprocessing.

**Parameters:**
- `drop_nulls`: Remove nulls (default: true)
- `encode_categorical`: Encode categorical vars (default: true)
- `normalize`: Normalize numerical features (default: false)
- `columns_to_drop`: List of columns to remove

### train_test_split
Split dataset for training and testing.

**Parameters:**
- `test_size`: Test proportion (default: 0.2)
- `target_column`: Target column name
- `stratify`: Use stratified split (default: false)
- `random_state`: Random seed (default: 42)
```

### 14.2 Add comprehensive logging
Update `backend/app/main.py`:

```python
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_level = logging.INFO if os.getenv("DEBUG") != "True" else logging.DEBUG
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            RotatingFileHandler(
                'logs/flowza.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific loggers
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if os.getenv("DEBUG") == "True" else logging.WARNING
    )
    logging.getLogger("celery").setLevel(logging.INFO)
    
    logger = logging.getLogger("flowza")
    logger.info("Logging configured successfully")

# Add this call before creating the FastAPI app
setup_logging()
```

## Step 15: Cleanup and Finalization

### 15.1 Create cleanup utilities
Create `backend/scripts/cleanup.py`:

```python
#!/usr/bin/env python3
"""
Cleanup script for Flowza development environment
"""
import os
import shutil
import sys
from pathlib import Path

def cleanup_temporary_files():
    """Clean up temporary datasets and logs"""
    print("🧹 Cleaning up temporary files...")
    
    # Clean up temporary datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        temp_files = [f for f in datasets_dir.glob("*_*.csv") if len(f.stem.split('_')[-1]) == 8]
        for temp_file in temp_files:
            temp_file.unlink()
            print(f"Removed temporary dataset: {temp_file}")
    
    # Clean up logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log*"):
            log_file.unlink()
            print(f"Removed log file: {log_file}")
    
    # Clean up Python cache
    cache_dirs = list(Path(".").rglob("__pycache__"))
    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir)
        print(f"Removed cache directory: {cache_dir}")
    
    print("✅ Cleanup completed")

if __name__ == "__main__":
    cleanup_temporary_files()
```

### 15.2 Update .gitignore
```bash
# Add to .gitignore
logs/
*.log
temp_datasets/
__pycache__/
.pytest_cache/
.coverage
htmlcov/
.env.local
.env.test
celerybeat-schedule
```

### 15.3 Create final validation script
Create `backend/scripts/validate_day2.py`:

```python
#!/usr/bin/env python3
"""
Validation script for Day 2 completion
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def validate_day2_completion():
    """Validate that all Day 2 features are working"""
    print("🔍 Validating Flowza Day 2 Implementation...")
    
    tests = []
    
    # Test 1: API Health Check
    try:
        response = requests.get(f"{BASE_URL}/health")
        tests.append(("API Health Check", response.status_code == 200))
    except:
        tests.append(("API Health Check", False))
    
    # Test 2: Available Nodes
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/available")
        data = response.json()
        has_nodes = len(data.get("preprocessing_nodes", [])) > 0
        tests.append(("Available ML Nodes", response.status_code == 200 and has_nodes))
    except:
        tests.append(("Available ML Nodes", False))
    
    # Test 3: Workflow Execution
    try:
        workflow = {
            "name": "Validation Test",
            "nodes": [{
                "id": "test_node",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Test"},
                "parameters": {"file_path": "datasets/test_data.csv"}
            }],
            "connections": []
        }
        
        response = requests.post(f"{BASE_URL}/api/workflows/run", json=workflow)
        workflow_created = response.status_code == 200
        
        if workflow_created:
            workflow_id = response.json().get("workflow_id")
            time.sleep(2)  # Wait for execution
            
            status_response = requests.get(f"{BASE_URL}/api/workflows/{workflow_id}/status")
            status_check = status_response.status_code == 200
            
            tests.append(("Workflow Execution", workflow_created and status_check))
        else:
            tests.append(("Workflow Execution", False))
    except:
        tests.append(("Workflow Execution", False))
    
    # Test 4: Pipeline Execution  
    try:
        pipeline = {
            "name": "Pipeline Test",
            "nodes": [
                {
                    "id": "load",
                    "type": "csv_loader",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Load"},
                    "parameters": {"file_path": "datasets/test_data.csv"}
                },
                {
                    "id": "clean",
                    "type": "drop_nulls",
                    "position": {"x": 300, "y": 100},
                    "data": {"label": "Clean"},
                    "parameters": {"how": "any"}
                }
            ],
            "connections": [{"source": "load", "target": "clean"}]
        }
        
        response = requests.post(f"{BASE_URL}/api/workflows/run", json=pipeline)
        tests.append(("Pipeline Execution", response.status_code == 200))
    except:
        tests.append(("Pipeline Execution", False))
    
    # Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Day 2 implementation is complete and working!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = validate_day2_completion()
    sys.exit(0 if success else 1)
```

## Step 16: Make Scripts Executable and Final Testing

### 16.1 Make all scripts executable
```bash
chmod +x backend/scripts/*.py
chmod +x backend/scripts/*.sh
```

### 16.2 Run final validation
```bash
# Ensure all services are running
docker-compose up -d
cd backend && python -m uvicorn app.main:app --reload &
cd backend && celery -A app.services.task_executor.celery worker --loglevel=info &

# Wait for services to start
sleep 10

# Create test data if it doesn't exist
if [ ! -f "backend/datasets/test_data.csv" ]; then
  cat > backend/datasets/test_data.csv << 'EOF'
name,age,salary,department,target
John,25,50000,Engineering,1
Jane,,60000,Marketing,1
Bob,30,,Engineering,0
Alice,28,55000,,1
Charlie,35,70000,Engineering,1
Diana,29,58000,Marketing,0
Eve,,62000,Engineering,1
Frank,32,48000,Marketing,0
Grace,26,52000,Engineering,1
Henry,31,,Marketing,0
EOF
fi

# Run validation
python backend/scripts/validate_day2.py
```

## Summary

At the end of Day 2, you should have:

### ✅ Completed Features
- **Workflow Execution Service**: Chain multiple nodes together
- **Enhanced ML Nodes**: CSV loader, drop nulls, preprocessing, train/test split
- **Database Persistence**: Store workflows, tasks, and datasets
- **Error Handling**: Comprehensive error management and recovery
- **Progress Tracking**: Real-time workflow progress monitoring
- **API Endpoints**: Complete REST API for workflow management
- **Data Flow**: Proper data passing between nodes
- **Async Execution**: Celery-based background processing

### 🛠️ Architecture Components
- **WorkflowService**: Core workflow execution logic
- **DatasetService**: Dataset persistence and management  
- **Task Executor**: Individual node execution
- **Error Handler**: Comprehensive error management
- **Database Models**: Enhanced with progress tracking
- **REST API**: Complete workflow management endpoints

### 📊 Available Nodes
- **csv_loader**: Load CSV files
- **drop_nulls**: Remove null values
- **preprocess**: Comprehensive data cleaning
- **train_test_split**: Split datasets for ML training

### 🔧 Development Tools
- **Test Scripts**: Automated testing and validation
- **Monitoring Tools**: Real-time workflow monitoring
- **Cleanup Utilities**: Environment maintenance
- **Documentation**: API and usage documentation

## Next Steps (Day 3-4)

Tomorrow you'll extend the system with:
- More ML nodes (normalize, encode, feature selection)
- Model training nodes (logistic regression, XGBoost)
- Model evaluation and metrics
- Advanced data visualizations
- Model persistence and versioning
- Performance optimizations for larger datasets