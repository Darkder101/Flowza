# Flowza Day 1: Project Setup & Infrastructure

## Prerequisites Check
Before starting, ensure you have:
- [ ] VS Code installed
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed (for future frontend)
- [ ] Docker Desktop installed and running
- [ ] Git connected to your Flowza repository

## Step 1: Project Structure Setup

### 1.1 Create the base project structure
```bash
# Navigate to your Flowza repository
cd Flowza

# Create the main project structure
mkdir backend frontend docs docker

# Backend structure
cd backend
mkdir app app/routers app/services app/services/ml_nodes app/models app/schemas app/ai app/database
touch app/__init__.py app/main.py app/database.py

# Create subdirectories in app
cd app
touch __init__.py
cd routers
touch __init__.py workflows.py tasks.py
cd ../services
touch __init__.py workflow_service.py task_executor.py
cd ml_nodes
touch __init__.py base.py csv_loader.py preprocess.py train_logreg.py train_xgboost.py evaluate.py
cd ../../models
touch __init__.py workflow.py task.py dataset.py
cd ../schemas
touch __init__.py workflow_schemas.py task_schemas.py
cd ../ai
touch __init__.py workflow_generator.py
cd ../database
touch __init__.py connection.py

# Go back to root
cd ../../..
```

### 1.2 Create configuration files
```bash
# In Flowza root directory
touch .env .gitignore docker-compose.yml README.md requirements.txt

# Backend specific files
cd backend
touch requirements.txt .env
cd ..

# Docker files
cd docker
touch ml-base.dockerfile
cd ..
```

## Step 2: VS Code Workspace Configuration

### 2.1 Create VS Code workspace file
Create `Flowza.code-workspace` in the root directory:

```json
{
    "folders": [
        {
            "path": "."
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "${workspaceFolder}/backend/venv/Scripts/python.exe",
        "python.terminal.activateEnvironment": true,
        "python.formatting.provider": "none",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": false,
        "python.linting.flake8Enabled": true,
        "python.analysis.autoImportCompletions": true,
        "python.analysis.extraPaths": ["./backend"],
        
        // Terminal defaults
        "terminal.integrated.defaultProfile.windows": "Git Bash",
        "terminal.integrated.cwd": "./backend",
        "terminal.integrated.profiles": {
            "Git Bash": {
                "path": "C:\\Program Files\\Git\\bin\\bash.exe"
            }
        },
        "terminal.integrated.env.windows": {
            "VIRTUAL_ENV": "${workspaceFolder}/backend/venv",
            "PATH": "${workspaceFolder}/backend/venv/Scripts;${env:PATH}"
        },

        // Python formatting and linting
        "[python]": {
            "editor.defaultFormatter": "ms-python.black-formatter",
            "editor.formatOnSave": true,
            "editor.codeActionsOnSave": {
                "source.organizeImports": "explicit"
            }
        },

        // File excludes
        "files.exclude": {
            "**/__pycache__": true,
            "**/*.pyc": true,
            "**/node_modules": true,
            "**/venv": true,
            "**/.env": false,
            "**/.pytest_cache": true,
            "**/datasets": false,
            "**/models": false
        },

        // File associations
        "files.associations": {
            "*.env": "dotenv"
        }
    },
    "extensions": {
        "recommendations": [
            "ms-python.python",
            "ms-python.flake8",
            "ms-python.black-formatter",
            "bradlc.vscode-tailwindcss",
            "esbenp.prettier-vscode",
            "ms-vscode.vscode-json",
            "redhat.vscode-yaml",
            "ms-azuretools.vscode-docker",
            "ms-python.isort"
        ]
    }
}
```

### 2.2 Open workspace in VS Code
```bash
# From Flowza root directory
code Flowza.code-workspace
```

## Step 3: Backend Environment Setup

### 3.1 Create Python virtual environment
```bash
# In VS Code terminal, navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation (should show (venv) prefix in terminal)
which python
```

### 3.2 Install core dependencies
Create `backend/requirements.txt`:
```txt
# FastAPI Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-dotenv==1.0.0

# Database & ORM
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1

# Task Queue
celery==5.3.4
redis==5.0.1

# ML Libraries
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.4
xgboost==2.0.2
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2

# Data Validation & Serialization
pydantic==2.5.0
pydantic-settings==2.1.0

# HTTP Client
httpx==0.25.2
requests==2.31.0

# Utilities
python-dateutil==2.8.2
openpyxl==3.1.2

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
black==23.11.0
flake8==6.1.0

# AI Integration (Optional for Week 7-8)
openai==1.3.7

# Docker
docker==6.1.3
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3.3 Create environment configuration
Create `backend/.env`:
```env
# Database Configuration
DATABASE_URL=postgresql://flowza:password123@localhost:5433/flowza_db
REDIS_URL=redis://localhost:6380/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-super-secret-ml-key-change-in-production
ALGORITHM=HS256

# ML Configuration
MAX_DATASET_SIZE_MB=100
MODEL_STORAGE_PATH=/app/models
DATASET_STORAGE_PATH=/app/datasets

# Docker Configuration
DOCKER_SOCKET=/var/run/docker.sock
ML_CONTAINER_PREFIX=flowza_node

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6380/0
CELERY_RESULT_BACKEND=redis://localhost:6380/0
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=["json"]

# AI Integration (for later phases)
OPENAI_API_KEY=your-openai-api-key-here
AI_WORKFLOW_GENERATION=False
```

## Step 4: Database & Docker Setup

### 4.1 Create docker-compose.yml
In the root directory, create `docker-compose.yml`:
```yaml


services:
  postgres:
    image: postgres:15
    container_name: flowza_postgres
    environment:
      POSTGRES_DB: flowza_db
      POSTGRES_USER: flowza
      POSTGRES_PASSWORD: password123
    ports:
      - "5433:5432"
    volumes:
      - flowza_postgres_data:/var/lib/postgresql/data
      - ./backend/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - flowza_network

  redis:
    image: redis:7-alpine
    container_name: mlflow_redis
    ports:
      - "6380:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - flowza_network

  # ML Base Container for running nodes
  ml-runner:
    build:
      context: .
      dockerfile: docker/ml-base.dockerfile
    container_name: flowza_ml_runner
    volumes:
      - ./backend/app/services/ml_nodes:/app/ml_nodes
      - ml_datasets:/app/datasets
      - ml_models:/app/models
    networks:
      - flowza_network
    command: tail -f /dev/null  # Keep container running for exec commands

  # Celery Worker
  celery-worker:
    build:
      context: .
      dockerfile: docker/ml-base.dockerfile
    container_name: flowza_celery_worker
    command: celery -A app.main.celery worker --loglevel=info
    volumes:
      - ./backend:/app
      - ml_datasets:/app/datasets
      - ml_models:/app/models
    environment:
      - DATABASE_URL=postgresql://flowza:password123@postgres:5432/flowza_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - flowza_network


volumes:
  flowza_postgres_data:
  redis_data:
  ml_datasets:
  ml_models:

networks:
  flowza_network:
    driver: bridge
```

### 4.2 Create ML base Docker image
Create `docker/ml-base.dockerfile`:
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and models
RUN mkdir -p /app/datasets /app/models /app/ml_nodes

# Copy ML nodes
COPY backend/app/services/ml_nodes /app/ml_nodes/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "print('ML Base Container Ready')"]
```

### 4.3 Start the services
```bash
# From Flowza root directory
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs if needed
docker-compose logs postgres
docker-compose logs redis
```

## Step 5: Database Models & Connection

### 5.1 Create database connection
Create `backend/app/database/connection.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True if os.getenv("DEBUG") == "True" else False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 5.2 Create core models
Create `backend/app/models/workflow.py`:
```python
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.connection import Base


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Workflow definition (nodes and connections)
    nodes = Column(JSON)  # List of workflow nodes
    connections = Column(JSON)  # Node connections/edges

    # Execution info
    status = Column(String(50), default="draft")  # draft, running, completed, failed  # noqa : E501
    results = Column(JSON)  # Final results
    execution_time = Column(Integer)  # seconds

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_template = Column(Boolean, default=False)

    # Relationships
    tasks = relationship("Task", back_populates="workflow", cascade="all, delete-orphan")  # noqa : E501

```

Create `backend/app/models/task.py`:
```python
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, ForeignKey, Float  # noqa : E501
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.connection import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)

    # Task definition
    node_id = Column(String(100), nullable=False)  # Unique within workflow
    task_type = Column(String(100), nullable=False)  # csv_loader, preprocess, train_logreg, etc.  # noqa : E501
    task_name = Column(String(255), nullable=False)

    # Task configuration
    parameters = Column(JSON)  # Input parameters for the task
    input_data = Column(JSON)  # References to input datasets/models
    output_data = Column(JSON)  # References to output datasets/models

    # Execution info
    status = Column(String(50), default="pending")  # pending, running, completed, failed  # noqa : E501
    result = Column(JSON)  # Task execution results
    error_message = Column(Text)
    execution_time = Column(Float)  # seconds

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    workflow = relationship("Workflow", back_populates="tasks")

```

Create `backend/app/models/dataset.py`:
```python
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Float, Boolean  # noqa : E501
from sqlalchemy.sql import func
from app.database.connection import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)

    # Dataset info
    format = Column(String(50))  # csv, json, parquet, etc.
    size_mb = Column(Float)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    columns_info = Column(JSON)  # Column names, types, stats

    # Metadata
    description = Column(Text)
    source = Column(String(255))  # uploaded, generated, etc.
    is_temporary = Column(Boolean, default=False)  # For intermediate datasets

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

```

## Step 6: Basic FastAPI Application

### 6.1 Create main FastAPI application
Create `backend/app/main.py`:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database.connection import engine, Base
from app.routers import workflows, tasks
import os
from dotenv import load_dotenv
from celery import Celery
import traceback
from contextlib import asynccontextmanager

load_dotenv()

# Initialize Celery
celery = Celery(
    "mlflow",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["app.services.task_executor"]
)

# Lifespan event handler (replaces on_event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    Base.metadata.create_all(bind=engine)
    print("📦 Database tables checked/created")

    yield  # app runs here

    # --- Shutdown ---
    print("🛑 Shutting down Flowza API")


app = FastAPI(
    title="Flowza - Visual ML Workflow Platform",
    description="Drag-and-drop interface for building and executing ML pipelines",  # noqa : E501
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files and models
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
app.mount("/models", StaticFiles(directory="models"), name="models")

# Include routers
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])  # noqa : E501
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to Flowza API",
        "version": "0.1.0",
        "status": "running",
        "available_endpoints": {
            "workflows": "/api/workflows",
            "tasks": "/api/tasks",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "redis": "connected",
        "ml_nodes": "ready"
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset for use in workflows"""
    import pandas as pd
    from app.models.dataset import Dataset
    from app.database.connection import SessionLocal

    try:
        # Save uploaded file
        file_path = f"datasets/{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Analyze dataset
        df = pd.read_csv(file_path)

        # Store dataset metadata in database
        db = SessionLocal()
        dataset = Dataset(
            name=file.filename,
            file_path=file_path,
            format="csv",
            size_mb=os.path.getsize(file_path) / 1024 / 1024,
            num_rows=len(df),
            num_columns=len(df.columns),
            columns_info={
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict()
            },
            source="uploaded"
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.close()

        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset.id,
            "name": dataset.name,
            "rows": dataset.num_rows,
            "columns": dataset.num_columns
        }

    except Exception as e:
        # Log the error to uvicorn console
        print("❌ Upload failed:", str(e))
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )

```

### 6.2 Create basic routers
Create `backend/app/routers/workflows.py`:
```python
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
        nodes=[node.model_dump() for node in workflow.nodes],
        connections=[conn.model_dump() for conn in workflow.connections]
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

```

Create `backend/app/routers/tasks.py`:
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.task import Task
from typing import List  # noqa : 

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
                    "file_path": {"type": "string", "required": True},
                    "separator": {"type": "string", "default": ","},
                    "encoding": {"type": "string", "default": "utf-8"}
                }
            }
        ],
        "preprocessing_nodes": [
            {
                "type": "preprocess",
                "name": "Data Preprocessing",
                "category": "preprocessing",
                "description": "Clean and preprocess dataset",
                "inputs": ["dataset"],
                "outputs": ["dataset"],
                "parameters": {
                    "drop_nulls": {"type": "boolean", "default": True},
                    "normalize": {"type": "boolean", "default": False},
                    "encode_categorical": {"type": "boolean", "default": True}
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
                    "test_size": {"type": "float", "default": 0.2},
                    "random_state": {"type": "integer", "default": 42},
                    "stratify": {"type": "boolean", "default": True}
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
                    "C": {"type": "float", "default": 1.0},
                    "max_iter": {"type": "integer", "default": 1000},
                    "random_state": {"type": "integer", "default": 42}
                }
            },
            {
                "type": "train_xgboost",
                "name": "XGBoost Classifier",
                "category": "training",
                "description": "Train an XGBoost classification model",
                "inputs": ["train_dataset"],
                "outputs": ["model"],
                "parameters": {
                    "n_estimators": {"type": "integer", "default": 100},
                    "max_depth": {"type": "integer", "default": 3},
                    "learning_rate": {"type": "float", "default": 0.1}
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
                    "metrics": {"type": "array", "default": ["accuracy", "f1_score", "precision", "recall"]}  # noqa : 
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
        "status": task.status,
        "result": task.result,
        "error_message": task.error_message,
        "execution_time": task.execution_time
    }

```

### 6.3 Create Pydantic schemas
Create `backend/app/schemas/workflow_schemas.py`:
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]

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

class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    status: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True
```

Create `backend/app/schemas/task_schemas.py`:
```python
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
```

## Step 7: Create First ML Node

### 7.1 Create base ML node class
Create `backend/app/services/ml_nodes/base.py`:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLNode(ABC):
    """Base class for all ML workflow nodes"""
    
    def __init__(self, node_id: str, parameters: Dict[str, Any] = None):
        self.node_id = node_id
        self.parameters = parameters or {}
        self.inputs = {}
        self.outputs = {}
        
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the ML node logic"""
        pass
        
    def set_input(self, input_name: str, input_data: Any):
        """Set input data for the node"""
        self.inputs[input_name] = input_data
        
    def get_output(self, output_name: str) -> Any:
        """Get output data from the node"""
        return self.outputs.get(output_name)
        
    def validate_inputs(self, required_inputs: List[str]) -> bool:
        """Validate that all required inputs are present"""
        for input_name in required_inputs:
            if input_name not in self.inputs:
                raise ValueError(f"Required input '{input_name}' is missing")
        return True
        
    def log_info(self, message: str):
        """Log information about node execution"""
        logger.info(f"[{self.node_id}] {message}")
        
    def log_error(self, message: str):
        """Log error about node execution"""
        logger.error(f"[{self.node_id}] {message}")
```

### 7.2 Create CSV Loader node
Create `backend/app/services/ml_nodes/csv_loader.py`:
```python
import pandas as pd
import os
from typing import Dict, Any
from .base import MLNode

class CSVLoader(MLNode):
    """Load data from CSV file"""
    
    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting CSV loading...")
            
            # Get parameters
            file_path = self.parameters.get('file_path')
            separator = self.parameters.get('separator', ',')
            encoding = self.parameters.get('encoding', 'utf-8')
            
            if not file_path:
                raise ValueError("file_path parameter is required")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Load CSV
            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            # Store output
            self.outputs['dataset'] = df
            
            # Log success
            self.log_info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return {
                "status": "success",
                "message": f"Loaded {df.shape[0]} rows and {df.shape[1]} columns",
                "output_shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
        except Exception as e:
            self.log_error(f"Failed to load CSV: {str(e)}")
### 7.3 Create basic preprocessing node
Create `backend/app/services/ml_nodes/preprocess.py`:
```python
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .base import MLNode

class Preprocess(MLNode):
    """Basic data preprocessing node"""
    
    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting data preprocessing...")
            
            # Validate inputs
            self.validate_inputs(['dataset'])
            df = self.inputs['dataset'].copy()
            
            # Get parameters
            drop_nulls = self.parameters.get('drop_nulls', True)
            normalize = self.parameters.get('normalize', False)
            encode_categorical = self.parameters.get('encode_categorical', True)
            
            original_shape = df.shape
            preprocessing_steps = []
            
            # Drop null values
            if drop_nulls:
                null_counts_before = df.isnull().sum().sum()
                df = df.dropna()
                null_counts_after = df.isnull().sum().sum()
                preprocessing_steps.append(f"Dropped nulls: {null_counts_before} -> {null_counts_after}")
            
            # Encode categorical variables
            if encode_categorical:
                categorical_columns = df.select_dtypes(include=['object']).columns
                encoders = {}
                
                for col in categorical_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
                    preprocessing_steps.append(f"Encoded categorical column: {col}")
                
                # Store encoders for later use
                self.outputs['encoders'] = encoders
            
            # Normalize numerical features
            if normalize:
                numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
                scaler = StandardScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
                preprocessing_steps.append(f"Normalized {len(numerical_columns)} numerical columns")
                
                # Store scaler for later use
                self.outputs['scaler'] = scaler
            
            # Store output dataset
            self.outputs['dataset'] = df
            
            self.log_info(f"Preprocessing completed: {original_shape} -> {df.shape}")
            
            return {
                "status": "success",
                "message": f"Preprocessing completed: {original_shape} -> {df.shape}",
                "preprocessing_steps": preprocessing_steps,
                "original_shape": original_shape,
                "final_shape": df.shape,
                "columns": list(df.columns)
            }
            
        except Exception as e:
            self.log_error(f"Preprocessing failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
```

### 7.4 Create task executor service
Create `backend/app/services/task_executor.py`:
```python
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
        task.status = "completed" if result.get("status") == "success" else "failed"
        task.error_message = result.get("message") if result.get("status") == "error" else None
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
    return {"status": "success", "message": f"Workflow {workflow_id} execution started"}
```

## Step 8: Test the Setup

### 8.1 Run the FastAPI server
```bash
# Make sure you're in the backend directory with venv activated
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 8.2 Test the API endpoints
Open your browser and visit:
- http://localhost:8000 (should show welcome message)
- http://localhost:8000/docs (FastAPI automatic documentation)
- http://localhost:8000/api/workflows (should return empty list)
- http://localhost:8000/api/tasks/available (should show available ML nodes)

### 8.3 Test with curl commands
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test available ML nodes
curl http://localhost:8000/api/tasks/available

# Test workflow templates
curl http://localhost:8000/api/workflows/templates/

# Create a test workflow
curl -X POST http://localhost:8000/api/workflows/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Classification Pipeline",
    "description": "A simple test workflow",
    "nodes": [
      {
        "id": "load_csv",
        "type": "csv_loader",
        "position": {"x": 100, "y": 100},
        "data": {"label": "Load CSV"}
      }
    ],
    "connections": []
  }'
```

### 8.4 Test file upload
Create a sample CSV file for testing:
```bash
# Create a sample dataset
echo "age,income,education,target
25,50000,bachelor,1
30,60000,master,1
22,35000,high_school,0
45,80000,phd,1
28,55000,bachelor,1" > sample_data.csv

# Upload the file
curl -X POST http://localhost:8000/api/upload \
  -F "file=@sample_data.csv"
```

### 8.5 Start Celery worker (in another terminal)
```bash
# In a new terminal, navigate to backend and activate venv
cd backend
source venv/bin/activate

# Start Celery worker
celery -A app.services.task_executor.celery worker --loglevel=info
```

## Step 9: Create Sample Datasets

### 9.1 Create sample datasets directory
```bash
# In the backend directory
mkdir -p sample_datasets

# Create sample classification dataset
cat > sample_datasets/iris.csv << EOF
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
EOF

# Create sample regression dataset
cat > sample_datasets/housing.csv << EOF
size,bedrooms,age,price
2000,3,10,250000
1800,3,15,220000
2200,4,8,280000
1600,2,20,200000
2400,4,5,320000
1900,3,12,240000
EOF
```

### 9.2 Test CSV loader with sample data
```bash
# Test loading the iris dataset
curl -X POST http://localhost:8000/api/upload \
  -F "file=@sample_datasets/iris.csv"
```

## Step 10: Update .gitignore

Create `.gitignore` in the root directory:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
*.pyc
.pytest_cache/

# FastAPI & ML
.env
*.db
datasets/
models/
*.pkl
*.joblib

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# ML & Data files
*.csv
*.json
*.parquet
*.h5
*.hdf5

# Node.js (for future frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn/
build/
dist/

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
.directory

# Docker
.docker/
docker-compose.override.yml

# Logs
*.log
logs/
celerybeat-schedule

# Database
*.sqlite3
postgres_data/
redis_data/

# ML Model artifacts
wandb/
mlruns/
.mlflow/

# Temporary files
tmp/
temp/
.tmp/

# Sample datasets (uncomment if you want to include them)
# sample_datasets/

# Environment files
.env.local
.env.development
.env.production
```

## Step 11: Create Development Scripts

### 11.1 Create startup script
Create `scripts/start_dev.sh`:
```bash
#!/bin/bash
# Development startup script

echo "🚀 Starting Flowza Development Environment"

# Start Docker services
echo "📦 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
docker-compose ps

# Activate Python environment and start FastAPI
echo "🐍 Starting FastAPI server..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

echo "✅ Development environment ready!"
echo "📊 API: http://localhost:8000"
echo "📚 Docs: http://localhost:8000/docs"
echo "🗄️ PostgreSQL: localhost:5433"
echo "🔄 Redis: localhost:6380"
```

### 11.2 Create Celery startup script
Create `scripts/start_celery.sh`:
```bash
#!/bin/bash
# Start Celery worker

echo "🔄 Starting Celery worker..."

cd backend
source venv/bin/activate
celery -A app.services.task_executor.celery worker --loglevel=info
```

Make scripts executable:
```bash
chmod +x scripts/start_dev.sh
chmod +x scripts/start_celery.sh
```

## 🏗️ Architecture

- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Task Queue**: Celery + Redis  
- **ML Execution**: Docker containers
- **Frontend**: React + React Flow (coming in Week 5-6)

## 📊 Available ML Nodes

### Data Input
- **CSV Loader**: Load data from CSV files

### Preprocessing  
- **Data Preprocessing**: Clean, normalize, encode categorical variables
- **Train/Test Split**: Split datasets for training and testing

### Training (Coming Week 3-4)
- **Logistic Regression**: Binary/multiclass classification
- **XGBoost**: Gradient boosting classifier

### Evaluation (Coming Week 3-4)
- **Model Evaluation**: Calculate accuracy, F1-score, precision, recall


## 📅 Development Roadmap

- [x] **Week 1**: Project setup, basic infrastructure
- [ ] **Week 2**: Workflow execution engine  
- [ ] **Week 3-4**: Core ML nodes (preprocess, train, evaluate)
- [ ] **Week 5-6**: React frontend with visual workflow editor
- [ ] **Week 7-8**: AI-assisted workflow generation

## 🛠️ Development

### Project Structure
```
MLFlowBuilder/
├── .env
├── .gitignore
├── docker-compose.yml
├── README.md
├── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
|   |   |   └── connection.py 
│   │   ├── models/
|   |   |   ├── dataset.py 
│   │   │   ├── workflow.py
│   │   │   └── task.py
│   │   ├── routers/
│   │   │   ├── workflows.py
│   │   │   └── tasks.py
│   │   ├── services/
|   |   |   ├── task_executor.py 
│   │   │   ├── workflow_service.py
│   │   │   └── ml_nodes/
│   │   │       ├── base.py
│   │   │       ├── csv_loader.py
│   │   │       ├── preprocess.py
│   │   │       ├── train_logreg.py
│   │   │       ├── train_xgboost.py
│   │   │       └── evaluate.py
│   │   ├── schemas/
│   │   │   ├── workflow_schemas.py
│   │   │   └── task_schemas.py
│   │   └── ai/
│   │       └── workflow_generator.py
|   ├── datasets/
|   |   ├── iris.csv
|   |   ├── sample_data.csv
|   |   └── housing.csv
|   ├── models/
|   ├── scripts/
|   |   ├── start_celery.sh
|   |   └── start_dev.sh
│   ├── requirements.txt
│   ├── init.sql
|   └── .env
├── frontend/  
├── docs/
│   ├── Ai_Helper/
|   ├── dev_journal/
|   ├── guide/
└── docker/
    └── ml-base.dockerfile
```

### Database Migrations
```bash
cd backend
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## 🔜 Next Steps (Week 2)

Tomorrow you'll implement:
- [ ] Workflow execution engine that chains multiple nodes
- [ ] Data flow between nodes (output → input passing)
- [ ] Error handling and recovery
- [ ] Progress tracking for long-running workflows
- [ ] Basic train/test split functionality