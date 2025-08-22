from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database.connection import engine, Base
from app.routers import workflows, tasks
import os
from dotenv import load_dotenv
from celery import Celery

load_dotenv()

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize Celery
celery = Celery(
    "mlflow",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    include=["app.services.task_executor"]
)

app = FastAPI(
    title="Flowza - Visual ML Workflow Platform",
    description="Drag-and-drop interface for building and executing ML pipelines",  # noqa : E501
    version="0.1.0"
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

    # Save uploaded file
    file_path = f"datasets/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )
