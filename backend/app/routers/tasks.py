from typing import List  # noqa : F401

import pandas as pd  # noqa : F401
from app.database.connection import get_db
from app.models.dataset import Dataset
from app.models.task import Task
from app.services.dataset_service import DatasetService
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

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
                        "description": "Path to the CSV file",
                    },
                    "separator": {
                        "type": "string",
                        "default": ",",
                        "description": "Column separator",
                    },
                    "encoding": {
                        "type": "string",
                        "default": "utf-8",
                        "description": "File encoding",
                    },
                },
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
                        "description": "Columns to check for nulls (null = check all)",  # noqa : E501
                    },
                    "how": {
                        "type": "string",
                        "default": "any",
                        "description": "Remove rows with 'any' or 'all' nulls",
                    },
                },
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
                        "description": "Remove rows with null values",
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": False,
                        "description": "Normalize numerical features",
                    },
                    "encode_categorical": {
                        "type": "boolean",
                        "default": True,
                        "description": "Encode categorical variables",
                    },
                    "columns_to_drop": {
                        "type": "array",
                        "default": [],
                        "description": "Columns to remove from dataset",
                    },
                },
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
                        "description": "Proportion of data for testing",
                    },
                    "random_state": {
                        "type": "integer",
                        "default": 42,
                        "description": "Random seed for reproducibility",
                    },
                    "target_column": {
                        "type": "string",
                        "default": None,
                        "description": "Target column for stratification",
                    },
                    "stratify": {
                        "type": "boolean",
                        "default": False,
                        "description": "Stratify split by target column",
                    },
                },
            },
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
                        "description": "Regularization strength",
                    },
                    "max_iter": {
                        "type": "integer",
                        "default": 1000,
                        "description": "Maximum iterations",
                    },
                    "random_state": {
                        "type": "integer",
                        "default": 42,
                        "description": "Random seed",
                    },
                },
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
                        "default": ["accuracy", "f1_score", "precision", "recall"],  # noqa : E501
                        "description": "Metrics to calculate",
                    }
                },
            }
        ],
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
        "output_data": task.output_data,
    }


@router.get("/{task_id}/dataset")
async def get_task_output_dataset(task_id: int, db: Session = Depends(get_db)):
    """Get dataset output from a task"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.output_data or "dataset" not in task.output_data:
        raise HTTPException(status_code=404, detail="No dataset output found")

    try:
        dataset_service = DatasetService()
        dataset_id = task.output_data["dataset"]

        # Get dataset info
        dataset_info = dataset_service.get_dataset_info(dataset_id)

        # Load and preview dataset
        df = dataset_service.load_dataset(dataset_id)
        preview = {
            "shape": df.shape,
            "columns": list(df.columns),
            "head": df.head(10).to_dict("records"),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }

        return {"dataset_info": dataset_info, "preview": preview}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/")
async def list_datasets(db: Session = Depends(get_db), workflow_id: int = None):  # noqa : E501
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
            "node_id": d.node_id,
        }
        for d in datasets
    ]
