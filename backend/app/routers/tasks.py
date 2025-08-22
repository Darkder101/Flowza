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
