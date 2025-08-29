from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from app.services.dataset_service import DatasetService
from app.services.task_executor import get_task_status as celery_get_task_status

router = APIRouter()
dataset_service = DatasetService()


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
                    "file_path": {"type": "string", "required": True, "description": "Path to the CSV file"},
                    "separator": {"type": "string", "default": ",", "description": "Column separator"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"},
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
                        "description": "Columns to check for nulls (null = check all)",
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
                    "drop_nulls": {"type": "boolean", "default": True, "description": "Remove rows with null values"},
                    "normalize": {"type": "boolean", "default": False, "description": "Normalize numerical features"},
                    "encode_categorical": {"type": "boolean", "default": True, "description": "Encode categorical variables"},
                    "columns_to_drop": {"type": "array", "default": [], "description": "Columns to remove from dataset"},
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
                    "test_size": {"type": "float", "default": 0.2, "description": "Proportion of data for testing"},
                    "random_state": {"type": "integer", "default": 42, "description": "Random seed for reproducibility"},
                    "target_column": {"type": "string", "default": None, "description": "Target column for stratification"},
                    "stratify": {"type": "boolean", "default": False, "description": "Stratify split by target column"},
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
                    "C": {"type": "float", "default": 1.0, "description": "Regularization strength"},
                    "max_iter": {"type": "integer", "default": 1000, "description": "Maximum iterations"},
                    "random_state": {"type": "integer", "default": 42, "description": "Random seed"},
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
                        "default": ["accuracy", "f1_score", "precision", "recall"],
                        "description": "Metrics to calculate",
                    }
                },
            }
        ],
    }


@router.get("/{task_id}")
async def get_task_status(task_id: int):
    """Get task execution status"""
    result = celery_get_task_status(task_id)
    if result.get("status") == "error" and result.get("message") == "Task not found":
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.get("/{task_id}/dataset")
async def get_task_output_dataset(task_id: int):
    """Get dataset output from a task"""
    try:
        # Use DatasetService to fetch dataset info
        dataset_info = dataset_service.get_dataset_info(task_id)

        # Load dataset for preview
        df = dataset_service.load_dataset(task_id)
        preview = {
            "shape": df.shape,
            "columns": list(df.columns),
            "head": df.head(10).to_dict("records"),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }

        return {"dataset_info": dataset_info, "preview": preview}

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/")
async def list_datasets(workflow_id: Optional[int] = Query(None)):
    """List all datasets optionally filtered by workflow"""
    datasets = dataset_service.list_datasets(workflow_id)
    return datasets
