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
