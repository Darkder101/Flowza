from typing import Any, Dict

import pandas as pd  # noqa : 
from sklearn.model_selection import train_test_split

from .base import MLNode


class TrainTestSplit(MLNode):
    """Split dataset into training and testing sets"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting train/test split...")

            # Validate inputs
            self.validate_inputs(["dataset"])
            df = self.inputs["dataset"].copy()

            # Get parameters
            test_size = self.parameters.get("test_size", 0.2)
            random_state = self.parameters.get("random_state", 42)
            target_column = self.parameters.get("target_column", None)
            stratify = self.parameters.get("stratify", False)

            # Validate parameters
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")

            if target_column and target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found in dataset"
                )

            # Prepare stratification
            stratify_data = None
            if stratify and target_column:
                stratify_data = df[target_column]

                # Check if stratification is possible
                value_counts = stratify_data.value_counts()
                min_class_count = value_counts.min()
                min_test_samples = int(len(df) * test_size)

                if min_class_count < 2:
                    self.log_info(
                        "Cannot stratify: some classes have only 1 sample. Using random split."  # noqa : E501
                    )
                    stratify_data = None
                elif min_class_count < min_test_samples:
                    self.log_info(
                        "Cannot stratify: insufficient samples per class. Using random split."  # noqa : E501
                    )
                    stratify_data = None

            # Perform train/test split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_data,
            )

            # Store outputs
            self.outputs["train_dataset"] = train_df
            self.outputs["test_dataset"] = test_df

            # Calculate split statistics
            train_size = len(train_df)
            test_size = len(test_df)
            total_size = len(df)

            split_info = {
                "total_samples": total_size,
                "train_samples": train_size,
                "test_samples": test_size,
                "train_ratio": train_size / total_size,
                "test_ratio": test_size / total_size,
                "stratified": stratify_data is not None,
                "target_column": target_column,
            }

            # Add class distribution if target column exists
            if target_column and target_column in df.columns:
                split_info["class_distribution"] = {
                    "overall": df[target_column].value_counts().to_dict(),
                    "train": train_df[target_column].value_counts().to_dict(),
                    "test": test_df[target_column].value_counts().to_dict(),
                }

            self.log_info(
                f"Split completed: {train_size} train, {test_size} test samples"  # noqa : E501
            )

            return {
                "status": "success",
                "message": f"Dataset split into {train_size} training and {test_size} test samples",  # noqa : E501
                "split_info": split_info,
            }

        except Exception as e:
            self.log_error(f"Train/test split failed: {str(e)}")
            return {"status": "error", "message": str(e)}
