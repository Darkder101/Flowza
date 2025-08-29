from typing import Any, Dict
import pandas as pd  # noqa : 
from sklearn.model_selection import train_test_split
from .base import MLNode


class TrainTestSplit(MLNode):
    """Split dataset into training and testing sets"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting train/test split...")

            self.validate_inputs(["dataset"])
            df = self.inputs["dataset"].copy()

            test_size = self.parameters.get("test_size", 0.2)
            random_state = self.parameters.get("random_state", 42)
            target_column = self.parameters.get("target_column", None)
            stratify = self.parameters.get("stratify", False)

            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            stratify_data = None
            if stratify and target_column:
                stratify_data = df[target_column]
                min_class_count = stratify_data.value_counts().min()
                min_test_samples = int(len(df) * test_size)
                if min_class_count < 2 or min_class_count < min_test_samples:
                    self.log_info("Cannot stratify: insufficient samples. Using random split.")
                    stratify_data = None

            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_data,
            )

            self.outputs["train_dataset"] = train_df
            self.outputs["test_dataset"] = test_df

            metadata = {
                "total_samples": len(df),
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "train_ratio": len(train_df) / len(df),
                "test_ratio": len(test_df) / len(df),
                "stratified": stratify_data is not None,
                "target_column": target_column
            }

            if target_column:
                metadata["class_distribution"] = {
                    "overall": df[target_column].value_counts().to_dict(),
                    "train": train_df[target_column].value_counts().to_dict(),
                    "test": test_df[target_column].value_counts().to_dict(),
                }

            self.log_info(f"Split completed: {len(train_df)} train, {len(test_df)} test samples")
            return self.make_response("success", f"Dataset split into {len(train_df)} train and {len(test_df)} test samples", metadata=metadata)

        except Exception as e:
            self.log_error(f"Train/test split failed: {str(e)}")
            return self.make_response("error", str(e))
