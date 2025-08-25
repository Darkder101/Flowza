from typing import Any, Dict, List, Optional  # noqa : 

import pandas as pd  # noqa  : 

from .base import MLNode


class DropNulls(MLNode):
    """Remove rows with null values from dataset"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting null value removal...")

            # Validate inputs
            self.validate_inputs(["dataset"])
            df = self.inputs["dataset"].copy()

            # Get parameters
            subset = self.parameters.get("subset", None)
            how = self.parameters.get("how", "any")

            original_shape = df.shape

            # Validate parameters
            if how not in ["any", "all"]:
                raise ValueError("Parameter 'how' must be 'any' or 'all'")

            if subset is not None:
                if isinstance(subset, str):
                    subset = [subset]

                # Check if columns exist
                missing_cols = [col for col in subset if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Columns not found in dataset: {missing_cols}")  # noqa : 

            # Count nulls before
            null_count_before = df.isnull().sum().sum()

            # Drop null values
            df_cleaned = df.dropna(subset=subset, how=how)

            # Count nulls after
            null_count_after = df_cleaned.isnull().sum().sum()
            rows_dropped = original_shape[0] - df_cleaned.shape[0]

            # Store output
            self.outputs["dataset"] = df_cleaned

            # Log success
            self.log_info(
                f"Dropped {rows_dropped} rows with null values. "
                f"Shape: {original_shape} -> {df_cleaned.shape}"
            )

            return {
                "status": "success",
                "message": f"Removed {rows_dropped} rows with null values",
                "original_shape": original_shape,
                "final_shape": df_cleaned.shape,
                "rows_dropped": rows_dropped,
                "nulls_before": int(null_count_before),
                "nulls_after": int(null_count_after),
                "columns_checked": subset if subset else "all",
            }

        except Exception as e:
            self.log_error(f"Failed to drop nulls: {str(e)}")
            return {"status": "error", "message": str(e)}
