from typing import Any, Dict, List  # noqa : 

import pandas as pd  # noqa : 

from .base import MLNode


class DropNulls(MLNode):
    """Remove rows with null values from dataset"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting null value removal...")

            self.validate_inputs(["dataset"])
            df = self.inputs["dataset"].copy()

            subset = self.parameters.get("subset", None)
            how = self.parameters.get("how", "any")

            if how not in ["any", "all"]:
                raise ValueError("Parameter 'how' must be 'any' or 'all'")

            if subset is not None:
                if isinstance(subset, str):
                    subset = [subset]
                missing_cols = [col for col in subset if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Columns not found in dataset: {missing_cols}")

            original_shape = df.shape
            null_count_before = df.isnull().sum().sum()
            df_cleaned = df.dropna(subset=subset, how=how)
            null_count_after = df_cleaned.isnull().sum().sum()
            rows_dropped = original_shape[0] - df_cleaned.shape[0]
            self.outputs["dataset"] = df_cleaned

            metadata = {
                "original_shape": original_shape,
                "final_shape": df_cleaned.shape,
                "rows_dropped": rows_dropped,
                "nulls_before": int(null_count_before),
                "nulls_after": int(null_count_after),
                "columns_checked": subset if subset else "all",
            }

            self.log_info(
                f"Dropped {rows_dropped} rows with null values. Shape: {original_shape} -> {df_cleaned.shape}"
            )
            return self.make_response(
                "success",
                f"Removed {rows_dropped} rows with null values",
                metadata=metadata,
            )

        except Exception as e:
            self.log_error(f"Failed to drop nulls: {str(e)}")
            return self.make_response("error", str(e))
            self.log_error(f"Failed to drop nulls: {str(e)}")
            return self.make_response("error", str(e))
