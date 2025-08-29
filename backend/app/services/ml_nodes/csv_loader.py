import pandas as pd
import os
from typing import Dict, Any
from .base import MLNode


class CSVLoader(MLNode):
    """Load data from CSV file"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting CSV loading...")

            file_path = self.parameters.get('file_path')
            separator = self.parameters.get('separator', ',')
            encoding = self.parameters.get('encoding', 'utf-8')

            if not file_path:
                raise ValueError("file_path parameter is required")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            self.outputs['dataset'] = df

            metadata = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }

            self.log_info(f"Successfully loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return self.make_response("success", f"Loaded {df.shape[0]} rows and {df.shape[1]} columns", metadata=metadata)

        except Exception as e:
            self.log_error(f"Failed to load CSV: {str(e)}")
            return self.make_response("error", str(e))
