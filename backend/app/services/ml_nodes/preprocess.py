import pandas as pd  # noqa : 
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .base import MLNode


class Preprocess(MLNode):
    """Basic data preprocessing node"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting data preprocessing...")

            self.validate_inputs(['dataset'])
            df = self.inputs['dataset'].copy()

            drop_nulls = self.parameters.get('drop_nulls', True)
            normalize = self.parameters.get('normalize', False)
            encode_categorical = self.parameters.get('encode_categorical', True)

            original_shape = df.shape
            steps = []

            if drop_nulls:
                null_before = df.isnull().sum().sum()
                df = df.dropna()
                null_after = df.isnull().sum().sum()
                steps.append(f"Dropped nulls: {null_before} -> {null_after}")

            encoders = {}
            if encode_categorical:
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
                    steps.append(f"Encoded categorical column: {col}")
                self.outputs['encoders'] = encoders

            if normalize:
                num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                scaler = StandardScaler()
                df[num_cols] = scaler.fit_transform(df[num_cols])
                steps.append(f"Normalized {len(num_cols)} numerical columns")
                self.outputs['scaler'] = scaler

            self.outputs['dataset'] = df

            metadata = {
                "original_shape": original_shape,
                "final_shape": df.shape,
                "columns": list(df.columns),
                "preprocessing_steps": steps
            }

            self.log_info(f"Preprocessing completed: {original_shape} -> {df.shape}")
            return self.make_response("success", f"Preprocessing completed: {original_shape} -> {df.shape}", metadata=metadata)

        except Exception as e:
            self.log_error(f"Preprocessing failed: {str(e)}")
            return self.make_response("error", str(e))
