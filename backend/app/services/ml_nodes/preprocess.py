import pandas as pd  # noqa :
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .base import MLNode


class Preprocess(MLNode):
    """Basic data preprocessing node"""

    def execute(self) -> Dict[str, Any]:
        try:
            self.log_info("Starting data preprocessing...")

            # Validate inputs
            self.validate_inputs(['dataset'])
            df = self.inputs['dataset'].copy()

            # Get parameters
            drop_nulls = self.parameters.get('drop_nulls', True)
            normalize = self.parameters.get('normalize', False)
            encode_categorical = self.parameters.get('encode_categorical', True)  # noqa : E501

            original_shape = df.shape
            preprocessing_steps = []

            # Drop null values
            if drop_nulls:
                null_counts_before = df.isnull().sum().sum()
                df = df.dropna()
                null_counts_after = df.isnull().sum().sum()
                preprocessing_steps.append(f"Dropped nulls: {null_counts_before} -> {null_counts_after}")  # noqa : E501

            # Encode categorical variables
            if encode_categorical:
                categorical_columns = df.select_dtypes(include=['object']).columns  # noqa : E501
                encoders = {}

                for col in categorical_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le
                    preprocessing_steps.append(f"Encoded categorical column: {col}")  # noqa : E501

                # Store encoders for later use
                self.outputs['encoders'] = encoders

            # Normalize numerical features
            if normalize:
                numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns  # noqa : E501
                scaler = StandardScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # noqa : E501
                preprocessing_steps.append(f"Normalized {len(numerical_columns)} numerical columns")  # noqa : E501

                # Store scaler for later use
                self.outputs['scaler'] = scaler

            # Store output dataset
            self.outputs['dataset'] = df

            self.log_info(f"Preprocessing completed: {original_shape} -> {df.shape}")  # noqa : E501

            return {
                "status": "success",
                "message": f"Preprocessing completed: {original_shape} -> {df.shape}",  # noqa : E501
                "preprocessing_steps": preprocessing_steps,
                "original_shape": original_shape,
                "final_shape": df.shape,
                "columns": list(df.columns)
            }

        except Exception as e:
            self.log_error(f"Preprocessing failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
