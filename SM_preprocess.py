import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class SimpleTargetEncoder:
    """Simple Target Encoder with smoothing to prevent overfitting"""
    
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.category_means = {}
        
    def fit(self, X, y):
        """Fit the encoder on training data"""
        self.global_mean = y.mean()
        self.category_means = {}
        
        # Calculate smoothed means for each category
        for category in X.unique():
            category_mask = X == category
            category_target = y[category_mask]
            n_samples = len(category_target)
            category_mean = category_target.mean()
            
            # Apply smoothing: (category_mean * n + global_mean * smoothing) / (n + smoothing)
            smoothed_mean = (category_mean * n_samples + self.global_mean * self.smoothing) / (n_samples + self.smoothing)
            self.category_means[category] = smoothed_mean
            
        return self
        
    def transform(self, X):
        """Transform new data using fitted encoder"""
        # Map categories to their encoded values, use global mean for unseen categories
        return X.map(self.category_means).fillna(self.global_mean)
        
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# To do:
# Add encoding for categorical variables - DONE
# split data by time, not random sampling - TODO

def preprocess_data(input_path, output_path, test_size=0.2, val_size=0.2, random_state=42):
    # Preprocess data and split into train/validation/test sets

    print(f"Loading data from {input_path}")

    # Read the raw data
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Basic data info
    print("Target distribution:")
    print(df['Renewed'].value_counts())

    # Identify categorical and numerical columns
    # Exclude the target column 'Renewed' from features
    feature_columns = [col for col in df.columns if col != 'Renewed']

    # Automatically identify categorical columns
    categorical_columns = []
    numerical_columns = []

    for col in feature_columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_columns.append(col)
        elif df[col].nunique() < 10 and df[col].dtype in ['int64', 'float64']:
            # Treat low-cardinality numeric columns as categorical
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)

    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")

    # Handle missing values
    print("Missing values per column:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])

    # Fill missing values
    df_processed = df.copy()

    # Fill categorical missing values with 'Unknown'
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna('Unknown')
            print(f"Filled {col} missing values with 'Unknown'")

    # Fill numerical missing values with median
    for col in numerical_columns:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            print(f"Filled {col} missing values with median: {median_value}")

    # Convert categorical columns to string type for consistency
    for col in categorical_columns:
        df_processed[col] = df_processed[col].astype(str)

    # Split data before encoding to prevent data leakage
    X = df_processed[feature_columns].copy()
    y = df_processed['Renewed'].copy()

    # First split: Training and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    # Second split: validation vs test
    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split:")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Encode categorical columns after splitting to prevent data leakage
    print("Encoding categorical variables...")
    
    # Store encoders for potential future use
    encoders = {}
    
    # Process each categorical column
    for col in categorical_columns:
        n_unique = X_train[col].nunique()
        print(f"Processing {col}: {n_unique} unique values")
        
        if n_unique > 20:  # Use Target Encoding for high cardinality
            print(f"Using Target Encoding for {col}")
            
            # Initialize target encoder with smoothing
            te = SimpleTargetEncoder(smoothing=10.0)
            
            # Fit on training data only and transform
            X_train[col] = te.fit_transform(X_train[col], y_train)
            
            # Transform validation and test sets
            X_val[col] = te.transform(X_val[col])
            X_test[col] = te.transform(X_test[col])
            
            encoders[col] = {'type': 'target', 'encoder_type': 'SimpleTargetEncoder'}
            
        else:  # Use One-Hot Encoding for low cardinality
            print(f"Using One-Hot Encoding for {col}")
            
            # Get dummy variables for training set
            train_dummies = pd.get_dummies(X_train[col], prefix=col, dummy_na=False)
            
            # Store column names for consistency
            dummy_cols = train_dummies.columns.tolist()
            encoders[col] = {'type': 'onehot', 'columns': dummy_cols}
            
            # Replace original column with dummy variables in training set
            X_train = X_train.drop(columns=[col])
            X_train = pd.concat([X_train, train_dummies], axis=1)
            
            # Process validation set
            val_dummies = pd.get_dummies(X_val[col], prefix=col, dummy_na=False)
            X_val = X_val.drop(columns=[col])
            
            # Ensure validation set has same dummy columns as training
            for dummy_col in dummy_cols:
                if dummy_col not in val_dummies.columns:
                    val_dummies[dummy_col] = 0
            val_dummies = val_dummies[dummy_cols]  # Reorder columns
            X_val = pd.concat([X_val, val_dummies], axis=1)
            
            # Process test set
            test_dummies = pd.get_dummies(X_test[col], prefix=col, dummy_na=False)
            X_test = X_test.drop(columns=[col])
            
            # Ensure test set has same dummy columns as training
            for dummy_col in dummy_cols:
                if dummy_col not in test_dummies.columns:
                    test_dummies[dummy_col] = 0
            test_dummies = test_dummies[dummy_cols]  # Reorder columns
            X_test = pd.concat([X_test, test_dummies], axis=1)

    # Reset indices to avoid issues when concatenating
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    print(f"Final feature shapes after encoding:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # Create output directories
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'validation')
    test_dir = os.path.join(output_path, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Combine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save datasets
    train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(val_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

    print(f"Saved datasets to {output_path}")

    # Create metadata for the pipeline
    metadata = {
        "feature_names": X_train.columns.tolist(),  # Updated feature names after encoding
        "original_categorical_columns": categorical_columns,
        "original_numerical_columns": numerical_columns,
        "target_column": "Renewed",
        "original_shape": df.shape,
        "train_shape": train_df.shape,
        "validation_shape": val_df.shape,
        "test_shape": test_df.shape,
        "encoding_info": {
            "target_encoded_columns": [k for k, v in encoders.items() if v['type'] == 'target'],
            "onehot_columns": {k: v['columns'] for k, v in encoders.items() if v['type'] == 'onehot'},
            "encoding_threshold": 20  # Columns with >10 unique values get target encoded
        },
        "preprocessing_info": {
            "missing_value_strategy": {
                "categorical": "Unknown",
                "numerical": "median"
            },
            "train_test_split": {
                "test_size": test_size,
                "validation_size": val_size,
                "random_state": random_state
            }
        }
    }

    # Save metadata to all directories (required by your training script)
    for directory in [train_dir, val_dir, test_dir]:
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print("Metadata saved to all directories")

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for XGBoost training')

    # SageMaker processing job arguments
    parser.add_argument('--input-data', 
                        type=str, 
                        default='/opt/ml/processing/input/Renewal_Forecast_individual_modeling_dataset_FINAL_250715.csv',
                        help='Path to input data file')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output',
                        help='Path to output directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Proportion of data to use for validation set')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    args = parser.parse_args()
    
    print("Starting data preprocessing...")
    print(f"Input data: {args.input_data}")
    print(f"Output directory: {args.output_data}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print(f"Random state: {args.random_state}")
    
    # Run preprocessing
    try:
        metadata = preprocess_data(
            input_path=args.input_data,
            output_path=args.output_data,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state
        )
        
        print("Preprocessing completed successfully!")
        print(f"Features: {len(metadata['feature_names'])}")
        print(f"Original categorical features: {len(metadata['original_categorical_columns'])}")
        print(f"Original numerical features: {len(metadata['original_numerical_columns'])}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise e