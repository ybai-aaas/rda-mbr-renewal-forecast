import os
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, OneHotEncoder

# To do:
# Add encoding for categorical variables
# split data by time, not random sampling

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

    # Split data into train/temp, then temp into validation/test
    X = df_processed[feature_columns]
    y = df_processed['Renewed']

    # Encode categorical columns ###########################################################################
    # NEW 0718: Encode categorical columns
    # if unique elements > 10, use LE: 
    # if unique elements < 10, use OHE: pd.get_dummies()
    # OR: if column is KeyCode__c, use Target Encoding; else use get dummies
    TE = TargetEncoder()
    for col in X:
        if df_processed[col].unique() > 20:
            # Use Target Encoding
            df_processed[col] = TE(smooth='auto').fit(df_processed[col], y)
        else:
            df_processed[col] = pd.get_dummies(df_processed[col])
    
    # First split: Training and validation
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
        "feature_names": feature_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "target_column": "Renewed",
        "original_shape": df.shape,
        "train_shape": train_df.shape,
        "validation_shape": val_df.shape,
        "test_shape": test_df.shape,
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
        print(f"Categorical features: {len(metadata['categorical_columns'])}")
        print(f"Numerical features: {len(metadata['numerical_columns'])}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise e