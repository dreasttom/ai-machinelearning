#!/usr/bin/env python3
"""
Train an Artificial Neural Network (ANN) on a CSV dataset that lives in the
same folder as this script.
NOTE; THIS REQUIRES TENSORFLOW
This script is intentionally written to be:
1. Easy to read
2. Heavily commented
3. Defensive against common data issues
4. Runnable without command-line arguments

Assumptions:
- The CSV file is named "AIS_2023_MINI.csv"
- The CSV file is in the same directory as this script
- The target variable is the LAST column in the dataset

Important note about this dataset:
The provided CSV appears to be saved *without a header row*, so this script
includes logic to detect that case and reload the file correctly.

Required packages:
- pandas
- numpy
- scikit-learn

Install them if needed:
    pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration section
# -----------------------------------------------------------------------------
# The script looks for the dataset in the same folder as the script itself.
DATASET_FILENAME = "AIS_2023_MINI.csv"

# Reproducibility: fixes the random seed for train/test split and neural net.
RANDOM_STATE = 42

# A reasonable threshold used to guess whether a numeric-looking target is
# actually categorical. For example, values like A/B/C are clearly categorical,
# but numeric codes like 0/1/2/3 are categorical too.
MAX_UNIQUE_NUMERIC_VALUES_FOR_CLASSIFICATION = 20


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def print_section(title: str) -> None:
    """Print a clearly separated section header for readability."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def fail(message: str, exit_code: int = 1) -> None:
    """Print a clean error message and exit."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(exit_code)



def detect_and_load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV safely.

    Why this function exists:
    - Some CSVs have a real header row.
    - Some CSVs do not have a header row, but pandas will incorrectly treat the
      first row as headers by default.
    - We therefore attempt an initial read, inspect the resulting column names,
      and if they look suspicious, reload with header=None.

    Returns:
        A pandas DataFrame.
    """
    if not csv_path.exists():
        fail(f"Dataset not found: {csv_path}")

    if not csv_path.is_file():
        fail(f"Path exists but is not a file: {csv_path}")

    try:
        # First attempt: standard read.
        trial_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        fail("The CSV file is empty.")
    except pd.errors.ParserError as exc:
        fail(f"The CSV file could not be parsed. Details: {exc}")
    except UnicodeDecodeError as exc:
        fail(f"The CSV file encoding could not be decoded as text. Details: {exc}")
    except Exception as exc:
        fail(f"Unexpected error while reading CSV: {exc}")

    # Heuristic: if most column names look like actual data values instead of
    # field names, the file probably has no header.
    suspicious_header_score = 0
    for col in trial_df.columns:
        col_str = str(col)

        # Column names that look like timestamps, floats, integers, or generic
        # unnamed placeholders are suspicious in this context.
        if col_str.startswith("Unnamed:"):
            suspicious_header_score += 1
            continue

        # Numeric-looking or timestamp-looking "headers" are often the first
        # row of data accidentally used as a header.
        if any(char.isdigit() for char in col_str):
            suspicious_header_score += 1

    # If a majority of column names look suspicious, reload without a header.
    if len(trial_df.columns) > 0 and suspicious_header_score >= max(1, len(trial_df.columns) // 2):
        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as exc:
            fail(f"Failed to reload the CSV without headers. Details: {exc}")

        # Assign safe generic column names.
        df.columns = [f"column_{i}" for i in range(df.shape[1])]
        return df

    # Otherwise keep the original read.
    return trial_df



def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert object columns that look like datetimes into numeric UNIX
    timestamps.

    Neural networks need numeric input after preprocessing. Converting true
    timestamp columns into numbers often helps.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype != object:
            continue

        # Try converting to datetime. If most values convert successfully, keep it.
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            success_ratio = parsed.notna().mean()

            # Only convert if it really looks like a datetime column.
            if success_ratio >= 0.80:
                # Convert to seconds since epoch, preserving NaT as NaN.
                df[col] = parsed.astype("int64") / 10**9
                # pandas stores NaT as a large negative number when cast directly,
                # so restore invalid positions to NaN.
                df.loc[parsed.isna(), col] = np.nan
        except Exception:
            # Never let a single bad conversion attempt break the pipeline.
            continue

    return df



def infer_task_type(target: pd.Series) -> str:
    """
    Decide whether the problem is classification or regression.

    Rules used:
    - Text/object targets -> classification
    - Boolean targets -> classification
    - Numeric targets with a small number of unique values -> classification
    - Otherwise -> regression
    """
    # Remove missing values before inference to avoid distortion.
    clean_target = target.dropna()

    if clean_target.empty:
        fail("The target column contains only missing values.")

    if pd.api.types.is_bool_dtype(clean_target):
        return "classification"

    if pd.api.types.is_object_dtype(clean_target) or pd.api.types.is_categorical_dtype(clean_target):
        return "classification"

    if pd.api.types.is_numeric_dtype(clean_target):
        unique_count = clean_target.nunique()
        if unique_count <= MAX_UNIQUE_NUMERIC_VALUES_FOR_CLASSIFICATION:
            return "classification"
        return "regression"

    # Safe fallback.
    return "classification"



def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic validation checks before training."""
    if df.empty:
        fail("The dataset has no rows.")

    if df.shape[1] < 2:
        fail("The dataset must contain at least one feature column and one target column.")

    if len(df) < 10:
        fail("The dataset is too small to train a meaningful ANN (fewer than 10 rows).")



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that handles numeric and categorical columns.

    Numeric columns:
    - Median imputation for missing values
    - Standard scaling

    Categorical columns:
    - Most-frequent imputation for missing values
    - One-hot encoding
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_features and not categorical_features:
        fail("No usable feature columns were found after preprocessing.")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor



def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Split the dataframe into features (X) and target (y).

    This script assumes the LAST column is the prediction target.
    """
    target_column = df.columns[-1]
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Remove rows where target is missing because supervised learning cannot
    # train on examples that do not have a known answer.
    non_missing_target_mask = y.notna()
    X = X.loc[non_missing_target_mask].reset_index(drop=True)
    y = y.loc[non_missing_target_mask].reset_index(drop=True)

    if X.empty or y.empty:
        fail("After removing missing target values, no training data remained.")

    task_type = infer_task_type(y)

    print_section("Target Selection")
    print(f"Assumed target column: {target_column}")
    print(f"Inferred task type: {task_type}")
    print(f"Number of usable rows after dropping missing targets: {len(y)}")

    return X, y, task_type



def train_and_evaluate(X: pd.DataFrame, y: pd.Series, task_type: str) -> Pipeline:
    """
    Train the ANN and print evaluation metrics.

    Returns:
        The fitted sklearn Pipeline.
    """
    preprocessor = build_preprocessor(X)

    # Train/test split.
    # For classification, stratify when possible so class proportions are more
    # consistent between train and test sets.
    try:
        stratify = y if task_type == "classification" and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        # If stratification fails because a class is too rare, retry without it.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=None,
        )

    print_section("Train/Test Split")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    if task_type == "classification":
        # MLPClassifier is a feedforward artificial neural network for
        # classification problems.
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )
    else:
        # MLPRegressor is the regression equivalent.
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )

    # Pipeline keeps preprocessing and model together. This is best practice
    # because it guarantees the exact same transformations are applied during
    # training and prediction.
    ann_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print_section("Training Model")
    try:
        ann_pipeline.fit(X_train, y_train)
    except MemoryError:
        fail(
            "Training ran out of memory. This can happen if there are too many "
            "categories after one-hot encoding."
        )
    except ValueError as exc:
        fail(f"Model training failed due to a value/shape issue: {exc}")
    except Exception as exc:
        fail(f"Unexpected model training error: {exc}")

    print("Model training completed successfully.")

    print_section("Evaluation")
    try:
        predictions = ann_pipeline.predict(X_test)
    except Exception as exc:
        fail(f"Prediction failed on the test set: {exc}")

    if task_type == "classification":
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R^2:  {r2:.4f}")

    return ann_pipeline


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------
def main() -> None:
    """Main entry point with top-level error handling."""
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / DATASET_FILENAME

    print_section("Artificial Neural Network Training Script")
    print(f"Script directory: {script_dir}")
    print(f"Expected dataset: {csv_path}")

    # Load data.
    df = detect_and_load_csv(csv_path)

    print_section("Dataset Loaded")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("First 5 rows:")
    print(df.head())

    # Basic validation.
    validate_dataset(df)

    # Try converting datetime-like columns into numeric values.
    df = convert_datetime_columns(df)

    print_section("Column Types After Datetime Processing")
    print(df.dtypes)

    # Split into features and target.
    X, y, task_type = prepare_features_and_target(df)

    # Train and evaluate the ANN.
    _ = train_and_evaluate(X, y, task_type)

    print_section("Done")
    print("The ANN pipeline finished without fatal errors.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Execution was interrupted by the user.")
    except SystemExit:
        # Let explicit exits pass through cleanly.
        raise
    except Exception as exc:
        print("\nA fatal unexpected error occurred.", file=sys.stderr)
        print(f"Error type: {type(exc).__name__}", file=sys.stderr)
        print(f"Error details: {exc}", file=sys.stderr)
        print("\nTraceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
