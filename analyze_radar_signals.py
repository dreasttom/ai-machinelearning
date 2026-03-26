#!/usr/bin/env python3
"""
analyze_radar_signals.py

Purpose
-------
A heavily commented example script for working with the radar_signals.csv dataset.
The script assumes the CSV file lives in the SAME folder as this script.

What it does
------------
1. Loads the dataset safely.
2. Validates that the expected columns exist.
3. Cleans obvious data issues (missing values, duplicate pulse IDs, etc.).
4. Prints a readable exploratory analysis summary.
5. Computes per-class feature averages.
6. Runs a simple Leave-One-Out nearest-centroid classifier as a lightweight baseline.
7. Writes analysis outputs to CSV files in the same folder.

Why a simple baseline?
----------------------
The dataset is very small, so a simple, transparent baseline is often more appropriate
than a complex model. This script focuses on robustness, readability, and being easy
to modify for coursework or experimentation.

Output files
------------
- radar_summary_statistics.csv
- radar_class_centroids.csv
- radar_predictions_leave_one_out.csv
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration / constants
# -----------------------------
# The script assumes the CSV is in the same folder as the script itself.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_NAME = "radar_signals.csv"
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, DEFAULT_CSV_NAME)

# These are the columns we expect to exist.
# If one is missing, the script will fail early with a clear message.
REQUIRED_COLUMNS = [
    "pulse_id",
    "frequency_mhz",
    "amplitude_db",
    "pulse_width_us",
    "pri_us",
    "doppler_hz",
    "snr_db",
    "range_km",
    "azimuth_deg",
    "elevation_deg",
    "label",
]

# Numeric features to analyze / model.
NUMERIC_FEATURES = [
    "frequency_mhz",
    "amplitude_db",
    "pulse_width_us",
    "pri_us",
    "doppler_hz",
    "snr_db",
    "range_km",
    "azimuth_deg",
    "elevation_deg",
]


# -----------------------------
# Utility functions
# -----------------------------
def print_section(title: str) -> None:
    """Pretty-print a section header to make console output easier to read."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fail(message: str, exit_code: int = 1) -> None:
    """
    Print an error message to stderr and exit.

    This centralizes the error-exit behavior so the script remains consistent.
    """
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(exit_code)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset from disk with defensive error handling.

    Raises:
        FileNotFoundError: if the CSV path does not exist.
        ValueError: if the CSV cannot be parsed or is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at: {csv_path}\n"
            f"Expected the CSV to be in the same folder as this script."
        )

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file exists but is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "The CSV file could not be parsed. "
            "Please verify it is a valid comma-separated file."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Unexpected error while reading the CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV loaded successfully, but it contains no rows.")

    return df


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Ensure all required columns are present.

    Raises:
        ValueError: if any expected columns are missing.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
        )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning and normalization.

    Cleaning steps:
    - Remove exact duplicate rows.
    - Strip whitespace from string columns.
    - Drop rows missing target labels.
    - Attempt to coerce numeric feature columns to numeric.
    - Drop rows that still contain missing numeric values after coercion.

    Returns:
        A cleaned copy of the dataframe.
    """
    cleaned = df.copy()

    # Remove exact duplicates.
    cleaned = cleaned.drop_duplicates()

    # Normalize string columns in a defensive way.
    for column in ["pulse_id", "label"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    # Drop rows where the label is missing or blank.
    cleaned["label"] = cleaned["label"].replace({"": np.nan, "nan": np.nan})
    cleaned = cleaned.dropna(subset=["label"])

    # Convert numeric columns to numeric values. Non-numeric entries become NaN.
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Keep track of rows before dropping incomplete numeric data.
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=NUMERIC_FEATURES)
    dropped = before_drop - len(cleaned)

    if dropped > 0:
        print(
            f"Note: Dropped {dropped} row(s) due to missing or invalid numeric values."
        )

    # Check for duplicate pulse IDs, which may indicate data quality issues.
    if cleaned["pulse_id"].duplicated().any():
        duplicate_ids = cleaned.loc[cleaned["pulse_id"].duplicated(), "pulse_id"].tolist()
        print(
            "Warning: Duplicate pulse_id values detected: "
            + ", ".join(map(str, duplicate_ids))
        )

    if cleaned.empty:
        raise ValueError("No usable rows remain after cleaning.")

    return cleaned


def dataset_overview(df: pd.DataFrame) -> None:
    """Print a human-friendly overview of the dataset."""
    print_section("DATASET OVERVIEW")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print_section("FIRST 5 ROWS")
    print(df.head().to_string(index=False))

    print_section("LABEL DISTRIBUTION")
    label_counts = df["label"].value_counts(dropna=False)
    print(label_counts.to_string())

    print_section("MISSING VALUES BY COLUMN")
    print(df.isna().sum().to_string())


def summarize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a numeric summary table with standard descriptive statistics.
    """
    summary = df[NUMERIC_FEATURES].describe().T
    summary["variance"] = df[NUMERIC_FEATURES].var()
    summary["missing_count"] = df[NUMERIC_FEATURES].isna().sum()
    return summary


def compute_class_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean value of each numeric feature for each class label.

    This is useful both for EDA and for a simple nearest-centroid classifier.
    """
    return df.groupby("label")[NUMERIC_FEATURES].mean().sort_index()


def zscore_standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize train and test data using only the training statistics.

    This prevents accidental data leakage from the test sample into training.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    means = train_df[features].mean()
    stds = train_df[features].std(ddof=0).replace(0, 1.0)

    train_scaled[features] = (train_df[features] - means) / stds
    test_scaled[features] = (test_df[features] - means) / stds

    return train_scaled, test_scaled


def nearest_centroid_predict(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    features: List[str],
) -> str:
    """
    Predict a label for one test row using nearest class centroid.

    Distance metric:
        Euclidean distance between the standardized test point and each
        class centroid derived from the standardized training data.

    Raises:
        ValueError: if the training data cannot form at least one class centroid.
    """
    if train_df.empty:
        raise ValueError("Training dataframe is empty; cannot predict.")

    centroids = train_df.groupby("label")[features].mean()
    if centroids.empty:
        raise ValueError("No class centroids available; check training data.")

    # Compute Euclidean distance from the test sample to each class centroid.
    distances: Dict[str, float] = {}
    for label, centroid in centroids.iterrows():
        diff = test_row[features] - centroid[features]
        distance = float(np.sqrt(np.sum(np.square(diff))))
        distances[label] = distance

    # Select the label with minimum distance.
    predicted_label = min(distances, key=distances.get)
    return predicted_label


def run_leave_one_out_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Leave-One-Out cross-validation with a nearest-centroid classifier.

    Why Leave-One-Out?
    - The dataset is tiny.
    - It lets us evaluate the baseline on every sample.

    Returns:
        DataFrame of predictions with actual/predicted labels and correctness.
    """
    results = []

    for test_index in df.index:
        # Partition data into training rows and exactly one test row.
        test_df = df.loc[[test_index]].copy()
        train_df = df.drop(index=test_index).copy()

        # Standardize using training statistics only.
        train_scaled, test_scaled = zscore_standardize(
            train_df=train_df,
            test_df=test_df,
            features=NUMERIC_FEATURES,
        )

        test_row = test_scaled.iloc[0]

        try:
            predicted = nearest_centroid_predict(
                train_df=train_scaled,
                test_row=test_row,
                features=NUMERIC_FEATURES,
            )
        except Exception as exc:
            # Store a failed prediction rather than crashing the entire run.
            predicted = f"PREDICTION_ERROR: {exc}"

        actual = test_df.iloc[0]["label"]

        results.append(
            {
                "pulse_id": test_df.iloc[0]["pulse_id"],
                "actual_label": actual,
                "predicted_label": predicted,
                "correct": actual == predicted,
            }
        )

    return pd.DataFrame(results)


def confusion_matrix_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a confusion matrix from actual vs predicted labels.

    We use pandas crosstab so this stays lightweight and dependency-free.
    """
    return pd.crosstab(
        results_df["actual_label"],
        results_df["predicted_label"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )


def write_output(df: pd.DataFrame, output_path: str, description: str) -> None:
    """
    Safely write a dataframe to CSV with explicit error handling.
    """
    try:
        df.to_csv(output_path, index=True)
        print(f"Saved {description}: {output_path}")
    except PermissionError as exc:
        print(
            f"Warning: Could not write {description} due to a permission error: {exc}",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"Warning: OS error while writing {description}: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"Warning: Unexpected error while writing {description}: {exc}",
            file=sys.stderr,
        )


def main() -> None:
    """
    Main program entry point.

    A top-level try/except is included so that unexpected failures produce
    a useful stack trace instead of a silent or cryptic crash.
    """
    try:
        print_section("RADAR SIGNAL ANALYSIS SCRIPT")

        # Allow the user to optionally pass a custom CSV path.
        # If no path is provided, default to "radar_signals.csv" in the same folder.
        csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
        print(f"Using dataset: {csv_path}")

        # Step 1: Load and validate the input file.
        df = load_dataset(csv_path)
        validate_columns(df, REQUIRED_COLUMNS)

        # Step 2: Clean the data.
        df = clean_dataset(df)

        # Step 3: Print overview and descriptive statistics.
        dataset_overview(df)

        numeric_summary = summarize_numeric_features(df)
        print_section("NUMERIC FEATURE SUMMARY")
        print(numeric_summary.to_string())

        class_centroids = compute_class_centroids(df)
        print_section("PER-CLASS FEATURE MEANS (CENTROIDS)")
        print(class_centroids.to_string())

        # Step 4: Run a simple Leave-One-Out baseline classifier.
        results_df = run_leave_one_out_baseline(df)

        print_section("LEAVE-ONE-OUT PREDICTIONS")
        print(results_df.to_string(index=False))

        accuracy = results_df["correct"].mean()
        print_section("BASELINE ACCURACY")
        print(f"Accuracy: {accuracy:.2%}")

        conf_matrix = confusion_matrix_table(results_df)
        print_section("CONFUSION MATRIX")
        print(conf_matrix.to_string())

        # Step 5: Save outputs for later inspection.
        write_output(
            numeric_summary,
            os.path.join(SCRIPT_DIR, "radar_summary_statistics.csv"),
            "summary statistics CSV",
        )
        write_output(
            class_centroids,
            os.path.join(SCRIPT_DIR, "radar_class_centroids.csv"),
            "class centroid CSV",
        )
        write_output(
            results_df.set_index("pulse_id"),
            os.path.join(SCRIPT_DIR, "radar_predictions_leave_one_out.csv"),
            "prediction results CSV",
        )

        print_section("DONE")
        print("Analysis completed successfully.")

    except FileNotFoundError as exc:
        fail(str(exc), exit_code=2)

    except ValueError as exc:
        fail(str(exc), exit_code=3)

    except KeyboardInterrupt:
        fail("Execution interrupted by user.", exit_code=130)

    except Exception as exc:
        # Print a friendly message plus full traceback for debugging.
        print("An unexpected error occurred.", file=sys.stderr)
        print(f"Error type: {type(exc).__name__}", file=sys.stderr)
        print(f"Error details: {exc}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(99)


if __name__ == "__main__":
    main()
