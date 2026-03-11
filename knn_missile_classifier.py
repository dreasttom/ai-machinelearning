#!/usr/bin/env python3
"""
K-Nearest Neighbors classifier for categorizing missile-related records.

This script was written against the attached `missiles.csv` dataset, but it is
flexible enough to be reused with similar CSV files.

What the script does
--------------------
1. Loads a CSV file.
2. Uses a target/category column (default: TYPE).
3. Builds a mixed-feature machine learning pipeline:
   - Text features from all non-target text columns using TF-IDF.
   - Numeric features extracted from text-like measurement columns such as
     MASS, LENGTH, WEIGHT, and DIAMETER.
4. Trains a K-Nearest Neighbors classifier.
5. Evaluates the model with standard metrics.
6. Produces graphical outputs:
   - Confusion matrix heatmap
   - Target class distribution chart
   - 2D projection of the feature space using Truncated SVD



Example usage
-------------
python knn_missile_classifier.py --csv /path/to/missiles.csv
python knn_missile_classifier.py --csv /path/to/missiles.csv --target TYPE --neighbors 7

Dependencies
------------
pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when the input data fails validation checks."""


class ModelingError(Exception):
    """Raised when the model cannot be trained or evaluated."""


# ---------------------------------------------------------------------------
# Small configuration container
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Holds runtime configuration parsed from command-line arguments."""

    csv_path: str
    target_column: str = "TYPE"
    neighbors: int = 5
    test_size: float = 0.25
    random_state: int = 42
    max_features: int = 2000
    output_dir: str = "knn_output"
    min_samples_per_class: int = 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def safe_make_dir(path: str) -> None:
    """Create an output directory if it does not already exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory '{path}': {exc}") from exc



def extract_first_number(value: object) -> float:
    """
    Extract the first numeric value from a messy text cell.

    Why this is needed:
    The source CSV stores many measurements as strings such as:
        '1,200 kg'
        '600 cm (236 in)'
        '1.59 metres (5.2 ft)'

    For a simple KNN baseline, extracting the first number gives us a usable
    numeric signal without needing domain-specific unit normalization.

    Returns:
        float if a number is found, else np.nan
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan

        text = str(value)

        # Replace non-breaking spaces and remove commas in numbers like 1,200.
        text = text.replace("\xa0", " ").replace(",", "")

        # Find the first integer or decimal number, optionally signed.
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            return float(match.group())
        return np.nan
    except (ValueError, TypeError, AttributeError):
        # Fail softly on badly formed input. Returning NaN lets the imputer deal with it.
        return np.nan



def summarize_exception(exc: BaseException) -> str:
    """Convert exceptions into concise, user-friendly messages."""
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Feature engineering transformer
# ---------------------------------------------------------------------------

class DatasetPreparer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that converts the original dataframe into a compact set
    of model-ready columns.

    Output columns:
        text_features  -> a single combined text field created by concatenating
                          all non-target categorical/text columns
        num_mass       -> numeric signal extracted from MASS
        num_length     -> numeric signal extracted from LENGTH
        num_weight     -> numeric signal extracted from WEIGHT
        num_diameter   -> numeric signal extracted from DIAMETER

    Why we combine text columns:
        KNN needs a numeric feature space. TF-IDF converts text into vectors.
        Concatenating the descriptive fields gives the model contextual clues
        like origin, guidance, propellant, engine type, and missile name.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.numeric_hint_columns = ["MASS", "LENGTH", "WEIGHT", "DIAMETER"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DatasetPreparer":
        # No learned parameters are needed, but sklearn expects fit() to exist.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input to DatasetPreparer must be a pandas DataFrame.")

            df = X.copy()

            # Decide which columns should contribute to the text representation.
            # Exclude target column and obvious ID columns from the text blob.
            exclude = {self.target_column, "ID"}
            text_columns = [col for col in df.columns if col not in exclude]

            # Convert all selected columns to strings, replacing missing values
            # with empty strings before concatenating them together row by row.
            text_df = df[text_columns].fillna("").astype(str)
            combined_text = text_df.apply(lambda row: " ".join(v.strip() for v in row if v.strip()), axis=1)

            prepared = pd.DataFrame({"text_features": combined_text})

            # Add lightweight numeric features extracted from known measurement fields.
            for col in self.numeric_hint_columns:
                out_col = f"num_{col.lower()}"
                if col in df.columns:
                    prepared[out_col] = df[col].apply(extract_first_number)
                else:
                    # If a column is missing in another dataset, preserve pipeline shape.
                    prepared[out_col] = np.nan

            return prepared
        except Exception as exc:
            raise DataValidationError(f"Failed while preparing features: {summarize_exception(exc)}") from exc


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


def load_and_validate_data(csv_path: str, target_column: str, min_samples_per_class: int) -> pd.DataFrame:
    """
    Load the dataset and perform defensive validation.

    Important validation steps:
    - File exists and is readable.
    - CSV is not empty.
    - Target column exists.
    - Target column has non-null values.
    - Extremely rare target classes are filtered out to avoid train/test split
      failures and impossible evaluation scenarios.
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise DataValidationError("The CSV file is empty. Cannot train a model.")

        if target_column not in df.columns:
            raise DataValidationError(
                f"Target column '{target_column}' was not found. Available columns: {list(df.columns)}"
            )

        # Remove rows without a valid target label.
        df = df.dropna(subset=[target_column]).copy()
        df[target_column] = df[target_column].astype(str).str.strip()
        df = df[df[target_column] != ""]

        if df.empty:
            raise DataValidationError(
                f"No usable rows remain after removing null/blank target values in '{target_column}'."
            )

        # Filter rare classes. KNN can technically fit them, but evaluation and
        # stratified splitting become fragile if a class appears only once.
        class_counts = df[target_column].value_counts()
        keep_classes = class_counts[class_counts >= min_samples_per_class].index
        filtered_df = df[df[target_column].isin(keep_classes)].copy()

        if filtered_df.empty:
            raise DataValidationError(
                "All classes were filtered out as too rare. Lower min_samples_per_class or inspect the data."
            )

        if filtered_df[target_column].nunique() < 2:
            raise DataValidationError(
                "Need at least two target classes to perform classification."
            )

        return filtered_df

    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The CSV appears to be empty or malformed.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Could not parse the CSV file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"File encoding problem while reading the CSV: {exc}") from exc


# ---------------------------------------------------------------------------
# Modeling pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline(target_column: str, neighbors: int, max_features: int) -> Pipeline:
    """
    Build a full sklearn pipeline.

    Pipeline structure:
        DatasetPreparer
            -> creates text_features + extracted numeric columns
        ColumnTransformer
            -> text_features transformed with TF-IDF
            -> numeric columns imputed + scaled
        KNeighborsClassifier

    Why scaling matters:
        KNN is distance-based, so unscaled numeric values can distort distances.
    """
    try:
        if neighbors < 1:
            raise ValueError("neighbors must be at least 1")
        if max_features < 100:
            raise ValueError("max_features should be at least 100 for useful text coverage")

        numeric_columns = ["num_mass", "num_length", "num_weight", "num_diameter"]

        numeric_pipeline = Pipeline(
            steps=[
                # Fill missing numeric values with the median.
                ("imputer", SimpleImputer(strategy="median")),
                # Standardize numeric columns for fair distance calculations.
                ("scaler", StandardScaler()),
            ]
        )

        text_pipeline = Pipeline(
            steps=[
                # Ensure the single text column becomes a 1D iterable of strings.
                ("flatten", FunctionTransformer(lambda x: x.squeeze(), validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        stop_words="english",
                        ngram_range=(1, 2),
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_pipeline, ["text_features"]),
                ("num", numeric_pipeline, numeric_columns),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("prepare", DatasetPreparer(target_column=target_column)),
                ("preprocess", preprocessor),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=neighbors,
                        weights="distance",  # Distance weighting often helps with mixed/noisy data.
                        metric="cosine",     # Cosine distance usually behaves better in sparse text spaces.
                    ),
                ),
            ]
        )

        return pipeline

    except Exception as exc:
        raise ModelingError(f"Could not build modeling pipeline: {summarize_exception(exc)}") from exc


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_class_distribution(y: pd.Series, output_path: str) -> None:
    """Save a bar chart showing class frequency in the cleaned dataset."""
    try:
        counts = y.value_counts().sort_values(ascending=False).head(20)

        plt.figure(figsize=(12, 7))
        counts.plot(kind="bar")
        plt.title("Top 20 Class Counts in Target Column")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to generate class distribution plot: {exc}") from exc



def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, labels: List[str], output_path: str) -> None:
    """
    Save a confusion matrix heatmap.

    To keep the plot readable on datasets with many classes, this function only
    uses labels observed in either y_true or y_pred for the test split.
    """
    try:
        observed_labels = sorted(set(y_true).union(set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=observed_labels)

        fig, ax = plt.subplots(figsize=(12, 10))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=observed_labels)
        display.plot(ax=ax, xticks_rotation=90, colorbar=False)
        plt.title("KNN Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate confusion matrix plot: {exc}") from exc



def plot_svd_projection(fitted_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, output_path: str) -> None:
    """
    Produce a 2D visualization of the high-dimensional feature space.

    We transform the prepared data using the already-fitted preprocessing steps,
    then reduce dimensions with TruncatedSVD so the user can visually inspect
    how classes separate in 2D.
    """
    try:
        # Access the fitted preparation and preprocessing stages directly.
        prepared = fitted_pipeline.named_steps["prepare"].transform(X)
        features = fitted_pipeline.named_steps["preprocess"].transform(prepared)

        # TruncatedSVD works well with sparse TF-IDF matrices.
        reducer = TruncatedSVD(n_components=2, random_state=42)
        coords = reducer.fit_transform(features)

        # Plot only the most frequent classes for readability.
        top_classes = y.value_counts().head(10).index
        mask = y.isin(top_classes)

        plt.figure(figsize=(12, 8))
        for cls in top_classes:
            cls_mask = (y == cls) & mask
            plt.scatter(coords[cls_mask, 0], coords[cls_mask, 1], s=28, alpha=0.7, label=str(cls))

        plt.title("2D Truncated SVD Projection of Feature Space (Top 10 Classes)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to generate SVD projection plot: {exc}") from exc


# ---------------------------------------------------------------------------
# Main execution logic
# ---------------------------------------------------------------------------


def run_knn_classification(config: RunConfig) -> None:
    """Main orchestration function for the end-to-end workflow."""
    try:
        safe_make_dir(config.output_dir)

        print("Loading and validating data...")
        df = load_and_validate_data(
            csv_path=config.csv_path,
            target_column=config.target_column,
            min_samples_per_class=config.min_samples_per_class,
        )

        X = df.drop(columns=[config.target_column])
        y = df[config.target_column]

        print(f"Rows after cleaning/filtering: {len(df):,}")
        print(f"Number of classes: {y.nunique()}")

        # Stratify when possible so train/test keep a similar class distribution.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=y,
            )
        except ValueError:
            # Fall back gracefully if stratification fails for any edge case.
            print("Warning: stratified split failed; falling back to non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=None,
            )

        # KNN requires n_neighbors <= number of training samples.
        effective_neighbors = min(config.neighbors, len(X_train))
        if effective_neighbors != config.neighbors:
            print(
                f"Adjusted neighbors from {config.neighbors} to {effective_neighbors} "
                f"because the training set is smaller than requested."
            )

        pipeline = build_pipeline(
            target_column=config.target_column,
            neighbors=effective_neighbors,
            max_features=config.max_features,
        )

        print("Training KNN model...")
        pipeline.fit(X_train, y_train)

        print("Generating predictions...")
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        print("\n" + "=" * 80)
        print("MODEL RESULTS")
        print("=" * 80)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification report:\n")
        print(report)

        # Write a text report to disk as a durable artifact.
        metrics_path = os.path.join(config.output_dir, "metrics_report.txt")
        with open(metrics_path, "w", encoding="utf-8") as report_file:
            report_file.write("KNN Classification Results\n")
            report_file.write("=" * 40 + "\n")
            report_file.write(f"CSV: {config.csv_path}\n")
            report_file.write(f"Target column: {config.target_column}\n")
            report_file.write(f"Neighbors: {effective_neighbors}\n")
            report_file.write(f"Rows after cleaning: {len(df)}\n")
            report_file.write(f"Class count: {y.nunique()}\n")
            report_file.write(f"Accuracy: {accuracy:.4f}\n\n")
            report_file.write(report)

        # Save graphical outputs.
        plot_class_distribution(
            y=y,
            output_path=os.path.join(config.output_dir, "class_distribution.png"),
        )
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=sorted(y.unique().tolist()),
            output_path=os.path.join(config.output_dir, "confusion_matrix.png"),
        )
        plot_svd_projection(
            fitted_pipeline=pipeline,
            X=X,
            y=y,
            output_path=os.path.join(config.output_dir, "svd_projection.png"),
        )

        print("\nArtifacts written to:")
        print(f"  - {metrics_path}")
        print(f"  - {os.path.join(config.output_dir, 'class_distribution.png')}")
        print(f"  - {os.path.join(config.output_dir, 'confusion_matrix.png')}")
        print(f"  - {os.path.join(config.output_dir, 'svd_projection.png')}")

    except (DataValidationError, ModelingError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"ERROR: {summarize_exception(exc)}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # Catch-all to make debugging easier in unexpected situations.
        print("UNEXPECTED ERROR:", summarize_exception(exc), file=sys.stderr)
        print("Detailed traceback follows:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Iterable[str]] = None) -> RunConfig:
    """Parse command-line arguments into a RunConfig instance."""
    parser = argparse.ArgumentParser(description="Train a KNN classifier on a CSV dataset.")
    parser.add_argument("--csv", dest="csv_path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", dest="target_column", default="TYPE", help="Target/category column.")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for KNN.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of rows used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-features", type=int, default=2000, help="Max TF-IDF features.")
    parser.add_argument("--output-dir", default="knn_output", help="Directory for reports and plots.")
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=2,
        help="Minimum number of examples required to keep a class.",
    )

    args = parser.parse_args(argv)

    # Defensive argument checks keep failures explicit and early.
    if not 0.05 <= args.test_size < 0.95:
        raise ValueError("--test-size must be between 0.05 and 0.95")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be >= 1")
    if args.min_samples_per_class < 1:
        raise ValueError("--min-samples-per-class must be >= 1")

    return RunConfig(
        csv_path=args.csv_path,
        target_column=args.target_column,
        neighbors=args.neighbors,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        output_dir=args.output_dir,
        min_samples_per_class=args.min_samples_per_class,
    )


if __name__ == "__main__":
    try:
        config = parse_args()
        run_knn_classification(config)
    except Exception as exc:
        print(f"Startup error: {summarize_exception(exc)}", file=sys.stderr)
        sys.exit(1)
