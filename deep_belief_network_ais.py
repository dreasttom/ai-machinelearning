#!/usr/bin/env python3
"""
Deep Belief Network (DBN) example for the AIS_2023_Set_One.csv dataset.

This script builds a practical "deep belief network style" model using stacked
Restricted Boltzmann Machines (RBMs) for unsupervised feature learning and a
Logistic Regression classifier for supervised prediction.

Why this design?
----------------
A classical DBN is formed by stacking RBMs. Scikit-learn provides a stable,
well-tested BernoulliRBM implementation, which lets us create a robust and
maintainable DBN-style pipeline without introducing fragile research code.

What the script does
--------------------
1. Loads the AIS CSV file safely with validation and error handling.
2. Uses `VesselType` as the prediction target.
3. Cleans and preprocesses numeric and categorical columns.
4. Samples the data (configurable) so training remains practical on large files.
5. Trains multiple stacked RBMs.
6. Trains a Logistic Regression classifier on the learned RBM features.
7. Saves evaluation metrics and several plots:
   - class distribution
   - confusion matrix
   - PCA scatter plot of learned DBN features
   - feature activation histograms after each RBM layer

Usage
-----
python deep_belief_network_ais.py --input /path/AIS_2023_Set_One.csv --output-dir dbn_output
If its in the same path this should work
python deep_belief_network_ais.py --input .\AIS_2023_Set_One.csv

Optional arguments are available; run:
python deep_belief_network_ais.py --help

This script uses AIS_2023_Set_One.csv which is an abbreviated version of data sets found at  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/index.html
NOTE: THIS SCRIPT IS SLOW TO RUN

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class DBNConfig:
    """Configuration container for the training process."""

    input_path: str
    output_dir: str = "dbn_output"
    target_column: str = "VesselType"
    max_rows: int = 50000
    random_state: int = 42
    test_size: float = 0.2
    min_class_count: int = 100
    rbm_hidden_layers: Tuple[int, ...] = (256, 128)
    rbm_learning_rate: float = 0.01
    rbm_batch_size: int = 64
    rbm_n_iter: int = 10
    logistic_max_iter: int = 1000


# -----------------------------
# Utility helpers
# -----------------------------
def setup_logging(output_dir: Path) -> None:
    """Configure console + file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def parse_hidden_layers(text: str) -> Tuple[int, ...]:
    """Parse a comma-separated hidden-layer specification like '256,128'."""
    try:
        layers = tuple(int(part.strip()) for part in text.split(",") if part.strip())
        if not layers:
            raise ValueError("At least one hidden layer size must be provided.")
        if any(size <= 0 for size in layers):
            raise ValueError("Hidden layer sizes must all be positive integers.")
        return layers
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid hidden layer specification: {text!r}") from exc


# -----------------------------
# DBN-style classifier
# -----------------------------
class DeepBeliefNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    A practical DBN-style classifier using stacked Bernoulli RBMs.

    Training flow:
    1. Fit RBM layer 1 on the preprocessed feature matrix.
    2. Transform the features using RBM layer 1.
    3. Fit RBM layer 2 on the transformed output from layer 1.
    4. Continue for all RBM layers.
    5. Fit a Logistic Regression classifier on the final learned features.

    This is a standard and interpretable way to approximate a Deep Belief Network
    workflow in production-friendly Python code.
    """

    def __init__(
        self,
        hidden_layers: Sequence[int] = (256, 128),
        learning_rate: float = 0.01,
        batch_size: int = 64,
        n_iter: int = 10,
        logistic_max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.logistic_max_iter = logistic_max_iter
        self.random_state = random_state

        self.rbms_: List[BernoulliRBM] = []
        self.classifier_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBeliefNetworkClassifier":
        """Fit stacked RBMs and then a final Logistic Regression classifier."""
        if X is None or len(X) == 0:
            raise ValueError("Training features are empty; cannot fit DBN.")
        if y is None or len(y) == 0:
            raise ValueError("Training labels are empty; cannot fit DBN.")
        if len(X) != len(y):
            raise ValueError("Feature and label lengths do not match.")

        X_current = np.asarray(X, dtype=np.float32)
        y_current = np.asarray(y)

        self.rbms_ = []
        logging.info("Starting DBN training with hidden layers: %s", self.hidden_layers)

        for layer_index, n_hidden in enumerate(self.hidden_layers, start=1):
            logging.info(
                "Fitting RBM layer %d/%d with %d hidden units...",
                layer_index,
                len(self.hidden_layers),
                n_hidden,
            )

            rbm = BernoulliRBM(
                n_components=n_hidden,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                n_iter=self.n_iter,
                verbose=True,
                random_state=self.random_state + layer_index,
            )
            rbm.fit(X_current)
            X_current = rbm.transform(X_current)
            self.rbms_.append(rbm)

            logging.info(
                "Completed RBM layer %d. Output shape is now %s.",
                layer_index,
                X_current.shape,
            )

        logging.info("Fitting Logistic Regression on learned DBN features...")
        self.classifier_ = LogisticRegression(
            max_iter=self.logistic_max_iter,
            multi_class="auto",
            solver="lbfgs",
            n_jobs=None,
            random_state=self.random_state,
        )
        self.classifier_.fit(X_current, y_current)
        self.classes_ = getattr(self.classifier_, "classes_", None)
        logging.info("DBN training complete.")
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features through all RBM layers and return learned features."""
        if not self.rbms_:
            raise RuntimeError("The DBN has not been fitted yet.")

        X_current = np.asarray(X, dtype=np.float32)
        for rbm in self.rbms_:
            X_current = rbm.transform(X_current)
        return X_current

    def transform_through_layers(self, X: np.ndarray) -> List[np.ndarray]:
        """Return intermediate outputs after each RBM layer for diagnostics/plots."""
        if not self.rbms_:
            raise RuntimeError("The DBN has not been fitted yet.")

        X_current = np.asarray(X, dtype=np.float32)
        outputs: List[np.ndarray] = []
        for rbm in self.rbms_:
            X_current = rbm.transform(X_current)
            outputs.append(X_current)
        return outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target classes using the learned DBN features."""
        if self.classifier_ is None:
            raise RuntimeError("The classifier has not been fitted yet.")
        features = self.transform_features(X)
        return self.classifier_.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities when supported by the final classifier."""
        if self.classifier_ is None:
            raise RuntimeError("The classifier has not been fitted yet.")
        features = self.transform_features(X)
        return self.classifier_.predict_proba(features)


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def validate_input_file(path: Path) -> None:
    """Validate that the input file exists and is readable."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")


def load_and_prepare_data(config: DBNConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the AIS CSV and prepare features/labels.

    Design choices:
    - We predict `VesselType`, a naturally occurring labeled column in the dataset.
    - Rows missing the target are removed because supervised classification needs
      known labels.
    - Rare classes are removed to reduce instability and severe class imbalance.
    - `BaseDateTime` is converted to engineered time features.
    """
    input_path = Path(config.input_path)
    validate_input_file(input_path)

    logging.info("Loading CSV data from %s", input_path)
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The CSV file could not be parsed correctly.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV file encoding could not be decoded as text.") from exc

    if df.empty:
        raise ValueError("Loaded DataFrame is empty after reading the CSV file.")

    if config.target_column not in df.columns:
        raise KeyError(
            f"Target column {config.target_column!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )

    logging.info("Initial dataset shape: %s", df.shape)

    # Parse BaseDateTime into usable numeric features if available.
    if "BaseDateTime" in df.columns:
        logging.info("Engineering time-based features from BaseDateTime.")
        dt = pd.to_datetime(df["BaseDateTime"], errors="coerce")
        df["Hour"] = dt.dt.hour
        df["DayOfWeek"] = dt.dt.dayofweek
        df["Month"] = dt.dt.month
        # We intentionally drop the raw datetime string after feature extraction.
        df = df.drop(columns=["BaseDateTime"])

    # Define the label and drop rows where it is missing.
    y = df[config.target_column].copy()
    valid_target_mask = y.notna()
    df = df.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].copy()

    if df.empty:
        raise ValueError(
            f"No rows remain after removing missing labels from target {config.target_column!r}."
        )

    # Convert target labels into a cleaner categorical string representation.
    # VesselType values are often stored as floats like 31.0, 80.0, etc.
    y = y.astype(float).round().astype(int).astype(str)

    # Remove rare classes so stratified training is stable and meaningful.
    class_counts = y.value_counts()
    allowed_classes = class_counts[class_counts >= config.min_class_count].index
    keep_mask = y.isin(allowed_classes)
    df = df.loc[keep_mask].copy()
    y = y.loc[keep_mask].copy()

    if y.nunique() < 2:
        raise ValueError(
            "After filtering rare classes, fewer than two target classes remain. "
            "Reduce --min-class-count or inspect the dataset."
        )

    # Drop the target from the feature frame.
    X = df.drop(columns=[config.target_column])

    # Sample rows to keep training feasible on very large AIS datasets.
    if config.max_rows is not None and len(X) > config.max_rows:
        logging.info(
            "Sampling %d rows from %d total labeled rows for practical training.",
            config.max_rows,
            len(X),
        )
        sampled_index = (
            pd.concat([X, y.rename(config.target_column)], axis=1)
            .groupby(config.target_column, group_keys=False)
            .apply(
                lambda part: part.sample(
                    n=max(1, int(round(config.max_rows * len(part) / len(X)))),
                    replace=False,
                    random_state=config.random_state,
                )
            )
            .index
        )
        sampled_index = list(dict.fromkeys(sampled_index))
        X = X.loc[sampled_index]
        y = y.loc[sampled_index]

        # If rounding caused the sample size to drift upward, trim safely.
        if len(X) > config.max_rows:
            trim_index = y.groupby(y, group_keys=False).sample(
                frac=1.0,
                random_state=config.random_state,
            ).index[: config.max_rows]
            X = X.loc[trim_index]
            y = y.loc[trim_index]

    logging.info("Prepared feature matrix shape: %s", X.shape)
    logging.info("Number of classes retained: %d", y.nunique())
    return X.reset_index(drop=True), y.reset_index(drop=True)



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for mixed numeric and categorical AIS data."""
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found after preprocessing.")

    logging.info("Numeric columns: %s", numeric_columns)
    logging.info("Categorical columns: %s", categorical_columns)

    numeric_pipeline = Pipeline(
        steps=[
            # Median imputation is robust for noisy real-world telemetry.
            ("imputer", SimpleImputer(strategy="median")),
            # Standardize numeric data before later min-max scaling.
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Missing categorical values are common in AIS metadata.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encoding makes string data usable for RBMs and linear models.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor


# -----------------------------
# Plotting helpers
# -----------------------------
def save_class_distribution(y: pd.Series, output_dir: Path) -> None:
    """Save a class distribution bar chart."""
    plt.figure(figsize=(12, 6))
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Target Class Distribution (VesselType)")
    plt.xlabel("VesselType")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()



def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Save a confusion matrix figure."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("DBN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)



def save_pca_feature_plot(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Project learned DBN features into 2D with PCA and save a scatter plot."""
    if features.shape[1] < 2:
        logging.warning("Skipping PCA plot because learned feature dimensionality is < 2.")
        return

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # To keep the plot readable, only show up to 12 legend entries.
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            components[mask, 0],
            components[mask, 1],
            s=10,
            alpha=0.5,
            label=label if idx < 12 else None,
        )

    plt.title("PCA Projection of Learned DBN Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if len(unique_labels) <= 12:
        plt.legend(title="VesselType", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "dbn_feature_pca.png", dpi=150)
    plt.close()



def save_layer_activation_histograms(layer_outputs: List[np.ndarray], output_dir: Path) -> None:
    """Save activation histograms for each RBM layer output."""
    for idx, output in enumerate(layer_outputs, start=1):
        plt.figure(figsize=(10, 6))
        plt.hist(output.ravel(), bins=50)
        plt.title(f"RBM Layer {idx} Activation Distribution")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"rbm_layer_{idx}_activations.png", dpi=150)
        plt.close()


# -----------------------------
# Main training function
# -----------------------------
def train_and_evaluate(config: DBNConfig) -> None:
    """End-to-end training, evaluation, and artifact generation."""
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    logging.info("Training configuration: %s", asdict(config))

    # Step 1: Load and prepare data.
    X, y = load_and_prepare_data(config)
    save_class_distribution(y, output_dir)

    # Step 2: Train/test split.
    logging.info("Creating train/test split with test_size=%.3f", config.test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    logging.info("Training rows: %d | Test rows: %d", len(X_train), len(X_test))

    # Step 3: Preprocess features.
    preprocessor = build_preprocessor(X_train)
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    except Exception as exc:
        raise RuntimeError("Preprocessing failed. Check feature types and missing values.") from exc

    # RBMs expect values in [0, 1] because they model Bernoulli-like visible units.
    minmax = MinMaxScaler()
    X_train_scaled = minmax.fit_transform(X_train_processed).astype(np.float32)
    X_test_scaled = minmax.transform(X_test_processed).astype(np.float32)

    logging.info(
        "Preprocessed feature shapes | train: %s | test: %s",
        X_train_scaled.shape,
        X_test_scaled.shape,
    )

    # Step 4: Fit DBN.
    dbn = DeepBeliefNetworkClassifier(
        hidden_layers=config.rbm_hidden_layers,
        learning_rate=config.rbm_learning_rate,
        batch_size=config.rbm_batch_size,
        n_iter=config.rbm_n_iter,
        logistic_max_iter=config.logistic_max_iter,
        random_state=config.random_state,
    )
    dbn.fit(X_train_scaled, y_train.to_numpy())

    # Step 5: Evaluate.
    y_pred = dbn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    logging.info("Accuracy: %.4f", accuracy)
    logging.info("Macro F1: %.4f", macro_f1)
    logging.info("Weighted F1: %.4f", weighted_f1)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics_payload = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_report": report,
        "config": asdict(config),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_after_preprocessing": int(X_train_scaled.shape[1]),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Step 6: Diagnostic plots.
    save_confusion_matrix_plot(y_test.to_numpy(), y_pred, output_dir)
    learned_test_features = dbn.transform_features(X_test_scaled)
    save_pca_feature_plot(learned_test_features, y_test.to_numpy(), output_dir)
    layer_outputs = dbn.transform_through_layers(X_test_scaled)
    save_layer_activation_histograms(layer_outputs, output_dir)

    # Step 7: Save minimal metadata about the preprocessing choices.
    metadata = {
        "feature_columns": list(X.columns),
        "target_column": config.target_column,
        "classes": sorted(y.unique().tolist()),
        "output_files": sorted([p.name for p in output_dir.iterdir() if p.is_file()]),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("All outputs saved to: %s", output_dir.resolve())


# -----------------------------
# CLI entry point
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a DBN-style classifier on the AIS CSV dataset."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output-dir",
        default="dbn_output",
        help="Directory where metrics, logs, and plots will be written.",
    )
    parser.add_argument(
        "--target-column",
        default="VesselType",
        help="Supervised target column. Default: VesselType",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Maximum number of labeled rows to sample for training. Default: 50000",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing. Default: 0.2",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=100,
        help="Drop target classes with fewer than this many rows. Default: 100",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(256, 128),
        help="Comma-separated RBM hidden layer sizes, e.g. 256,128",
    )
    parser.add_argument(
        "--rbm-learning-rate",
        type=float,
        default=0.01,
        help="RBM learning rate. Default: 0.01",
    )
    parser.add_argument(
        "--rbm-batch-size",
        type=int,
        default=64,
        help="RBM batch size. Default: 64",
    )
    parser.add_argument(
        "--rbm-n-iter",
        type=int,
        default=10,
        help="Number of RBM epochs per layer. Default: 10",
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=1000,
        help="Max iterations for Logistic Regression. Default: 1000",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    return parser



def main() -> int:
    """Main program entry point with comprehensive error handling."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate argument ranges early so users get clear, actionable feedback.
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be a positive integer.")
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1.")
    if args.min_class_count <= 0:
        parser.error("--min-class-count must be positive.")
    if args.rbm_learning_rate <= 0:
        parser.error("--rbm-learning-rate must be positive.")
    if args.rbm_batch_size <= 0:
        parser.error("--rbm-batch-size must be positive.")
    if args.rbm_n_iter <= 0:
        parser.error("--rbm-n-iter must be positive.")
    if args.logistic_max_iter <= 0:
        parser.error("--logistic-max-iter must be positive.")

    config = DBNConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        target_column=args.target_column,
        max_rows=args.max_rows,
        random_state=args.random_state,
        test_size=args.test_size,
        min_class_count=args.min_class_count,
        rbm_hidden_layers=args.hidden_layers,
        rbm_learning_rate=args.rbm_learning_rate,
        rbm_batch_size=args.rbm_batch_size,
        rbm_n_iter=args.rbm_n_iter,
        logistic_max_iter=args.logistic_max_iter,
    )

    try:
        train_and_evaluate(config)
        return 0
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        # We print a short, user-friendly message to stderr and preserve the full
        # traceback in case the script is being run in a development environment.
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
