#!/usr/bin/env python3
"""
DBSCAN clustering script for the uploaded missile dataset.

This script:
1. Loads a CSV file.
2. Cleans and validates the data.
3. Builds a mixed feature representation from numeric and text columns.
4. Runs DBSCAN clustering.
5. Saves graphical output and a text summary.

uses https://www.kaggle.com/datasets/fanbyprinciple/north-korea-missile-test-database


"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Custom exception definitions
# -----------------------------
# Using custom exception classes makes it easier to separate failures caused by
# invalid input data from unexpected runtime failures.

class DatasetError(Exception):
    """Raised when the input dataset is invalid or unusable."""


class FeatureEngineeringError(Exception):
    """Raised when feature construction fails."""


class ClusteringError(Exception):
    """Raised when DBSCAN clustering fails."""


@dataclass
class DBSCANResult:
    """Container for final clustering artifacts and metadata."""

    dataframe: pd.DataFrame
    feature_matrix: sparse.spmatrix | np.ndarray
    reduced_2d: np.ndarray
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    eps: float
    min_samples: int
    silhouette: Optional[float]


# -----------------------------
# Logging configuration helpers
# -----------------------------

def configure_logging(verbose: bool = False) -> None:
    """Configure logging for clear console diagnostics."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------
# Utility functions
# -----------------------------

def ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if it does not already exist."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise DatasetError(f"Unable to create output directory '{output_dir}': {exc}") from exc


# The source CSV stores numeric values in messy human-readable forms like
# '1,200 kg', '6.4 m', or possibly mixed text. This helper extracts the first
# numeric value it can find and converts it to float.
def safe_extract_numeric(value: object) -> float:
    """Safely extract a numeric value from messy text.

    Returns NaN when conversion is not possible instead of crashing.
    """
    if pd.isna(value):
        return np.nan

    # Already numeric? Return it directly.
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    # Remove commas to handle thousands separators such as "1,200".
    text = text.replace(",", "")

    # Find the first number, including decimals and optional scientific notation.
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan

    try:
        return float(match.group(0))
    except (ValueError, OverflowError):
        return np.nan


# Text fields are combined into one searchable / vectorizable text block so
# TF-IDF can represent descriptive information such as missile type, engine,
# guidance, launch platform, and origin.
def combine_text_columns(df: pd.DataFrame, text_columns: Sequence[str]) -> pd.Series:
    """Combine a collection of text columns into a single string per row."""
    try:
        available_columns = [c for c in text_columns if c in df.columns]
        if not available_columns:
            raise FeatureEngineeringError(
                "No requested text columns were found in the dataset."
            )

        # Fill missing values with empty strings to avoid 'nan' text pollution.
        cleaned = df[available_columns].fillna("").astype(str)

        # Joining with spaces gives TF-IDF a unified text representation.
        combined = cleaned.apply(lambda row: " ".join(item.strip() for item in row if item.strip()), axis=1)

        # If a row has no text at all, provide a placeholder token so the row
        # is not lost during vectorization.
        return combined.replace("", "missing_text")
    except Exception as exc:
        raise FeatureEngineeringError(f"Failed to combine text columns: {exc}") from exc


# -----------------------------
# Data loading and validation
# -----------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the dataset from CSV with validation and good diagnostics."""
    if not os.path.exists(csv_path):
        raise DatasetError(f"CSV file does not exist: {csv_path}")

    if not os.path.isfile(csv_path):
        raise DatasetError(f"Provided path is not a file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(f"The CSV file could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"The CSV file encoding could not be decoded: {exc}") from exc
    except Exception as exc:
        raise DatasetError(f"Unexpected error reading CSV: {exc}") from exc

    if df.empty:
        raise DatasetError("The CSV loaded successfully but contains no rows.")

    return df


# -----------------------------
# Feature engineering
# -----------------------------

def prepare_features(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    text_columns: Sequence[str],
    max_text_features: int,
) -> Tuple[pd.DataFrame, sparse.spmatrix]:
    """Build a mixed numeric + TF-IDF feature matrix.

    Returns a cleaned dataframe plus the final sparse feature matrix.
    """
    try:
        working_df = df.copy()

        # Convert all requested numeric columns into actual numeric values.
        numeric_present = [c for c in numeric_columns if c in working_df.columns]
        for column in numeric_present:
            working_df[column] = working_df[column].apply(safe_extract_numeric)

        # Prepare numeric matrix with imputation and scaling.
        if numeric_present:
            numeric_frame = working_df[numeric_present]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    # with_mean=False is mandatory for sparse compatibility if
                    # later combined with sparse TF-IDF features.
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            numeric_matrix = num_pipeline.fit_transform(numeric_frame)
            if not sparse.issparse(numeric_matrix):
                numeric_matrix = sparse.csr_matrix(numeric_matrix)
        else:
            numeric_matrix = None

        # Prepare text matrix using TF-IDF. We allow unigrams and bigrams to
        # capture both single terms and common phrases such as 'solid fuel'.
        text_series = combine_text_columns(working_df, text_columns)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_text_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        text_matrix = vectorizer.fit_transform(text_series)

        # Combine numeric and text features into one matrix.
        if numeric_matrix is not None:
            final_matrix = sparse.hstack([numeric_matrix, text_matrix], format="csr")
        else:
            final_matrix = text_matrix.tocsr()

        if final_matrix.shape[0] < 5:
            raise FeatureEngineeringError(
                "Not enough usable rows after preprocessing; at least 5 are recommended."
            )
        if final_matrix.shape[1] < 2:
            raise FeatureEngineeringError(
                "Feature matrix is too small to cluster meaningfully."
            )

        return working_df, final_matrix
    except FeatureEngineeringError:
        raise
    except Exception as exc:
        raise FeatureEngineeringError(f"Feature engineering failed: {exc}") from exc


# -----------------------------
# Parameter suggestion helpers
# -----------------------------

def suggest_eps(feature_matrix: sparse.spmatrix | np.ndarray, min_samples: int) -> float:
    """Estimate a reasonable epsilon value from the k-distance distribution.

    This is a heuristic, not a guaranteed best parameter.
    """
    if min_samples < 2:
        raise ClusteringError("min_samples must be at least 2 to estimate epsilon.")

    try:
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
        neighbors.fit(feature_matrix)
        distances, _ = neighbors.kneighbors(feature_matrix)

        # The k-th nearest-neighbor distance is commonly used to pick eps.
        kth_distances = np.sort(distances[:, -1])

        # Use a high percentile as a conservative automatic estimate.
        estimated_eps = float(np.percentile(kth_distances, 90))

        if not math.isfinite(estimated_eps) or estimated_eps <= 0:
            raise ClusteringError("Estimated epsilon was non-finite or non-positive.")

        return estimated_eps
    except Exception as exc:
        raise ClusteringError(f"Failed to estimate epsilon: {exc}") from exc


# -----------------------------
# Dimensionality reduction
# -----------------------------

def reduce_to_two_dimensions(feature_matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """Reduce the feature matrix to 2 dimensions for plotting.

    SVD is preferred for sparse matrices; PCA is used for dense matrices.
    """
    try:
        if sparse.issparse(feature_matrix):
            reducer = TruncatedSVD(n_components=2, random_state=42)
            reduced = reducer.fit_transform(feature_matrix)
        else:
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(feature_matrix)

        if reduced.shape[1] != 2:
            raise FeatureEngineeringError("2D reduction did not produce exactly two components.")

        return reduced
    except Exception as exc:
        raise FeatureEngineeringError(f"Failed to reduce data to 2D: {exc}") from exc


# -----------------------------
# Clustering execution
# -----------------------------

def run_dbscan(
    df: pd.DataFrame,
    feature_matrix: sparse.spmatrix | np.ndarray,
    eps: float,
    min_samples: int,
) -> DBSCANResult:
    """Run DBSCAN and compute summary metrics."""
    try:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = model.fit_predict(feature_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        reduced_2d = reduce_to_two_dimensions(feature_matrix)

        # Silhouette requires at least 2 clusters and at least one non-noise sample
        # in multiple clusters. We compute it only when valid.
        valid_silhouette = None
        unique_non_noise = sorted(label for label in set(labels) if label != -1)
        if len(unique_non_noise) >= 2:
            mask = labels != -1
            if np.sum(mask) > len(unique_non_noise):
                try:
                    feature_for_score = feature_matrix[mask]
                    valid_silhouette = float(silhouette_score(feature_for_score, labels[mask]))
                except Exception:
                    valid_silhouette = None

        labeled_df = df.copy()
        labeled_df["DBSCAN_CLUSTER"] = labels

        return DBSCANResult(
            dataframe=labeled_df,
            feature_matrix=feature_matrix,
            reduced_2d=reduced_2d,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            eps=eps,
            min_samples=min_samples,
            silhouette=valid_silhouette,
        )
    except Exception as exc:
        raise ClusteringError(f"DBSCAN clustering failed: {exc}") from exc


# -----------------------------
# Plotting functions
# -----------------------------

def save_k_distance_plot(feature_matrix: sparse.spmatrix | np.ndarray, min_samples: int, output_path: str) -> None:
    """Save a k-distance plot to help interpret epsilon selection."""
    try:
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean")
        neighbors.fit(feature_matrix)
        distances, _ = neighbors.kneighbors(feature_matrix)
        kth_distances = np.sort(distances[:, -1])

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(kth_distances) + 1), kth_distances)
        plt.title(f"k-Distance Plot (k = {min_samples})")
        plt.xlabel("Sorted Data Point Index")
        plt.ylabel(f"Distance to {min_samples}-th Nearest Neighbor")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as exc:
        raise ClusteringError(f"Failed to save k-distance plot: {exc}") from exc


# The scatter plot uses the 2D projection purely for visualization. It does NOT
# mean DBSCAN operated on only two features; the clustering was done in the full
# engineered feature space.
def save_cluster_scatter_plot(result: DBSCANResult, output_path: str) -> None:
    """Save a 2D scatter plot of clustering results."""
    try:
        plt.figure(figsize=(10, 7))
        labels = result.labels
        reduced = result.reduced_2d

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = labels == label
            display_name = "Noise" if label == -1 else f"Cluster {label}"
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                s=35,
                alpha=0.75,
                label=display_name,
            )

        plt.title("DBSCAN Clusters (2D Projection)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as exc:
        raise ClusteringError(f"Failed to save cluster scatter plot: {exc}") from exc


def save_cluster_distribution_plot(labels: np.ndarray, output_path: str) -> None:
    """Save a bar chart showing cluster membership counts."""
    try:
        label_series = pd.Series(labels)
        counts = label_series.value_counts().sort_index()

        display_labels = ["Noise" if idx == -1 else f"C{idx}" for idx in counts.index]

        plt.figure(figsize=(10, 6))
        plt.bar(display_labels, counts.values)
        plt.title("Cluster Membership Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as exc:
        raise ClusteringError(f"Failed to save cluster distribution plot: {exc}") from exc


# -----------------------------
# Reporting
# -----------------------------

def write_summary_report(result: DBSCANResult, output_path: str) -> None:
    """Write a text report describing clustering results."""
    try:
        labels = result.labels
        label_counts = pd.Series(labels).value_counts().sort_index()

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("DBSCAN CLUSTERING REPORT\n")
            handle.write("=" * 80 + "\n")
            handle.write(f"Rows processed: {len(result.dataframe)}\n")
            handle.write(f"Features used: {result.feature_matrix.shape[1]}\n")
            handle.write(f"Epsilon (eps): {result.eps:.6f}\n")
            handle.write(f"min_samples: {result.min_samples}\n")
            handle.write(f"Clusters found (excluding noise): {result.n_clusters}\n")
            handle.write(f"Noise points: {result.n_noise}\n")
            handle.write(
                f"Silhouette score (non-noise only): {result.silhouette if result.silhouette is not None else 'Not available'}\n"
            )
            handle.write("\nCluster counts:\n")
            for label, count in label_counts.items():
                name = "Noise" if label == -1 else f"Cluster {label}"
                handle.write(f"  - {name}: {int(count)}\n")

            # Include a small preview of each cluster for quick inspection.
            handle.write("\nExample records by cluster:\n")
            for label in sorted(set(labels)):
                name = "Noise" if label == -1 else f"Cluster {label}"
                subset = result.dataframe[result.dataframe["DBSCAN_CLUSTER"] == label].head(5)
                handle.write(f"\n{name}\n")
                handle.write("-" * 40 + "\n")
                columns_to_show = [c for c in ["NAME", "TYPE", "ORIGIN", "ENGINE", "LAUNCH"] if c in subset.columns]
                if subset.empty:
                    handle.write("No rows.\n")
                elif columns_to_show:
                    handle.write(subset[columns_to_show].to_string(index=False))
                    handle.write("\n")
                else:
                    handle.write(subset.head(5).to_string(index=False))
                    handle.write("\n")
    except OSError as exc:
        raise DatasetError(f"Unable to write summary report: {exc}") from exc
    except Exception as exc:
        raise DatasetError(f"Unexpected error while writing report: {exc}") from exc


# -----------------------------
# Main program workflow
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DBSCAN clustering on a missile CSV dataset and save graphical output."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-dir", default="dbscan_output", help="Directory for reports and plots.")
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="DBSCAN epsilon value. If omitted, the script estimates one automatically.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter (default: 5).",
    )
    parser.add_argument(
        "--max-text-features",
        type=int,
        default=500,
        help="Maximum TF-IDF text features to keep (default: 500).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args(argv)


# A dedicated main function keeps the top-level control flow readable and makes
# unit testing easier.
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.min_samples < 2:
            raise DatasetError("--min-samples must be at least 2.")
        if args.max_text_features < 10:
            raise DatasetError("--max-text-features must be at least 10.")
        if args.eps is not None and args.eps <= 0:
            raise DatasetError("--eps must be greater than 0 when provided.")

        ensure_output_dir(args.output_dir)
        logging.info("Loading dataset from %s", args.csv)
        df = load_dataset(args.csv)
        logging.info("Loaded %d rows and %d columns.", len(df), len(df.columns))

        numeric_columns = ["MASS", "LENGTH", "WEIGHT", "DIAMETER", "DESIGNED", "ID"]
        text_columns = [
            "TYPE",
            "NAME",
            "ENGINE",
            "LAUNCH",
            "ORIGIN",
            "WARHEAD",
            "GUIDANCE",
            "PROPELLANT",
        ]

        logging.info("Preparing mixed numeric/text feature matrix...")
        cleaned_df, feature_matrix = prepare_features(
            df=df,
            numeric_columns=numeric_columns,
            text_columns=text_columns,
            max_text_features=args.max_text_features,
        )
        logging.info(
            "Feature matrix created with shape %s.",
            tuple(feature_matrix.shape),
        )

        eps = args.eps
        if eps is None:
            logging.info("Estimating epsilon automatically...")
            eps = suggest_eps(feature_matrix, args.min_samples)
            logging.info("Estimated eps = %.6f", eps)
        else:
            logging.info("Using user-provided eps = %.6f", eps)

        logging.info("Running DBSCAN clustering...")
        result = run_dbscan(cleaned_df, feature_matrix, eps, args.min_samples)
        logging.info(
            "DBSCAN complete. Clusters=%d, Noise=%d",
            result.n_clusters,
            result.n_noise,
        )

        # Save labeled data for later inspection.
        labeled_csv_path = os.path.join(args.output_dir, "clustered_data.csv")
        result.dataframe.to_csv(labeled_csv_path, index=False)

        # Save plots.
        save_k_distance_plot(
            feature_matrix,
            args.min_samples,
            os.path.join(args.output_dir, "k_distance_plot.png"),
        )
        save_cluster_scatter_plot(
            result,
            os.path.join(args.output_dir, "dbscan_cluster_scatter.png"),
        )
        save_cluster_distribution_plot(
            result.labels,
            os.path.join(args.output_dir, "cluster_distribution.png"),
        )

        # Save text report.
        write_summary_report(
            result,
            os.path.join(args.output_dir, "dbscan_report.txt"),
        )

        logging.info("All outputs were written to: %s", args.output_dir)
        logging.info("Done.")
        return 0

    except (DatasetError, FeatureEngineeringError, ClusteringError) as exc:
        logging.error("Known error: %s", exc)
        if args.verbose:
            logging.debug("Detailed traceback:\n%s", traceback.format_exc())
        return 1
    except KeyboardInterrupt:
        logging.error("Execution interrupted by user.")
        return 130
    except Exception as exc:
        # Catch-all handler for anything unexpected. This is intentionally broad
        # because the user requested robust exception handling.
        logging.critical("Unexpected fatal error: %s", exc)
        logging.critical("Traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
