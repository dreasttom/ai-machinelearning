
#!/usr/bin/env python3
"""
dtw_time_series_clustering.py

Heavily commented, robust Python script for clustering time-series data using
Dynamic Time Warping (DTW) on sliding windows extracted from a CSV file.

Designed for datasets like radar/sensor readings with a timestamp column and
multiple numeric signal columns.

What the script does
--------------------
1. Reads a CSV file safely with validation and helpful error messages.
2. Parses / sorts timestamps.
3. Selects numeric features (or a user-specified subset).
4. Standardizes the selected features.
5. Builds overlapping sliding windows so each window becomes a multivariate
   time series sample.
6. Computes a pairwise DTW distance matrix across windows.
7. Runs hierarchical clustering using the DTW distances.
8. Saves graphical outputs:
      - dendrogram
      - 2D MDS projection colored by cluster
      - per-cluster average trajectory plots
9. Saves tabular outputs describing cluster assignments.

Why sliding windows?
--------------------
The attached CSV is tabular time-indexed data, not a collection of separate
time-series objects. To perform time-series clustering, we transform the single
long sequence into many shorter overlapping time-series windows. Each window is
clustered based on its temporal shape.

Dependencies
------------
Standard scientific Python stack only:
    pandas, numpy, matplotlib, scipy, scikit-learn

Example
-------
python dtw_time_series_clustering.py \
    --input /path/to/test_military_radar_readings.csv \
    --output-dir ./dtw_results \
    --window-size 12 \
    --step-size 3 \
    --clusters 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so the script works on servers/headless systems.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------

@dataclass
class WindowMetadata:
    """Stores metadata about a generated sliding window."""
    window_id: int
    start_index: int
    end_index: int
    start_timestamp: str
    end_timestamp: str


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    """
    Configure logging once, with optional verbose mode.

    Parameters
    ----------
    verbose : bool
        If True, log at DEBUG level. Otherwise INFO level.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with practical defaults."""
    parser = argparse.ArgumentParser(
        description="Cluster multivariate time-series windows using DTW distance."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default="dtw_clustering_output",
        help="Directory where plots and result files will be written."
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Name of the timestamp column. Default: timestamp"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Optional list of numeric feature columns to use. "
            "If omitted, all numeric columns except obviously non-measurement columns are used."
        )
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Number of rows per sliding window. Default: 12"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=3,
        help="Number of rows to move between windows. Default: 3"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of clusters to extract from the hierarchical tree. Default: 4"
    )
    parser.add_argument(
        "--dtw-window",
        type=int,
        default=None,
        help=(
            "Optional Sakoe-Chiba warping window (in time steps). "
            "Use this to speed up DTW and limit unrealistic warping."
        )
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=120,
        help=(
            "Safety cap on number of windows used for clustering. "
            "DTW is O(N^2) over samples and O(T^2) within a pair. Default: 120"
        )
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging."
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------

def load_csv(input_path: Path) -> pd.DataFrame:
    """
    Read the CSV into a pandas DataFrame with robust error messages.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The input CSV is empty: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV file: {input_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while reading CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV loaded successfully but contains no rows.")

    logging.info("Loaded CSV with %d rows and %d columns.", df.shape[0], df.shape[1])
    return df


def parse_and_sort_timestamps(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Parse timestamps and sort chronologically.

    We do not fail if parsing cannot interpret some rows; instead we coerce invalid
    rows to NaT, count them, and then drop them with a clear log message.
    """
    if timestamp_column not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")

    invalid_ts = result[timestamp_column].isna().sum()
    if invalid_ts > 0:
        logging.warning(
            "Found %d rows with invalid timestamps. These rows will be dropped.",
            invalid_ts,
        )
        result = result.dropna(subset=[timestamp_column])

    if result.empty:
        raise ValueError("No usable rows remain after timestamp parsing.")

    result = result.sort_values(timestamp_column).reset_index(drop=True)
    logging.info("Timestamp parsing/sorting complete.")
    return result


def choose_feature_columns(df: pd.DataFrame, timestamp_column: str, requested: Sequence[str] | None) -> List[str]:
    """
    Determine which numeric columns to use for clustering.

    If the user explicitly requested features, validate them.
    Otherwise, auto-select numeric columns while excluding common label-like fields.
    """
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise KeyError(f"Requested feature columns not found: {missing}")

        non_numeric = [col for col in requested if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise TypeError(
                f"These requested feature columns are not numeric and cannot be used directly: {non_numeric}"
            )
        features = list(requested)
    else:
        exclude_names = {
            timestamp_column,
            "target_type",   # Categorical label in the attached dataset
            "label",
            "class",
            "category",
            "id",
            "identifier",
        }
        numeric_candidates = [
            col for col in df.columns
            if col not in exclude_names and pd.api.types.is_numeric_dtype(df[col])
        ]
        features = numeric_candidates

    if not features:
        raise ValueError(
            "No numeric feature columns were selected. "
            "Provide numeric columns via --features."
        )

    logging.info("Using feature columns: %s", features)
    return features


# -----------------------------------------------------------------------------
# Sliding window construction
# -----------------------------------------------------------------------------

def build_sliding_windows(
    df: pd.DataFrame,
    timestamp_column: str,
    feature_columns: Sequence[str],
    window_size: int,
    step_size: int,
    max_windows: int,
) -> Tuple[np.ndarray, List[WindowMetadata]]:
    """
    Convert one long multivariate time series into many smaller time-series windows.

    Output array shape:
        (n_windows, window_size, n_features)

    Important:
    ----------
    Sliding-window construction is how we transform a single long sensor log into
    multiple comparable time-series objects suitable for clustering.
    """
    if window_size < 2:
        raise ValueError("--window-size must be at least 2.")
    if step_size < 1:
        raise ValueError("--step-size must be at least 1.")
    if max_windows < 2:
        raise ValueError("--max-windows must be at least 2.")

    num_rows = len(df)
    if num_rows < window_size:
        raise ValueError(
            f"Not enough rows ({num_rows}) for window size {window_size}. "
            f"Reduce --window-size."
        )

    values = df.loc[:, feature_columns].to_numpy(dtype=float)
    timestamps = df.loc[:, timestamp_column].tolist()

    windows = []
    metadata = []
    window_id = 0

    for start in range(0, num_rows - window_size + 1, step_size):
        end = start + window_size
        windows.append(values[start:end, :])

        metadata.append(
            WindowMetadata(
                window_id=window_id,
                start_index=start,
                end_index=end - 1,
                start_timestamp=str(timestamps[start]),
                end_timestamp=str(timestamps[end - 1]),
            )
        )
        window_id += 1

        if len(windows) >= max_windows:
            logging.warning(
                "Reached max window cap (%d). Remaining windows will be ignored "
                "to keep DTW computation tractable.",
                max_windows,
            )
            break

    if len(windows) < 2:
        raise ValueError(
            "Window generation produced fewer than 2 windows; clustering is not meaningful."
        )

    X = np.stack(windows, axis=0)
    logging.info("Built %d windows with shape %s.", len(windows), X.shape)
    return X, metadata


# -----------------------------------------------------------------------------
# DTW implementation
# -----------------------------------------------------------------------------

def euclidean_step_cost(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute local cost between two feature vectors in a multivariate series.
    """
    return float(np.linalg.norm(vec_a - vec_b))


def dtw_distance(
    series_a: np.ndarray,
    series_b: np.ndarray,
    warping_window: int | None = None,
) -> float:
    """
    Compute DTW distance between two multivariate time series.

    Parameters
    ----------
    series_a, series_b : np.ndarray
        Arrays of shape (time_steps, n_features)
    warping_window : int or None
        Optional Sakoe-Chiba constraint limiting how far off-diagonal the warp path
        can move. This often improves realism and performance.

    Notes
    -----
    DTW allows two sequences with similar shapes but small temporal shifts /
    stretches to be considered close.
    """
    if series_a.ndim != 2 or series_b.ndim != 2:
        raise ValueError("DTW expects 2D arrays of shape (time_steps, n_features).")

    len_a, len_b = len(series_a), len(series_b)

    if warping_window is None:
        # No constraint: allow the full matrix.
        warping_window = max(len_a, len_b)
    else:
        warping_window = max(warping_window, abs(len_a - len_b))

    # We use a classic dynamic-programming DTW matrix initialized with infinity.
    dtw = np.full((len_a + 1, len_b + 1), np.inf, dtype=float)
    dtw[0, 0] = 0.0

    for i in range(1, len_a + 1):
        # Restrict j range if a warping window is used.
        j_start = max(1, i - warping_window)
        j_end = min(len_b, i + warping_window)

        for j in range(j_start, j_end + 1):
            cost = euclidean_step_cost(series_a[i - 1], series_b[j - 1])

            # Classic DTW recurrence:
            # cost + min(insertion, deletion, match)
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1],  # match
            )

    return float(dtw[len_a, len_b])


def compute_pairwise_dtw_matrix(
    X: np.ndarray,
    warping_window: int | None = None,
) -> np.ndarray:
    """
    Compute a full symmetric pairwise DTW distance matrix.

    Output shape:
        (n_samples, n_samples)
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, time_steps, n_features).")

    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=float)

    total_pairs = n_samples * (n_samples - 1) // 2
    processed = 0

    logging.info("Computing %d pairwise DTW distances. This may take a moment...", total_pairs)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = dtw_distance(X[i], X[j], warping_window)
            distances[j, i] = distances[i, j]

            processed += 1
            if processed % 100 == 0 or processed == total_pairs:
                logging.info("Processed %d / %d DTW pairs.", processed, total_pairs)

    return distances


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def cluster_from_distance_matrix(distance_matrix: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering from a precomputed distance matrix.

    Returns
    -------
    labels : np.ndarray
        Cluster labels starting at 1 (scipy convention for fcluster).
    linkage_matrix : np.ndarray
        Hierarchical linkage matrix suitable for dendrogram plots.
    """
    if n_clusters < 2:
        raise ValueError("--clusters must be at least 2.")

    if distance_matrix.shape[0] < n_clusters:
        raise ValueError(
            f"Cannot request {n_clusters} clusters with only {distance_matrix.shape[0]} windows."
        )

    # scipy's linkage expects a condensed-form distance vector.
    condensed = squareform(distance_matrix, checks=False)

    # Average linkage is a stable / interpretable default for DTW-based dissimilarities.
    linkage_matrix = linkage(condensed, method="average")

    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    return labels, linkage_matrix


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def save_cluster_assignments(
    output_dir: Path,
    labels: np.ndarray,
    metadata: Sequence[WindowMetadata],
) -> None:
    """
    Save cluster assignments and window metadata to CSV.
    """
    rows = []
    for meta, label in zip(metadata, labels):
        rows.append({
            "window_id": meta.window_id,
            "cluster": int(label),
            "start_index": meta.start_index,
            "end_index": meta.end_index,
            "start_timestamp": meta.start_timestamp,
            "end_timestamp": meta.end_timestamp,
        })

    out_csv = output_dir / "cluster_assignments.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info("Wrote cluster assignments to %s", out_csv)


def save_run_summary(
    output_dir: Path,
    input_path: Path,
    feature_columns: Sequence[str],
    X: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """
    Save a compact JSON summary of the run for reproducibility.
    """
    unique, counts = np.unique(labels, return_counts=True)
    summary = {
        "input_file": str(input_path),
        "n_windows": int(X.shape[0]),
        "window_size": int(X.shape[1]),
        "n_features": int(X.shape[2]),
        "feature_columns": list(feature_columns),
        "clusters_requested": int(args.clusters),
        "cluster_sizes": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "step_size": int(args.step_size),
        "dtw_window": None if args.dtw_window is None else int(args.dtw_window),
    }

    out_json = output_dir / "run_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Wrote run summary to %s", out_json)


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def plot_dendrogram(linkage_matrix: np.ndarray, output_dir: Path) -> None:
    """
    Save a hierarchical dendrogram plot.
    """
    plt.figure(figsize=(14, 6))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None)
    plt.title("Hierarchical Clustering Dendrogram (DTW Distance)")
    plt.xlabel("Window index")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    out_path = output_dir / "dendrogram.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    logging.info("Saved dendrogram plot to %s", out_path)


def plot_mds_projection(distance_matrix: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """
    Project the precomputed distance matrix into 2D using MDS so the user can
    visually inspect cluster separation.
    """
    # MDS may emit warnings on some datasets; we allow them unless the fit fails.
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    points_2d = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, s=60)
    plt.title("2D MDS Projection of DTW Distance Matrix")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.tight_layout()
    out_path = output_dir / "mds_projection.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    logging.info("Saved MDS projection to %s", out_path)


def plot_cluster_average_trajectories(
    X: np.ndarray,
    labels: np.ndarray,
    feature_columns: Sequence[str],
    output_dir: Path,
) -> None:
    """
    For each feature, plot the average window trajectory per cluster.

    Note:
    -----
    Because windows were standardized before clustering, the plots show
    standardized feature behavior (z-score like values), which makes
    cross-feature temporal shape easier to compare.
    """
    unique_labels = sorted(np.unique(labels).tolist())

    for feature_idx, feature_name in enumerate(feature_columns):
        plt.figure(figsize=(12, 6))

        for cluster_id in unique_labels:
            cluster_windows = X[labels == cluster_id, :, feature_idx]
            if cluster_windows.size == 0:
                continue

            mean_trajectory = cluster_windows.mean(axis=0)
            std_trajectory = cluster_windows.std(axis=0)
            time_axis = np.arange(X.shape[1])

            plt.plot(time_axis, mean_trajectory, label=f"Cluster {cluster_id}")
            plt.fill_between(
                time_axis,
                mean_trajectory - std_trajectory,
                mean_trajectory + std_trajectory,
                alpha=0.2,
            )

        plt.title(f"Average Window Trajectories by Cluster: {feature_name}")
        plt.xlabel("Time step within window")
        plt.ylabel("Standardized value")
        plt.legend()
        plt.tight_layout()

        out_path = output_dir / f"cluster_trajectory_{feature_name}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        logging.info("Saved cluster trajectory plot for '%s' to %s", feature_name, out_path)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> int:
    """
    Main program entry point.

    Returns
    -------
    int
        Process exit code. 0 for success, non-zero for failure.
    """
    args = parse_args()
    configure_logging(args.verbose)

    try:
        input_path = Path(args.input).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Starting DTW time-series clustering workflow.")
        logging.info("Input file: %s", input_path)
        logging.info("Output directory: %s", output_dir)

        # ---------------------------------------------------------------------
        # Load and validate the raw data
        # ---------------------------------------------------------------------
        df = load_csv(input_path)
        df = parse_and_sort_timestamps(df, args.timestamp_column)

        feature_columns = choose_feature_columns(
            df=df,
            timestamp_column=args.timestamp_column,
            requested=args.features,
        )

        # ---------------------------------------------------------------------
        # Clean missing values in the selected features
        # ---------------------------------------------------------------------
        feature_df = df.loc[:, feature_columns].copy()
        missing_counts = feature_df.isna().sum()

        if missing_counts.sum() > 0:
            logging.warning(
                "Missing numeric values detected in feature columns. "
                "Applying forward-fill, backward-fill, then median fallback."
            )
            feature_df = feature_df.ffill().bfill()
            for col in feature_columns:
                if feature_df[col].isna().any():
                    median_value = feature_df[col].median()
                    feature_df[col] = feature_df[col].fillna(median_value)

        if feature_df.isna().any().any():
            raise ValueError(
                "Feature cleaning completed but NaN values remain. "
                "Please inspect the input data."
            )

        # Replace the original numeric columns with cleaned values.
        df.loc[:, feature_columns] = feature_df

        # ---------------------------------------------------------------------
        # Standardize features
        # ---------------------------------------------------------------------
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df.loc[:, feature_columns].to_numpy(dtype=float))
        df_scaled = df.copy()
        df_scaled.loc[:, feature_columns] = scaled_values
        logging.info("Feature scaling complete.")

        # ---------------------------------------------------------------------
        # Build time-series windows
        # ---------------------------------------------------------------------
        X, metadata = build_sliding_windows(
            df=df_scaled,
            timestamp_column=args.timestamp_column,
            feature_columns=feature_columns,
            window_size=args.window_size,
            step_size=args.step_size,
            max_windows=args.max_windows,
        )

        # ---------------------------------------------------------------------
        # Pairwise DTW distance computation
        # ---------------------------------------------------------------------
        distance_matrix = compute_pairwise_dtw_matrix(
            X=X,
            warping_window=args.dtw_window,
        )

        # ---------------------------------------------------------------------
        # Hierarchical clustering
        # ---------------------------------------------------------------------
        labels, linkage_matrix = cluster_from_distance_matrix(
            distance_matrix=distance_matrix,
            n_clusters=args.clusters,
        )
        logging.info("Clustering complete.")

        # ---------------------------------------------------------------------
        # Save machine-readable outputs
        # ---------------------------------------------------------------------
        save_cluster_assignments(output_dir, labels, metadata)
        save_run_summary(output_dir, input_path, feature_columns, X, labels, args)

        np.save(output_dir / "distance_matrix.npy", distance_matrix)
        logging.info("Saved raw DTW distance matrix to %s", output_dir / "distance_matrix.npy")

        # ---------------------------------------------------------------------
        # Save plots
        # ---------------------------------------------------------------------
        plot_dendrogram(linkage_matrix, output_dir)
        plot_mds_projection(distance_matrix, labels, output_dir)
        plot_cluster_average_trajectories(X, labels, feature_columns, output_dir)

        logging.info("Workflow finished successfully.")
        return 0

    except KeyboardInterrupt:
        logging.error("Execution interrupted by user.")
        return 130
    except Exception as exc:
        # Log the full traceback for developer / analyst troubleshooting.
        logging.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
