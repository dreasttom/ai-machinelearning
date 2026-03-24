#!/usr/bin/env python3
"""
optics_radar_clustering.py

A standalone script that implements the OPTICS clustering algorithm from scratch
and applies it to a CSV file located in the same folder as this script.

Designed for robust, practical use with datasets like radar readings.

Key features:
- Reads CSV data from the same directory as the script
- Automatically selects numeric columns for clustering
- Handles missing values, invalid parameters, and common file issues
- Normalizes numeric features before clustering
- Computes core distances and reachability distances
- Produces an OPTICS ordering
- Extracts simple clusters using a DBSCAN-style epsilon cut over OPTICS results
- Writes output with cluster assignments to a new CSV file
- Includes extensive comments for learning and maintenance

IMPORTANT:
This script implements OPTICS directly and does not depend on scikit-learn's OPTICS.

Usage examples:
    python optics_radar_clustering.py
    python optics_radar_clustering.py --input test_military_radar_readings.csv
    python optics_radar_clustering.py --input test_military_radar_readings.csv --min-samples 8 --max-eps 3.0 --cluster-eps 1.2

Default assumptions:
- The input file is named "test_military_radar_readings.csv"
- The input file is in the same folder as this script
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Custom exception types
# =============================================================================

class OpticsError(Exception):
    """Base exception for OPTICS-related failures."""
    pass


class DataValidationError(OpticsError):
    """Raised when the input data cannot be used for clustering."""
    pass


class ParameterValidationError(OpticsError):
    """Raised when user parameters are invalid."""
    pass


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OpticsResult:
    """
    Stores the main outputs of the OPTICS algorithm.

    Attributes:
        ordering:
            The order in which points were processed by OPTICS.
        reachability:
            Reachability distance for each point.
            Unreachable points remain np.inf.
        core_distance:
            Core distance for each point.
            Undefined core distances remain np.inf.
        predecessor:
            Index of the predecessor point that last updated the reachability,
            or -1 if none exists.
    """
    ordering: List[int]
    reachability: np.ndarray
    core_distance: np.ndarray
    predecessor: np.ndarray


# =============================================================================
# Utility functions
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with helpful descriptions."""
    parser = argparse.ArgumentParser(
        description="Run a from-scratch OPTICS clustering implementation on a CSV file."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="test_military_radar_readings.csv",
        help="Name of the CSV file in the same folder as this script."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="optics_clustered_output.csv",
        help="Output CSV filename to write results to, in the same folder."
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum number of samples required for a core point."
    )

    parser.add_argument(
        "--max-eps",
        type=float,
        default=np.inf,
        help="Maximum neighborhood radius considered by OPTICS. Use a positive number, or omit for infinity."
    )

    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=1.5,
        help=(
            "Epsilon used AFTER OPTICS ordering to extract simple DBSCAN-style clusters "
            "from reachability/core distances."
        )
    )

    parser.add_argument(
        "--drop-na",
        action="store_true",
        help=(
            "Drop rows with missing numeric values. "
            "If not provided, missing numeric values are imputed with the median."
        )
    )

    return parser.parse_args()


def validate_parameters(min_samples: int, max_eps: float, cluster_eps: float) -> None:
    """Validate user-provided clustering parameters."""
    if min_samples < 2:
        raise ParameterValidationError("--min-samples must be at least 2.")

    if not (math.isinf(max_eps) or max_eps > 0):
        raise ParameterValidationError("--max-eps must be positive or omitted.")

    if cluster_eps <= 0:
        raise ParameterValidationError("--cluster-eps must be > 0.")


def resolve_file_path(filename: str) -> Path:
    """
    Resolve a file path relative to the script location.

    This makes the script robust even if the user runs it from another working
    directory. The script always looks in its own folder first.
    """
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / filename
    return file_path


def load_csv_file(file_path: Path) -> pd.DataFrame:
    """
    Load the CSV file with robust error handling.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\n"
            f"Make sure the CSV is in the same folder as this script."
        )

    if not file_path.is_file():
        raise FileNotFoundError(f"Path exists but is not a file: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"The file is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            f"Could not parse the CSV file: {file_path}\n"
            f"Please verify that it is a valid comma-separated file."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            f"The file could not be decoded as text: {file_path}"
        ) from exc
    except Exception as exc:
        raise OpticsError(f"Unexpected error while reading CSV: {exc}") from exc

    if df.empty:
        raise DataValidationError("The CSV file contains no rows.")

    return df


def prepare_numeric_features(df: pd.DataFrame, drop_na: bool) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Select and clean numeric columns for clustering.

    Returns:
        original_aligned_df:
            Original dataframe aligned to the rows used in clustering.
        numeric_df:
            Numeric-only dataframe used for clustering.
        numeric_columns:
            Names of the selected numeric columns.

    Notes:
    - Non-numeric columns are ignored automatically.
    - Missing numeric values are either dropped or median-imputed.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        raise DataValidationError(
            "No numeric columns were found. OPTICS requires numeric features."
        )

    numeric_columns = numeric_df.columns.tolist()

    # Replace positive/negative infinity with NaN so missing-data handling
    # can process them consistently.
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_na:
        valid_mask = ~numeric_df.isna().any(axis=1)
        numeric_df = numeric_df.loc[valid_mask].copy()
        original_aligned_df = df.loc[valid_mask].copy()

        if numeric_df.empty:
            raise DataValidationError(
                "All rows were removed after dropping missing numeric values."
            )
    else:
        # Median imputation is often a reasonable, stable default for numeric data.
        for col in numeric_columns:
            if numeric_df[col].isna().all():
                raise DataValidationError(
                    f"Numeric column '{col}' contains only missing values."
                )
            median_value = numeric_df[col].median()
            numeric_df[col] = numeric_df[col].fillna(median_value)

        original_aligned_df = df.copy()

    if len(numeric_df) < 2:
        raise DataValidationError("At least 2 usable rows are required for clustering.")

    return original_aligned_df, numeric_df, numeric_columns


def standardize_features(numeric_df: pd.DataFrame) -> np.ndarray:
    """
    Standardize features to zero mean and unit variance.

    Why this matters:
    OPTICS relies on distances. If one feature has a much larger scale than
    another, it can dominate the distance computation and distort clusters.

    Zero-variance columns are handled safely by replacing std=0 with std=1.
    """
    X = numeric_df.to_numpy(dtype=float)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Avoid division by zero for constant-value columns.
    stds_safe = np.where(stds == 0, 1.0, stds)

    X_scaled = (X - means) / stds_safe

    if not np.isfinite(X_scaled).all():
        raise DataValidationError(
            "Scaled feature matrix contains invalid values after preprocessing."
        )

    return X_scaled


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute a full pairwise Euclidean distance matrix.

    This implementation is simple and clear, which is useful for educational
    purposes and moderate-sized datasets. For very large datasets, a more
    memory-efficient neighborhood search structure would be preferable.
    """
    n_samples = X.shape[0]

    if n_samples == 0:
        raise DataValidationError("Cannot compute distances on an empty dataset.")

    # Broadcasting-based computation of pairwise Euclidean distances.
    # Shape evolution:
    # X[:, None, :] -> (n, 1, d)
    # X[None, :, :] -> (1, n, d)
    # Difference -> (n, n, d)
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    if dist_matrix.shape != (n_samples, n_samples):
        raise OpticsError("Unexpected shape encountered in distance matrix.")

    return dist_matrix


# =============================================================================
# OPTICS implementation
# =============================================================================

def get_neighbors(dist_matrix: np.ndarray, point_idx: int, max_eps: float) -> List[int]:
    """
    Return indices of all points within max_eps of point_idx, including itself.

    OPTICS commonly includes the point itself in the neighborhood when counting
    min_samples for core-distance determination.
    """
    distances = dist_matrix[point_idx]
    if math.isinf(max_eps):
        return list(np.arange(len(distances)))
    return list(np.where(distances <= max_eps)[0])


def compute_core_distance(
    dist_matrix: np.ndarray,
    point_idx: int,
    neighbors: List[int],
    min_samples: int
) -> float:
    """
    Compute the core distance of a point.

    Definition:
    The core distance of a point is the distance from that point to its
    min_samples-th nearest neighbor (counting the point itself if included).

    If there are fewer than min_samples points in its neighborhood, the core
    distance is undefined, and we represent it as np.inf.
    """
    if len(neighbors) < min_samples:
        return np.inf

    neighbor_distances = np.sort(dist_matrix[point_idx, neighbors])

    # Since neighbors includes the point itself at distance 0,
    # the min_samples-th neighbor is index min_samples - 1.
    core_dist = neighbor_distances[min_samples - 1]
    return float(core_dist)


def update_seeds(
    point_idx: int,
    neighbors: List[int],
    processed: np.ndarray,
    reachability: np.ndarray,
    predecessor: np.ndarray,
    seeds: List[Tuple[float, int]],
    dist_matrix: np.ndarray,
    core_distance: np.ndarray
) -> None:
    """
    Update reachability distances of unprocessed neighbors.

    For each unprocessed neighbor o of point p:
        new_reachability = max(core_distance(p), distance(p, o))

    If this improves the neighbor's current reachability value,
    update it and push it into the priority queue.

    Notes:
    - We allow duplicate entries in the heap for simplicity.
    - When popping from the heap later, we skip already-processed points.
    - This is a standard practical approach and keeps the code simpler.
    """
    p_core = core_distance[point_idx]
    if not np.isfinite(p_core):
        return

    for neighbor_idx in neighbors:
        if processed[neighbor_idx]:
            continue

        new_reachability = max(p_core, dist_matrix[point_idx, neighbor_idx])

        if new_reachability < reachability[neighbor_idx]:
            reachability[neighbor_idx] = new_reachability
            predecessor[neighbor_idx] = point_idx
            heapq.heappush(seeds, (new_reachability, neighbor_idx))


def optics(
    X: np.ndarray,
    min_samples: int = 5,
    max_eps: float = np.inf
) -> OpticsResult:
    """
    Run the OPTICS algorithm on a numeric feature matrix.

    Parameters:
        X:
            2D numeric array of shape (n_samples, n_features)
        min_samples:
            Minimum number of samples for a point to be considered core
        max_eps:
            Maximum radius for neighborhood queries

    Returns:
        OpticsResult containing ordering, reachability, core distances, predecessor
    """
    if X.ndim != 2:
        raise DataValidationError("Feature matrix X must be 2-dimensional.")

    n_samples = X.shape[0]

    if n_samples < min_samples:
        raise DataValidationError(
            f"Dataset has only {n_samples} rows, but min_samples={min_samples}. "
            "Please reduce min_samples or use more data."
        )

    dist_matrix = pairwise_distances(X)

    processed = np.zeros(n_samples, dtype=bool)
    reachability = np.full(n_samples, np.inf, dtype=float)
    core_distance = np.full(n_samples, np.inf, dtype=float)
    predecessor = np.full(n_samples, -1, dtype=int)
    ordering: List[int] = []

    # Main OPTICS loop:
    # Repeatedly expand the next unprocessed point.
    for point_idx in range(n_samples):
        if processed[point_idx]:
            continue

        neighbors = get_neighbors(dist_matrix, point_idx, max_eps)
        processed[point_idx] = True
        ordering.append(point_idx)

        core_distance[point_idx] = compute_core_distance(
            dist_matrix=dist_matrix,
            point_idx=point_idx,
            neighbors=neighbors,
            min_samples=min_samples
        )

        if np.isfinite(core_distance[point_idx]):
            seeds: List[Tuple[float, int]] = []

            update_seeds(
                point_idx=point_idx,
                neighbors=neighbors,
                processed=processed,
                reachability=reachability,
                predecessor=predecessor,
                seeds=seeds,
                dist_matrix=dist_matrix,
                core_distance=core_distance
            )

            while seeds:
                _, next_point = heapq.heappop(seeds)

                if processed[next_point]:
                    continue

                next_neighbors = get_neighbors(dist_matrix, next_point, max_eps)
                processed[next_point] = True
                ordering.append(next_point)

                core_distance[next_point] = compute_core_distance(
                    dist_matrix=dist_matrix,
                    point_idx=next_point,
                    neighbors=next_neighbors,
                    min_samples=min_samples
                )

                if np.isfinite(core_distance[next_point]):
                    update_seeds(
                        point_idx=next_point,
                        neighbors=next_neighbors,
                        processed=processed,
                        reachability=reachability,
                        predecessor=predecessor,
                        seeds=seeds,
                        dist_matrix=dist_matrix,
                        core_distance=core_distance
                    )

    return OpticsResult(
        ordering=ordering,
        reachability=reachability,
        core_distance=core_distance,
        predecessor=predecessor
    )


# =============================================================================
# Cluster extraction
# =============================================================================

def extract_clusters_dbscan_style(
    optics_result: OpticsResult,
    cluster_eps: float
) -> np.ndarray:
    """
    Extract simple clusters from OPTICS ordering using a DBSCAN-style epsilon cut.

    This is a common and easy-to-understand post-processing approach:
    - Iterate through points in OPTICS order
    - Start a new cluster when reachability > eps but core_distance <= eps
    - Add subsequent points while density-connectivity holds
    - Label points as noise when they are not density-reachable under eps

    Returns:
        cluster_labels:
            Array of cluster IDs, with -1 meaning noise
    """
    ordering = optics_result.ordering
    reachability = optics_result.reachability
    core_distance = optics_result.core_distance

    n_samples = len(reachability)
    labels = np.full(n_samples, -1, dtype=int)

    current_cluster_id = -1

    for point_idx in ordering:
        r = reachability[point_idx]
        c = core_distance[point_idx]

        # If the point is not reachable within cluster_eps,
        # it may still start a new cluster if it is itself a core point.
        if r > cluster_eps:
            if c <= cluster_eps:
                current_cluster_id += 1
                labels[point_idx] = current_cluster_id
            else:
                labels[point_idx] = -1
        else:
            # Reachable from the current cluster.
            labels[point_idx] = current_cluster_id if current_cluster_id >= 0 else -1

    return labels


# =============================================================================
# Reporting / output
# =============================================================================

def summarize_clusters(labels: np.ndarray) -> str:
    """
    Build a readable cluster summary string.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    lines = []
    for label, count in zip(unique_labels, counts):
        if label == -1:
            lines.append(f"Noise points: {count}")
        else:
            lines.append(f"Cluster {label}: {count} points")

    return "\n".join(lines)


def build_output_dataframe(
    original_df: pd.DataFrame,
    numeric_columns: List[str],
    optics_result: OpticsResult,
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Merge clustering outputs back into a copy of the original data.
    """
    output_df = original_df.copy()

    output_df["optics_cluster"] = cluster_labels
    output_df["optics_reachability"] = optics_result.reachability
    output_df["optics_core_distance"] = optics_result.core_distance
    output_df["optics_predecessor"] = optics_result.predecessor

    # Rank indicates the point's position in the OPTICS ordering.
    rank = np.full(len(output_df), -1, dtype=int)
    for idx, point_idx in enumerate(optics_result.ordering):
        rank[point_idx] = idx
    output_df["optics_order"] = rank

    # Optional convenience field so the user can see what features were used.
    output_df.attrs["numeric_columns_used"] = numeric_columns

    return output_df


def save_output_csv(output_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save clustering results to CSV with robust error handling.
    """
    try:
        output_df.to_csv(output_path, index=False)
    except PermissionError as exc:
        raise OpticsError(
            f"Permission denied while writing output file: {output_path}"
        ) from exc
    except OSError as exc:
        raise OpticsError(
            f"Failed to write output file: {output_path}\n{exc}"
        ) from exc
    except Exception as exc:
        raise OpticsError(
            f"Unexpected error while saving output file: {exc}"
        ) from exc


# =============================================================================
# Main program flow
# =============================================================================

def main() -> int:
    """
    Main entry point.

    Returns:
        0 on success, non-zero on failure.
    """
    try:
        args = parse_arguments()

        validate_parameters(
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            cluster_eps=args.cluster_eps
        )

        input_path = resolve_file_path(args.input)
        output_path = resolve_file_path(args.output)

        print(f"Loading input file: {input_path}")
        df = load_csv_file(input_path)

        print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

        original_aligned_df, numeric_df, numeric_columns = prepare_numeric_features(
            df=df,
            drop_na=args.drop_na
        )

        print("Numeric columns selected for clustering:")
        for col in numeric_columns:
            print(f"  - {col}")

        X = standardize_features(numeric_df)

        print(
            f"Running OPTICS with min_samples={args.min_samples}, "
            f"max_eps={args.max_eps}, cluster_eps={args.cluster_eps}"
        )

        optics_result = optics(
            X=X,
            min_samples=args.min_samples,
            max_eps=args.max_eps
        )

        cluster_labels = extract_clusters_dbscan_style(
            optics_result=optics_result,
            cluster_eps=args.cluster_eps
        )

        output_df = build_output_dataframe(
            original_df=original_aligned_df,
            numeric_columns=numeric_columns,
            optics_result=optics_result,
            cluster_labels=cluster_labels
        )

        save_output_csv(output_df, output_path)

        print("\nOPTICS completed successfully.")
        print("\nCluster summary:")
        print(summarize_clusters(cluster_labels))
        print(f"\nResults written to: {output_path}")

        return 0

    except FileNotFoundError as exc:
        print(f"[FILE ERROR] {exc}", file=sys.stderr)
        return 1

    except ParameterValidationError as exc:
        print(f"[PARAMETER ERROR] {exc}", file=sys.stderr)
        return 2

    except DataValidationError as exc:
        print(f"[DATA ERROR] {exc}", file=sys.stderr)
        return 3

    except OpticsError as exc:
        print(f"[OPTICS ERROR] {exc}", file=sys.stderr)
        return 4

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution cancelled by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        # Catch-all for unexpected failures so the script exits cleanly
        # and provides a useful message rather than a raw traceback only.
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    sys.exit(main())
