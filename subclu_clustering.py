#!/usr/bin/env python3
"""
subclu_clustering.py

A heavily commented, self-contained Python script that performs SUBCLU
(Subspace Clustering based on Density-Connected Sets) on a CSV file located in
the same folder as this script.

What this script does
---------------------
1. Loads a CSV file from the same directory (defaults to
   'test_military_radar_readings.csv').
2. Identifies numeric columns and optionally derives extra numeric time features
   from a timestamp column.
3. Cleans and standardizes the numeric data.
4. Runs a practical implementation of the SUBCLU algorithm using DBSCAN in all
   promising subspaces.
5. Writes detailed outputs:
   - A cluster membership CSV for every subspace where clusters were found.
   - A summary CSV of all discovered subspace clusters.
   - Optional scatter plots for 2D subspaces.

Important note
--------------
SUBCLU is a subspace clustering algorithm built on DBSCAN. Exact textbook
implementations can become computationally expensive very quickly as the number
of dimensions grows. This script implements the core SUBCLU idea with careful,
robust engineering and extensive comments so it is usable and understandable on
real datasets.

Usage examples
--------------
python subclu_clustering.py
python subclu_clustering.py --csv my_data.csv
python subclu_clustering.py --eps 0.8 --min-samples 6
python subclu_clustering.py --no-plots

Dependencies
------------
- pandas
- numpy
- scikit-learn
- matplotlib

Install them with:
    pip install pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Data structures used throughout the script.
# -----------------------------------------------------------------------------

@dataclass
class SubspaceResult:
    """
    Stores the output for one subspace.

    Attributes
    ----------
    columns:
        Tuple of column names defining the subspace.
    labels:
        DBSCAN labels for every row in the processed dataframe.
        Noise points are labeled -1.
    cluster_count:
        Number of non-noise clusters in this subspace.
    noise_count:
        Number of points marked as noise.
    clustered_count:
        Number of points assigned to real clusters.
    """

    columns: Tuple[str, ...]
    labels: np.ndarray
    cluster_count: int
    noise_count: int
    clustered_count: int


# -----------------------------------------------------------------------------
# Helper functions for argument parsing and logging.
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run SUBCLU clustering on a CSV file in the same folder."
    )

    parser.add_argument(
        "--csv",
        default="test_military_radar_readings.csv",
        help="CSV filename to load from the same folder as this script. Default: %(default)s",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.9,
        help="DBSCAN eps radius used inside SUBCLU. Default: %(default)s",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter. Default: %(default)s",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=3,
        help=(
            "Maximum subspace dimensionality to explore. Higher values are much more expensive. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Name of an optional timestamp column to derive time features from. Default: %(default)s",
    )
    parser.add_argument(
        "--include-derived-time-features",
        action="store_true",
        help="If set, derive numeric time features from the timestamp column when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="subclu_output",
        help="Directory (created next to the script) where outputs will be written. Default: %(default)s",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of scatter plots for 2D subspaces.",
    )
    parser.add_argument(
        "--min-cluster-size-to-save",
        type=int,
        default=2,
        help=(
            "Minimum number of points in a discovered cluster for the result to be written out. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity. Default: %(default)s",
    )

    return parser



def configure_logging(level: str) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Core utility functions.
# -----------------------------------------------------------------------------


def resolve_paths(csv_name: str, output_dir_name: str) -> Tuple[Path, Path, Path]:
    """
    Resolve paths relative to the script location, not the current shell.

    This makes the script more reliable when users launch it from a different
    working directory.
    """
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    csv_path = script_dir / csv_name
    output_dir = script_dir / output_dir_name
    return script_path, csv_path, output_dir



def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments early and clearly."""
    if args.eps <= 0:
        raise ValueError("--eps must be greater than 0.")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be greater than 0.")
    if args.max_dim <= 0:
        raise ValueError("--max-dim must be greater than 0.")
    if args.min_cluster_size_to_save <= 0:
        raise ValueError("--min-cluster-size-to-save must be greater than 0.")



def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV with robust error handling and useful error messages."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file was not found: {csv_path}\n"
            "Place the CSV in the same folder as this script, or use --csv to specify its filename."
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Path exists but is not a regular file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV file: {csv_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while reading CSV: {csv_path}") from exc

    if df.empty:
        raise ValueError("The CSV loaded successfully but contains no rows.")

    logging.info("Loaded CSV with %d rows and %d columns.", df.shape[0], df.shape[1])
    return df



def add_derived_time_features(
    df: pd.DataFrame, timestamp_column: str
) -> pd.DataFrame:
    """
    Derive numeric features from a timestamp column if it exists and can be parsed.

    Why this helps:
    SUBCLU operates on numeric subspaces. Raw timestamps are not always directly
    useful, but components such as hour-of-day or day-of-week can be.
    """
    if timestamp_column not in df.columns:
        logging.warning(
            "Timestamp column '%s' not found. Skipping derived time features.",
            timestamp_column,
        )
        return df

    result = df.copy()

    try:
        parsed = pd.to_datetime(result[timestamp_column], errors="coerce")
    except Exception as exc:
        logging.warning(
            "Failed to parse timestamp column '%s': %s. Skipping derived time features.",
            timestamp_column,
            exc,
        )
        return result

    valid_count = parsed.notna().sum()
    if valid_count == 0:
        logging.warning(
            "Timestamp column '%s' could not be parsed for any rows. Skipping derived time features.",
            timestamp_column,
        )
        return result

    result["derived_hour"] = parsed.dt.hour
    result["derived_minute"] = parsed.dt.minute
    result["derived_dayofweek"] = parsed.dt.dayofweek
    result["derived_dayofmonth"] = parsed.dt.day
    result["derived_month"] = parsed.dt.month

    logging.info(
        "Derived time features from '%s' using %d parsable timestamps.",
        timestamp_column,
        valid_count,
    )
    return result



def select_numeric_features(df: pd.DataFrame) -> List[str]:
    """
    Select numeric columns for clustering.

    SUBCLU is designed around numerical distance-based density clustering.
    Non-numeric columns are excluded to avoid invalid distance calculations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError(
            "No numeric columns were found after preprocessing. SUBCLU requires numeric features."
        )

    logging.info("Numeric columns selected for clustering: %s", numeric_cols)
    return numeric_cols



def prepare_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Clean and standardize the selected numeric columns.

    Steps:
    1. Keep only the requested numeric columns.
    2. Replace inf values with NaN.
    3. Drop rows that contain NaNs in any selected feature.
    4. Standardize columns so one large-scale feature does not dominate DBSCAN.

    Returns
    -------
    cleaned_df:
        Original dataframe rows aligned to the cleaned numeric matrix.
    scaled_matrix:
        Standardized numeric matrix for clustering.
    """
    feature_df = df.loc[:, feature_cols].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    valid_mask = feature_df.notna().all(axis=1)
    dropped_rows = (~valid_mask).sum()
    if dropped_rows > 0:
        logging.warning(
            "Dropping %d rows with missing or infinite values in the selected numeric features.",
            dropped_rows,
        )

    cleaned_df = df.loc[valid_mask].reset_index(drop=False).rename(columns={"index": "original_row_index"})
    cleaned_features = feature_df.loc[valid_mask].reset_index(drop=True)

    if cleaned_features.empty:
        raise ValueError("After removing invalid numeric rows, no data remains for clustering.")

    try:
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(cleaned_features.values)
    except Exception as exc:
        raise RuntimeError("Failed to standardize numeric features.") from exc

    logging.info(
        "Prepared feature matrix with %d valid rows and %d numeric features.",
        scaled_matrix.shape[0],
        scaled_matrix.shape[1],
    )
    return cleaned_df, scaled_matrix



def run_dbscan(matrix: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN safely and return labels."""
    try:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = model.fit_predict(matrix)
        return labels
    except Exception as exc:
        raise RuntimeError("DBSCAN failed during execution.") from exc



def summarize_labels(labels: np.ndarray) -> Tuple[int, int, int, Dict[int, int]]:
    """
    Summarize DBSCAN labels.

    Returns:
    - cluster_count: number of labels excluding noise (-1)
    - noise_count: count of -1 labels
    - clustered_count: count of labels not equal to -1
    - cluster_sizes: mapping label -> size for non-noise clusters
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    noise_count = int(label_to_count.get(-1, 0))
    cluster_sizes = {int(k): int(v) for k, v in label_to_count.items() if k != -1}
    cluster_count = len(cluster_sizes)
    clustered_count = int(sum(cluster_sizes.values()))
    return cluster_count, noise_count, clustered_count, cluster_sizes


# -----------------------------------------------------------------------------
# SUBCLU implementation.
# -----------------------------------------------------------------------------


def generate_candidates(prev_subspaces: List[Tuple[str, ...]], target_dim: int) -> List[Tuple[str, ...]]:
    """
    Generate candidate subspaces of size target_dim using an Apriori-style join.

    SUBCLU relies on downward closure ideas: we only consider larger subspaces
    when their lower-dimensional projections were promising.
    """
    candidates = set()
    prev_set = set(prev_subspaces)

    for left, right in itertools.combinations(sorted(prev_subspaces), 2):
        merged = tuple(sorted(set(left) | set(right)))
        if len(merged) != target_dim:
            continue

        # Prune candidates whose all (target_dim - 1)-subspaces were not known/promising.
        all_subsets_exist = all(
            tuple(sorted(subset)) in prev_set
            for subset in itertools.combinations(merged, target_dim - 1)
        )
        if all_subsets_exist:
            candidates.add(merged)

    return sorted(candidates)



def subclu(
    cleaned_df: pd.DataFrame,
    scaled_matrix: np.ndarray,
    feature_cols: Sequence[str],
    eps: float,
    min_samples: int,
    max_dim: int,
    min_cluster_size_to_save: int,
) -> List[SubspaceResult]:
    """
    Run a practical SUBCLU search over promising subspaces.

    High-level algorithm:
    1. Run DBSCAN on every 1D subspace.
    2. Keep subspaces that contain at least one non-noise cluster.
    3. Iteratively build larger candidates only from successful lower-dimensional subspaces.
    4. Run DBSCAN in each candidate subspace.
    5. Save only subspaces with at least one sufficiently large cluster.
    """
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    all_results: List[SubspaceResult] = []
    promising_by_dim: Dict[int, List[Tuple[str, ...]]] = {}

    # ----------------------------
    # Step 1: evaluate 1D subspaces
    # ----------------------------
    promising_1d: List[Tuple[str, ...]] = []
    logging.info("Evaluating 1D subspaces...")

    for col in feature_cols:
        subspace = (col,)
        indices = [feature_index[col]]
        labels = run_dbscan(scaled_matrix[:, indices], eps=eps, min_samples=min_samples)
        cluster_count, noise_count, clustered_count, cluster_sizes = summarize_labels(labels)

        # Count how many clusters are large enough to care about.
        valid_cluster_sizes = [size for size in cluster_sizes.values() if size >= min_cluster_size_to_save]
        if valid_cluster_sizes:
            promising_1d.append(subspace)
            all_results.append(
                SubspaceResult(
                    columns=subspace,
                    labels=labels,
                    cluster_count=cluster_count,
                    noise_count=noise_count,
                    clustered_count=clustered_count,
                )
            )
            logging.info(
                "1D subspace %s produced %d clusters (%d clustered, %d noise).",
                subspace,
                cluster_count,
                clustered_count,
                noise_count,
            )
        else:
            logging.debug("1D subspace %s produced no usable clusters.", subspace)

    promising_by_dim[1] = promising_1d

    # -------------------------------------------------------------
    # Step 2+: iteratively build and evaluate larger-dimensional spaces
    # -------------------------------------------------------------
    for dim in range(2, max_dim + 1):
        prev_promising = promising_by_dim.get(dim - 1, [])
        if not prev_promising:
            logging.info(
                "No promising %dD subspaces found, so search stops before %dD.",
                dim - 1,
                dim,
            )
            break

        logging.info("Generating candidate %dD subspaces...", dim)
        candidates = generate_candidates(prev_promising, dim)
        if not candidates:
            logging.info("No candidate %dD subspaces generated.", dim)
            break

        logging.info("Evaluating %d candidate %dD subspaces...", len(candidates), dim)
        promising_current: List[Tuple[str, ...]] = []

        for subspace in candidates:
            try:
                indices = [feature_index[col] for col in subspace]
            except KeyError as exc:
                logging.error("Internal feature lookup error for subspace %s: %s", subspace, exc)
                continue

            labels = run_dbscan(scaled_matrix[:, indices], eps=eps, min_samples=min_samples)
            cluster_count, noise_count, clustered_count, cluster_sizes = summarize_labels(labels)
            valid_cluster_sizes = [size for size in cluster_sizes.values() if size >= min_cluster_size_to_save]

            if valid_cluster_sizes:
                promising_current.append(subspace)
                all_results.append(
                    SubspaceResult(
                        columns=subspace,
                        labels=labels,
                        cluster_count=cluster_count,
                        noise_count=noise_count,
                        clustered_count=clustered_count,
                    )
                )
                logging.info(
                    "%dD subspace %s produced %d clusters (%d clustered, %d noise).",
                    dim,
                    subspace,
                    cluster_count,
                    clustered_count,
                    noise_count,
                )
            else:
                logging.debug("%dD subspace %s produced no usable clusters.", dim, subspace)

        promising_by_dim[dim] = promising_current

    # Sort results for stable, readable output.
    all_results.sort(key=lambda r: (len(r.columns), r.columns))
    logging.info("SUBCLU finished. %d promising subspaces discovered.", len(all_results))
    return all_results


# -----------------------------------------------------------------------------
# Output generation.
# -----------------------------------------------------------------------------


def sanitize_filename_part(text: str) -> str:
    """Convert a subspace label into a safe filename fragment."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)



def save_subspace_memberships(
    cleaned_df: pd.DataFrame,
    results: Sequence[SubspaceResult],
    output_dir: Path,
) -> List[dict]:
    """
    Save per-subspace membership CSVs and build a summary structure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[dict] = []

    for idx, result in enumerate(results, start=1):
        subspace_name = "__".join(result.columns)
        safe_name = sanitize_filename_part(subspace_name)
        subspace_df = cleaned_df.copy()
        subspace_df["cluster_label"] = result.labels

        # Compute cluster sizes for convenience in the output file.
        label_counts = pd.Series(result.labels).value_counts().to_dict()
        subspace_df["cluster_size"] = subspace_df["cluster_label"].map(label_counts)
        subspace_df["subspace"] = subspace_name

        membership_path = output_dir / f"subspace_{idx:03d}_{safe_name}_memberships.csv"
        try:
            subspace_df.to_csv(membership_path, index=False)
        except Exception as exc:
            logging.error("Failed to write membership CSV for %s: %s", result.columns, exc)
            continue

        summary_rows.append(
            {
                "subspace": subspace_name,
                "dimension": len(result.columns),
                "columns": ", ".join(result.columns),
                "cluster_count": result.cluster_count,
                "clustered_count": result.clustered_count,
                "noise_count": result.noise_count,
                "membership_file": membership_path.name,
            }
        )

    return summary_rows



def save_summary(summary_rows: Sequence[dict], output_dir: Path) -> Path:
    """Write the overall subspace summary CSV."""
    summary_path = output_dir / "subclu_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["dimension", "clustered_count", "cluster_count"], ascending=[True, False, False])
    summary_df.to_csv(summary_path, index=False)
    return summary_path



def plot_2d_subspaces(
    cleaned_df: pd.DataFrame,
    results: Sequence[SubspaceResult],
    output_dir: Path,
) -> int:
    """
    Create scatter plots for all 2D subspaces.

    Why only 2D plots?
    They are easy to interpret visually and useful for quick sanity checking.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_count = 0

    for idx, result in enumerate(results, start=1):
        if len(result.columns) != 2:
            continue

        x_col, y_col = result.columns
        if x_col not in cleaned_df.columns or y_col not in cleaned_df.columns:
            logging.warning("Skipping plot for %s because columns are missing.", result.columns)
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            cleaned_df[x_col],
            cleaned_df[y_col],
            c=result.labels,
            alpha=0.8,
        )
        ax.set_title(f"SUBCLU / DBSCAN clusters for subspace: {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

        # Note: We deliberately keep the legend simple because DBSCAN labels can
        # vary and may include noise (-1). A full label legend is still possible,
        # but often becomes cluttered for dense outputs.
        plt.tight_layout()

        safe_name = sanitize_filename_part(f"{x_col}__{y_col}")
        plot_path = plots_dir / f"subspace_{idx:03d}_{safe_name}.png"
        try:
            fig.savefig(plot_path, dpi=150)
            plot_count += 1
        except Exception as exc:
            logging.error("Failed to save plot for %s: %s", result.columns, exc)
        finally:
            plt.close(fig)

    return plot_count


# -----------------------------------------------------------------------------
# Main application flow.
# -----------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        validate_args(args)
        script_path, csv_path, output_dir = resolve_paths(args.csv, args.output_dir)

        logging.info("Script location: %s", script_path)
        logging.info("CSV path: %s", csv_path)
        logging.info("Output directory: %s", output_dir)

        df = load_csv(csv_path)

        if args.include_derived_time_features:
            df = add_derived_time_features(df, args.timestamp_column)

        feature_cols = select_numeric_features(df)
        cleaned_df, scaled_matrix = prepare_feature_matrix(df, feature_cols)

        if args.max_dim > len(feature_cols):
            logging.warning(
                "Requested max_dim=%d, but only %d numeric features are available. Reducing max_dim.",
                args.max_dim,
                len(feature_cols),
            )
            args.max_dim = len(feature_cols)

        results = subclu(
            cleaned_df=cleaned_df,
            scaled_matrix=scaled_matrix,
            feature_cols=feature_cols,
            eps=args.eps,
            min_samples=args.min_samples,
            max_dim=args.max_dim,
            min_cluster_size_to_save=args.min_cluster_size_to_save,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        if not results:
            logging.warning(
                "No subspaces produced usable clusters. Consider adjusting --eps, --min-samples, or preprocessing choices."
            )
            # Still write an empty summary so the run is fully traceable.
            empty_summary_path = output_dir / "subclu_summary.csv"
            pd.DataFrame(
                columns=[
                    "subspace",
                    "dimension",
                    "columns",
                    "cluster_count",
                    "clustered_count",
                    "noise_count",
                    "membership_file",
                ]
            ).to_csv(empty_summary_path, index=False)
            logging.info("Wrote empty summary to %s", empty_summary_path)
            return 0

        summary_rows = save_subspace_memberships(cleaned_df, results, output_dir)
        summary_path = save_summary(summary_rows, output_dir)
        logging.info("Wrote summary CSV: %s", summary_path)

        if not args.no_plots:
            plot_count = plot_2d_subspaces(cleaned_df, results, output_dir)
            logging.info("Generated %d plot(s).", plot_count)

        logging.info("Run complete. Discovered %d promising subspaces.", len(results))
        return 0

    except KeyboardInterrupt:
        logging.error("Execution interrupted by user.")
        return 130
    except Exception as exc:
        logging.error("Fatal error: %s", exc)
        logging.debug("Full traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
