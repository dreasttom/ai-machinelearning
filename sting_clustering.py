#!/usr/bin/env python3
"""
STING (Statistical Information Grid) clustering for CSV data.

This script assumes the input dataset CSV is in the same folder as the script by default.
It performs a practical STING-style grid-based clustering workflow:

1. Load CSV data safely.
2. Select numeric columns for clustering.
3. Clean / impute missing numeric values.
4. Scale features robustly.
5. Build an N-dimensional grid over the feature space.
6. Compute per-cell counts/statistics.
7. Mark statistically dense cells using a configurable threshold.
8. Merge adjacent dense cells into connected components (clusters).
9. Save labeled output and graphics.

Notes:
- "Classical" STING is hierarchical and cell-statistics driven. This script implements a
  practical single-resolution STING-style approach that is faithful to the grid-statistics
  clustering concept while remaining readable and maintainable.
- For high-dimensional data, adjacency checks can become expensive. This implementation is
  designed for small to medium numeric datasets.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Custom exception definitions
# -----------------------------
class StingClusteringError(Exception):
    """Base exception for predictable clustering failures."""


class DataValidationError(StingClusteringError):
    """Raised when the input dataset cannot be used safely."""


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class StingResult:
    """Container for STING clustering output."""

    labels: np.ndarray
    dense_cell_map: Dict[Tuple[int, ...], int]
    cell_counts: Dict[Tuple[int, ...], int]
    scaled_data: np.ndarray
    numeric_columns: List[str]
    pca_2d: np.ndarray
    pca_model: PCA
    dense_threshold: float
    bins_per_dim: int


# -----------------------------
# Utility functions
# -----------------------------
def log(msg: str) -> None:
    """Print status messages consistently to stderr."""
    print(msg, file=sys.stderr)



def safe_mkdir(path: str) -> None:
    """Create a directory if it does not exist, with clear failure handling."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise StingClusteringError(f"Unable to create directory: {path!r}. {exc}") from exc



def validate_file_exists(path: str) -> None:
    """Ensure the input file exists before attempting to load it."""
    if not os.path.isfile(path):
        raise DataValidationError(
            f"Input file not found: {path!r}. Place the CSV in the same folder as the script "
            "or pass --input explicitly."
        )



def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with robust error messages for common failures."""
    validate_file_exists(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"The CSV file is empty: {path!r}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"The CSV file could not be parsed: {path!r}. {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            f"The CSV file could not be decoded as text: {path!r}. Try saving as UTF-8."
        ) from exc
    except Exception as exc:
        raise DataValidationError(f"Unexpected error while loading CSV {path!r}: {exc}") from exc

    if df.empty:
        raise DataValidationError("The CSV loaded successfully but contains no rows.")

    return df



def select_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select numeric columns only.

    STING clustering is naturally defined over numeric feature space. Non-numeric columns
    (timestamps, category labels, free text) are preserved in the final output but excluded
    from clustering.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_columns = numeric_df.columns.tolist()

    if len(numeric_columns) < 2:
        raise DataValidationError(
            "Need at least two numeric columns for meaningful grid-based clustering and graphics. "
            f"Found numeric columns: {numeric_columns}"
        )

    # Remove constant columns because they do not help clustering and can break binning logic.
    constant_cols = [col for col in numeric_columns if numeric_df[col].nunique(dropna=True) <= 1]
    if constant_cols:
        numeric_df = numeric_df.drop(columns=constant_cols)
        numeric_columns = numeric_df.columns.tolist()

    if len(numeric_columns) < 2:
        raise DataValidationError(
            "After removing constant numeric columns, fewer than two numeric columns remain. "
            "Cannot build meaningful clusters/plots."
        )

    return numeric_df, numeric_columns



def clean_and_scale(numeric_df: pd.DataFrame) -> np.ndarray:
    """
    Impute missing values and scale the data.

    Why scale?
    Grid methods depend on feature space geometry. Without scaling, a large-range feature can
    dominate the grid and overwhelm smaller-range features.
    """
    try:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(numeric_df)
    except ValueError as exc:
        raise DataValidationError(f"Unable to impute numeric data: {exc}") from exc

    if not np.isfinite(imputed).all():
        raise DataValidationError(
            "Numeric data still contains non-finite values after imputation. "
            "Please inspect NaN/inf values in the dataset."
        )

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(imputed)
    except Exception as exc:
        raise StingClusteringError(f"Failed to scale numeric data: {exc}") from exc

    if scaled.shape[0] == 0 or scaled.shape[1] == 0:
        raise DataValidationError("Scaled feature matrix is empty.")

    return scaled



def compute_edges(values: np.ndarray, bins_per_dim: int) -> List[np.ndarray]:
    """
    Compute bin edges per feature.

    Uses min/max of scaled data. A tiny padding is added to ensure the maximum value is included.
    """
    if bins_per_dim < 2:
        raise ValueError("bins_per_dim must be at least 2.")

    edges: List[np.ndarray] = []
    for dim in range(values.shape[1]):
        col = values[:, dim]
        col_min = float(np.min(col))
        col_max = float(np.max(col))

        if math.isclose(col_min, col_max):
            # Degenerate edge case: if somehow a constant column remained, give it a tiny range.
            col_min -= 1e-6
            col_max += 1e-6
        else:
            pad = max(1e-9, (col_max - col_min) * 1e-6)
            col_max += pad

        edges.append(np.linspace(col_min, col_max, bins_per_dim + 1))
    return edges



def assign_points_to_cells(values: np.ndarray, edges: Sequence[np.ndarray]) -> np.ndarray:
    """
    Convert each point to an integer cell index tuple.

    Each dimension is digitized independently, then combined into an N-dimensional cell index.
    """
    n_rows, n_dims = values.shape
    if n_dims != len(edges):
        raise ValueError("Mismatch between data dimensions and edge definitions.")

    cell_indices = np.zeros((n_rows, n_dims), dtype=int)
    for dim in range(n_dims):
        idx = np.digitize(values[:, dim], bins=edges[dim][1:-1], right=False)
        # Guarantee bounds even in the presence of floating-point edge effects.
        idx = np.clip(idx, 0, len(edges[dim]) - 2)
        cell_indices[:, dim] = idx

    return cell_indices



def count_cells(cell_indices: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """Count how many points fall into each occupied grid cell."""
    counts: Dict[Tuple[int, ...], int] = {}
    for row in cell_indices:
        key = tuple(int(x) for x in row)
        counts[key] = counts.get(key, 0) + 1
    return counts



def choose_dense_threshold(
    cell_counts: Dict[Tuple[int, ...], int],
    percentile: float,
    min_count: int,
) -> float:
    """
    Choose a density threshold using occupied-cell counts.

    Dense cells are identified as cells with counts >= max(min_count, percentile(counts)).
    This produces a flexible threshold that adapts to dataset density while enforcing a hard
    lower bound so that singletons do not become clusters by default.
    """
    if not cell_counts:
        raise StingClusteringError("No occupied grid cells were created from the data.")

    counts = np.array(list(cell_counts.values()), dtype=float)
    if counts.size == 0:
        raise StingClusteringError("No cell counts available for threshold selection.")

    percentile = float(np.clip(percentile, 0.0, 100.0))
    adaptive = float(np.percentile(counts, percentile))
    threshold = max(float(min_count), adaptive)
    return threshold



def neighbor_cells(cell: Tuple[int, ...], bins_per_dim: int) -> Iterable[Tuple[int, ...]]:
    """
    Yield adjacent cells using full Moore neighborhood in N dimensions.

    Two dense cells touching by face/edge/corner are considered connected. This is a practical
    and common choice in grid-based clustering because it merges contiguous dense regions.
    """
    deltas = [-1, 0, 1]

    def rec_build(prefix: List[int], dim: int):
        if dim == len(cell):
            if any(prefix[i] != cell[i] for i in range(len(cell))):
                yield tuple(prefix)
            return
        for delta in deltas:
            coord = cell[dim] + delta
            if 0 <= coord < bins_per_dim:
                prefix.append(coord)
                yield from rec_build(prefix, dim + 1)
                prefix.pop()

    yield from rec_build([], 0)



def connected_components(
    dense_cells: Sequence[Tuple[int, ...]],
    bins_per_dim: int,
) -> Dict[Tuple[int, ...], int]:
    """
    Merge adjacent dense cells into connected components.

    Returns a mapping from dense cell -> cluster id.
    """
    dense_set = set(dense_cells)
    cluster_map: Dict[Tuple[int, ...], int] = {}
    current_cluster_id = 0

    for cell in dense_cells:
        if cell in cluster_map:
            continue

        current_cluster_id += 1
        stack = [cell]
        cluster_map[cell] = current_cluster_id

        while stack:
            cur = stack.pop()
            for neighbor in neighbor_cells(cur, bins_per_dim):
                if neighbor in dense_set and neighbor not in cluster_map:
                    cluster_map[neighbor] = current_cluster_id
                    stack.append(neighbor)

    return cluster_map



def project_pca_2d(values: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """
    Project the numeric feature matrix to 2D for plotting.

    PCA is used only for visualization. Clustering itself still happens in the full numeric
    feature space.
    """
    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(values)
    except Exception as exc:
        raise StingClusteringError(f"Failed to compute PCA projection for plots: {exc}") from exc
    return coords, pca


# -----------------------------
# Core STING-style clustering
# -----------------------------
def run_sting(
    scaled_data: np.ndarray,
    bins_per_dim: int = 6,
    density_percentile: float = 65.0,
    min_dense_count: int = 2,
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int], Dict[Tuple[int, ...], int], float]:
    """
    Execute STING-style clustering.

    Returns
    -------
    labels
        Cluster labels per row. Noise / sparse cells are labeled -1.
    dense_cell_map
        Mapping from dense cell coordinate tuple to connected-component cluster id.
    cell_counts
        Occupied-cell counts.
    dense_threshold
        Final dense-cell threshold used.
    """
    if scaled_data.ndim != 2:
        raise ValueError("scaled_data must be a 2D array.")

    n_rows, n_dims = scaled_data.shape
    if n_rows < 2:
        raise DataValidationError("Need at least two rows to perform clustering.")
    if n_dims < 2:
        raise DataValidationError("Need at least two numeric dimensions for this implementation.")

    edges = compute_edges(scaled_data, bins_per_dim=bins_per_dim)
    cell_indices = assign_points_to_cells(scaled_data, edges)
    cell_counts = count_cells(cell_indices)

    dense_threshold = choose_dense_threshold(
        cell_counts=cell_counts,
        percentile=density_percentile,
        min_count=min_dense_count,
    )

    dense_cells = [cell for cell, count in cell_counts.items() if count >= dense_threshold]

    if not dense_cells:
        raise StingClusteringError(
            "No dense cells were found with the current parameters. Try fewer bins, a lower "
            "density percentile, or a smaller minimum dense-cell count."
        )

    dense_cell_map = connected_components(dense_cells, bins_per_dim=bins_per_dim)

    labels = np.full(shape=(n_rows,), fill_value=-1, dtype=int)
    for i, row in enumerate(cell_indices):
        cell = tuple(int(x) for x in row)
        if cell in dense_cell_map:
            labels[i] = dense_cell_map[cell]

    return labels, dense_cell_map, cell_counts, dense_threshold


# -----------------------------
# Output writers / plotters
# -----------------------------
def summarize_clusters(labels: np.ndarray) -> pd.DataFrame:
    """Build a summary table of cluster sizes, including noise."""
    unique, counts = np.unique(labels, return_counts=True)
    summary = pd.DataFrame({"cluster_label": unique, "row_count": counts})
    summary = summary.sort_values(by=["cluster_label"]).reset_index(drop=True)
    return summary



def save_outputs(
    original_df: pd.DataFrame,
    result: StingResult,
    output_dir: str,
) -> Tuple[str, str]:
    """Save labeled data and cluster summary as CSV files."""
    safe_mkdir(output_dir)

    labeled_df = original_df.copy()
    labeled_df["sting_cluster"] = result.labels

    labeled_path = os.path.join(output_dir, "sting_clustered_output.csv")
    summary_path = os.path.join(output_dir, "sting_cluster_summary.csv")

    try:
        labeled_df.to_csv(labeled_path, index=False)
        summarize_clusters(result.labels).to_csv(summary_path, index=False)
    except Exception as exc:
        raise StingClusteringError(f"Failed to save CSV outputs: {exc}") from exc

    return labeled_path, summary_path



def plot_cluster_scatter(result: StingResult, output_dir: str) -> str:
    """
    Create a 2D PCA scatter plot colored by cluster.

    Noise points (-1) are shown too, allowing the user to see sparse regions.
    """
    safe_mkdir(output_dir)
    labels = result.labels
    coords = result.pca_2d

    unique_labels = sorted(np.unique(labels))
    # Create a color map large enough for all labels including noise.
    base_colors = plt.get_cmap("tab20", max(2, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=45,
            alpha=0.8,
            label=name,
            color=base_colors(idx),
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_title("STING Clustering (PCA 2D Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "sting_clusters_pca.png")
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    except Exception as exc:
        plt.close(fig)
        raise StingClusteringError(f"Failed to save PCA cluster plot: {exc}") from exc
    plt.close(fig)
    return path



def plot_cluster_bar(labels: np.ndarray, output_dir: str) -> str:
    """Create a bar chart of cluster membership counts."""
    safe_mkdir(output_dir)
    summary = summarize_clusters(labels)
    label_names = ["Noise" if x == -1 else f"Cluster {x}" for x in summary["cluster_label"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(label_names, summary["row_count"])
    ax.set_title("STING Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Rows")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    path = os.path.join(output_dir, "sting_cluster_sizes.png")
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    except Exception as exc:
        plt.close(fig)
        raise StingClusteringError(f"Failed to save cluster size chart: {exc}") from exc
    plt.close(fig)
    return path



def plot_dense_grid(result: StingResult, output_dir: str) -> Optional[str]:
    """
    Plot dense cells on a 2D PCA projection grid when possible.

    Because clustering is performed in the original N-dimensional numeric space, directly showing
    the full grid is only easy in 2D. For higher-dimensional data, this plot uses the first two
    PCA components as a visual companion plot instead of a literal original-space grid.
    """
    safe_mkdir(output_dir)
    coords = result.pca_2d
    labels = result.labels

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=35, alpha=0.75)
    ax.set_title("STING Cluster Visualization (PCA Space)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "sting_visualization.png")
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    except Exception as exc:
        plt.close(fig)
        raise StingClusteringError(f"Failed to save visualization plot: {exc}") from exc
    plt.close(fig)
    return path


# -----------------------------
# Main workflow
# -----------------------------
def infer_default_input_path(script_dir: str) -> str:
    """Assume the radar dataset filename if it exists; otherwise pick the first CSV found."""
    preferred = os.path.join(script_dir, "test_military_radar_readings.csv")
    if os.path.isfile(preferred):
        return preferred

    csvs = sorted(
        [os.path.join(script_dir, name) for name in os.listdir(script_dir) if name.lower().endswith(".csv")]
    )
    if not csvs:
        raise DataValidationError(
            "No CSV files were found in the script directory. Provide --input or place a CSV next to the script."
        )
    return csvs[0]



def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run STING (Statistical Information Grid) clustering on a CSV dataset."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input CSV. Defaults to a CSV in the same folder as the script.",
    )
    parser.add_argument(
        "--output-dir",
        default="sting_output",
        help="Directory for CSV and plot outputs. Default: sting_output",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=6,
        help="Number of bins per dimension for the STING grid. Default: 6",
    )
    parser.add_argument(
        "--density-percentile",
        type=float,
        default=65.0,
        help="Percentile of occupied-cell counts used to define dense cells. Default: 65",
    )
    parser.add_argument(
        "--min-dense-count",
        type=int,
        default=2,
        help="Minimum count required for a cell to be considered dense. Default: 2",
    )
    return parser



def main() -> int:
    """Entry point with robust top-level error handling."""
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = args.input if args.input else infer_default_input_path(script_dir)
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(script_dir, output_dir)

        if args.bins < 2:
            raise DataValidationError("--bins must be at least 2.")
        if args.min_dense_count < 1:
            raise DataValidationError("--min-dense-count must be at least 1.")
        if not (0 <= args.density_percentile <= 100):
            raise DataValidationError("--density-percentile must be between 0 and 100.")

        log(f"Loading input CSV: {input_path}")
        df = load_csv(input_path)

        log("Selecting numeric features...")
        numeric_df, numeric_columns = select_numeric_features(df)
        log(f"Using numeric columns: {numeric_columns}")

        log("Cleaning and scaling numeric data...")
        scaled_data = clean_and_scale(numeric_df)

        log("Running STING clustering...")
        try:
            labels, dense_cell_map, cell_counts, dense_threshold = run_sting(
                scaled_data=scaled_data,
                bins_per_dim=args.bins,
                density_percentile=args.density_percentile,
                min_dense_count=args.min_dense_count,
            )
        except StingClusteringError as first_exc:
            log(f"Primary STING parameters did not produce dense cells: {first_exc}")
            log("Retrying automatically with relaxed parameters...")
            relaxed_bins = max(3, min(args.bins, 4))
            relaxed_percentile = min(args.density_percentile, 50.0)
            relaxed_min_dense = 1
            labels, dense_cell_map, cell_counts, dense_threshold = run_sting(
                scaled_data=scaled_data,
                bins_per_dim=relaxed_bins,
                density_percentile=relaxed_percentile,
                min_dense_count=relaxed_min_dense,
            )
            args.bins = relaxed_bins
            args.density_percentile = relaxed_percentile
            args.min_dense_count = relaxed_min_dense

        log("Computing PCA projection for graphics...")
        pca_2d, pca_model = project_pca_2d(scaled_data)

        result = StingResult(
            labels=labels,
            dense_cell_map=dense_cell_map,
            cell_counts=cell_counts,
            scaled_data=scaled_data,
            numeric_columns=numeric_columns,
            pca_2d=pca_2d,
            pca_model=pca_model,
            dense_threshold=dense_threshold,
            bins_per_dim=args.bins,
        )

        log("Saving CSV outputs...")
        labeled_path, summary_path = save_outputs(df, result, output_dir)

        log("Saving graphics...")
        plot_paths = [
            plot_cluster_scatter(result, output_dir),
            plot_cluster_bar(result.labels, output_dir),
            plot_dense_grid(result, output_dir),
        ]

        summary = summarize_clusters(result.labels)
        n_clusters = int((summary["cluster_label"] >= 0).sum())
        n_noise = int(summary.loc[summary["cluster_label"] == -1, "row_count"].sum()) if (-1 in summary["cluster_label"].values) else 0

        print("STING clustering completed successfully.")
        print(f"Input file: {input_path}")
        print(f"Numeric columns used: {', '.join(numeric_columns)}")
        print(f"Bins per dimension: {args.bins}")
        print(f"Dense cell threshold used: {dense_threshold:.3f}")
        print(f"Detected clusters (excluding noise): {n_clusters}")
        print(f"Noise rows: {n_noise}")
        print(f"Labeled CSV: {labeled_path}")
        print(f"Summary CSV: {summary_path}")
        print("Plots:")
        for path in plot_paths:
            if path:
                print(f"  - {path}")

        return 0

    except StingClusteringError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Operation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("UNEXPECTED ERROR: An unhandled exception occurred.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("\nTraceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
