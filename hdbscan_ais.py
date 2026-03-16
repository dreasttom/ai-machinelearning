#!/usr/bin/env python3
"""
HDBSCAN clustering for AIS_2023_Set_One.csv

This script:
1. Loads AIS data from a CSV file.
2. Performs careful validation and preprocessing.
3. Selects numeric features and engineers time-based features.
4. Runs HDBSCAN clustering.
5. Saves a clustered CSV plus several graphical outputs.

Design goals:
- Heavy inline comments for readability and maintenance.
- Robust error handling with clear, actionable messages.
- Reasonable defaults: if no arguments are supplied, the script looks for
  AIS_2023_Set_One.csv in the same folder as the script and writes outputs there. The data is a subset of
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/index.html
  This is a huge dataset and runs slow
  

Typical usage:
    python hdbscan_ais.py
or:
    python hdbscan_ais.py --input path/to/AIS_2023_Set_One.csv --output-dir results

Dependencies:
    pandas, numpy, matplotlib, scikit-learn, hdbscan

Install example:
    pip install pandas numpy matplotlib scikit-learn hdbscan
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def eprint(*args, **kwargs) -> None:
    """Print to stderr so error and status messages are easier to distinguish."""
    print(*args, file=sys.stderr, **kwargs)


def safe_mkdir(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Raises:
        RuntimeError: If the directory cannot be created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to create output directory: {path}") from exc


def validate_input_file(path: Path) -> None:
    """
    Verify the input file exists and is a CSV-like file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}\n"
            f"Tip: If the CSV is in the same folder as the script, run the script with no arguments."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV file, but got: {path.suffix}")


def import_required_packages():
    """
    Import optional runtime dependencies here so we can emit friendly messages
    if something is missing instead of failing with a cryptic ImportError.

    Returns:
        tuple: (SimpleImputer, StandardScaler, PCA, HDBSCAN)
    """
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required but could not be imported.\n"
            "Install it with: pip install scikit-learn"
        ) from exc

    try:
        import hdbscan
    except Exception as exc:
        raise ImportError(
            "The 'hdbscan' package is required but could not be imported.\n"
            "Install it with: pip install hdbscan"
        ) from exc

    return SimpleImputer, StandardScaler, PCA, hdbscan.HDBSCAN


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the AIS CSV with guarded error handling.

    Raises:
        RuntimeError: For parsing or decoding problems.
        ValueError: If the file is empty.
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise RuntimeError(
            f"Failed to parse CSV file: {csv_path}\n"
            f"The file may be malformed or use an unexpected delimiter."
        ) from exc
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Failed to decode CSV file: {csv_path}\n"
            f"Try saving the file as UTF-8."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while reading CSV: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"The CSV loaded successfully but contains no rows: {csv_path}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based features from BaseDateTime if available.

    We transform time into cyclic features because hour-of-day and day-of-year
    are circular variables: 23:00 and 00:00 are close together, not far apart.

    Returns:
        pd.DataFrame: Copy of the dataframe with extra engineered features.
    """
    df = df.copy()

    if "BaseDateTime" not in df.columns:
        return df

    # Convert timestamps carefully; invalid entries become NaT instead of crashing.
    dt = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=False)

    # Add raw numeric time descriptors.
    df["Hour"] = dt.dt.hour
    df["DayOfYear"] = dt.dt.dayofyear
    df["Month"] = dt.dt.month
    df["DayOfWeek"] = dt.dt.dayofweek

    # Add cyclical encodings for more cluster-friendly geometry.
    if df["Hour"].notna().any():
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

    if df["DayOfYear"].notna().any():
        df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
        df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and prepare numeric features for clustering.

    Strategy:
    - Engineer time features.
    - Keep numeric columns only.
    - Drop obviously identifier-like columns if present and too unique, because
      unique IDs tend to harm unsupervised clustering.
    - Preserve interpretable maritime movement / vessel dimensions where possible.

    Returns:
        tuple:
            feature_df: numeric dataframe ready for imputation/scaling
            feature_names: list of selected feature names
    """
    df = add_time_features(df)

    # Keep numeric columns only for HDBSCAN input.
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns were found after preprocessing. "
            "HDBSCAN in this script expects numeric features."
        )

    # Candidate identifier columns that often represent unique IDs rather than
    # meaningful geometric relationships for clustering.
    possible_id_cols = ["MMSI", "IMO"]

    for col in possible_id_cols:
        if col in numeric_df.columns:
            unique_ratio = numeric_df[col].nunique(dropna=True) / max(len(numeric_df), 1)
            # If nearly every row has a unique ID, it is usually better to remove it.
            if unique_ratio > 0.90:
                numeric_df.drop(columns=[col], inplace=True)

    # Remove columns that contain all missing values.
    all_missing_cols = [c for c in numeric_df.columns if numeric_df[c].isna().all()]
    if all_missing_cols:
        numeric_df.drop(columns=all_missing_cols, inplace=True)

    if numeric_df.empty:
        raise ValueError(
            "All numeric columns were removed during preprocessing. "
            "Please inspect the CSV contents."
        )

    # Remove zero-variance columns because they add no information and can
    # sometimes cause numerical instability.
    zero_var_cols = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        numeric_df.drop(columns=zero_var_cols, inplace=True)

    if numeric_df.empty:
        raise ValueError(
            "No usable numeric columns remain after dropping constant columns."
        )

    return numeric_df, list(numeric_df.columns)


def preprocess_features(feature_df: pd.DataFrame):
    """
    Impute missing values and standardize features.

    Returns:
        tuple:
            X_scaled (ndarray): matrix for clustering
            imputer, scaler: fitted preprocessing objects
    """
    SimpleImputer, StandardScaler, _, _ = import_required_packages()

    # Median imputation is robust for skewed real-world maritime data.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    try:
        X_imputed = imputer.fit_transform(feature_df)
        X_scaled = scaler.fit_transform(X_imputed)
    except Exception as exc:
        raise RuntimeError("Failed during feature imputation/scaling.") from exc

    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite values remain after preprocessing.")

    return X_scaled, imputer, scaler


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def run_hdbscan(
    X_scaled: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
):
    """
    Fit HDBSCAN and return the trained clusterer.

    HDBSCAN is robust for variable-density clustering and naturally identifies noise.
    """
    _, _, _, HDBSCAN = import_required_packages()

    try:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
        )
        clusterer.fit(X_scaled)
    except Exception as exc:
        raise RuntimeError("HDBSCAN fitting failed.") from exc

    return clusterer


def summarize_clusters(labels: np.ndarray) -> pd.Series:
    """
    Build a clean cluster-size summary, including noise if present (-1 label).
    """
    summary = pd.Series(labels).value_counts(dropna=False).sort_index()
    summary.index = [f"Cluster {i}" if i != -1 else "Noise" for i in summary.index]
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_missingness_plot(feature_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a bar chart showing missingness percentage by feature before imputation.
    """
    missing_pct = feature_df.isna().mean().sort_values(ascending=False) * 100

    plt.figure(figsize=(12, 6))
    missing_pct.plot(kind="bar")
    plt.ylabel("Missing values (%)")
    plt.title("Feature Missingness Before Imputation")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_by_feature.png", dpi=150)
    plt.close()


def save_cluster_count_plot(labels: np.ndarray, output_dir: Path) -> None:
    """
    Save a bar chart of cluster membership counts, including noise.
    """
    summary = summarize_clusters(labels)

    plt.figure(figsize=(10, 6))
    summary.plot(kind="bar")
    plt.ylabel("Number of records")
    plt.title("HDBSCAN Cluster Sizes")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=150)
    plt.close()


def save_probability_histogram(probabilities: np.ndarray, output_dir: Path) -> None:
    """
    Save a histogram of HDBSCAN membership probabilities.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=30)
    plt.xlabel("Membership Probability")
    plt.ylabel("Frequency")
    plt.title("HDBSCAN Membership Probability Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "membership_probabilities.png", dpi=150)
    plt.close()


def save_outlier_histogram(outlier_scores: np.ndarray, output_dir: Path) -> None:
    """
    Save a histogram of HDBSCAN outlier scores.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(outlier_scores, bins=30)
    plt.xlabel("Outlier Score")
    plt.ylabel("Frequency")
    plt.title("HDBSCAN Outlier Score Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "outlier_scores.png", dpi=150)
    plt.close()


def save_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """
    Project high-dimensional features to 2D via PCA and color points by cluster label.
    """
    _, _, PCA, _ = import_required_packages()

    try:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
    except Exception as exc:
        raise RuntimeError("Failed to compute PCA projection for visualization.") from exc

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=12, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection Colored by HDBSCAN Cluster")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_clusters.png", dpi=150)
    plt.close()


def save_geo_scatter(df: pd.DataFrame, labels: np.ndarray, output_dir: Path) -> None:
    """
    If latitude/longitude exist, create a geographic scatter plot colored by cluster.
    """
    if "LAT" not in df.columns or "LON" not in df.columns:
        return

    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    mask = lat.notna() & lon.notna()

    if mask.sum() == 0:
        return

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(lon[mask], lat[mask], c=np.asarray(labels)[mask], s=12, alpha=0.8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Geographic Distribution Colored by HDBSCAN Cluster")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.savefig(output_dir / "geographic_clusters.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Reporting and persistence
# ---------------------------------------------------------------------------

def save_outputs(
    original_df: pd.DataFrame,
    feature_names: List[str],
    clusterer,
    output_dir: Path,
) -> None:
    """
    Save the clustered dataset and a small text summary.
    """
    output_df = original_df.copy()
    output_df["HDBSCAN_Cluster"] = clusterer.labels_
    output_df["HDBSCAN_Probability"] = clusterer.probabilities_
    output_df["HDBSCAN_OutlierScore"] = clusterer.outlier_scores_

    try:
        output_df.to_csv(output_dir / "ais_hdbscan_clustered.csv", index=False)
    except Exception as exc:
        raise RuntimeError("Failed to save clustered CSV output.") from exc

    noise_count = int(np.sum(clusterer.labels_ == -1))
    non_noise_labels = sorted(set(clusterer.labels_) - {-1})
    cluster_count = len(non_noise_labels)

    summary_lines = [
        "HDBSCAN AIS Clustering Summary",
        "=============================",
        f"Rows processed: {len(output_df)}",
        f"Features used: {len(feature_names)}",
        f"Feature list: {', '.join(feature_names)}",
        f"Clusters found (excluding noise): {cluster_count}",
        f"Noise points: {noise_count}",
        "",
        "Cluster membership counts:",
        summarize_clusters(clusterer.labels_).to_string(),
    ]

    try:
        (output_dir / "hdbscan_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("Failed to save summary text file.") from exc


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments with script-directory defaults.
    """
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "AIS_2023_Set_One.csv"
    default_output = script_dir

    parser = argparse.ArgumentParser(
        description="Run HDBSCAN clustering on AIS data with robust preprocessing and plotting."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="Path to input CSV file. Default: AIS_2023_Set_One.csv in the script directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output),
        help="Directory for output files. Default: the script directory.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="Minimum cluster size for HDBSCAN. Default: 100",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Optional HDBSCAN min_samples. Default: None (library default behavior)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default="eom",
        help="HDBSCAN cluster selection method. Default: eom",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main program entry point.

    Returns:
        int: process exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    # Show the exact paths being used. This is very helpful when debugging path issues.
    print(f"Input CSV:  {input_path}")
    print(f"Output dir: {output_dir}")

    try:
        validate_input_file(input_path)
        safe_mkdir(output_dir)

        # Load original data.
        df = load_csv(input_path)

        # Build modeling matrix.
        feature_df, feature_names = build_feature_matrix(df)
        X_scaled, _, _ = preprocess_features(feature_df)

        # Fit HDBSCAN.
        clusterer = run_hdbscan(
            X_scaled=X_scaled,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method,
        )

        # Persist tabular outputs first.
        save_outputs(
            original_df=df,
            feature_names=feature_names,
            clusterer=clusterer,
            output_dir=output_dir,
        )

        # Generate graphical outputs.
        save_missingness_plot(feature_df, output_dir)
        save_cluster_count_plot(clusterer.labels_, output_dir)
        save_probability_histogram(clusterer.probabilities_, output_dir)
        save_outlier_histogram(clusterer.outlier_scores_, output_dir)
        save_pca_scatter(X_scaled, clusterer.labels_, output_dir)
        save_geo_scatter(df, clusterer.labels_, output_dir)

        # Console summary for convenience.
        summary = summarize_clusters(clusterer.labels_)
        non_noise_cluster_count = len(set(clusterer.labels_) - {-1})
        print("\nHDBSCAN completed successfully.")
        print(f"Rows processed: {len(df)}")
        print(f"Features used: {len(feature_names)}")
        print(f"Clusters found (excluding noise): {non_noise_cluster_count}")
        print("\nCluster counts:")
        print(summary.to_string())
        print(f"\nSaved outputs to: {output_dir.resolve()}")

        return 0

    except KeyboardInterrupt:
        eprint("\nExecution interrupted by user.")
        return 130

    except Exception as exc:
        # Print a concise error message first.
        eprint(f"\nERROR: {exc}")

        # Then print a short traceback to help advanced users debug.
        eprint("\nDetailed traceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
