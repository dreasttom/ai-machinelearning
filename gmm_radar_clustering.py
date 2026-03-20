#!/usr/bin/env python3
"""
gmm_radar_clustering.py

A heavily commented example script that loads a radar-reading CSV file from the
same folder as the script, preprocesses the numeric radar features, and fits a
Gaussian Mixture Model (GMM) for unsupervised clustering.

Expected input file (default):
  test_military_radar_readings.csv
  This dataset was created for testing purposes and is in the github with the code.

Expected output file (default):
    hypothetical_military_radar_readings_with_gmm_clusters.csv

What this script does:
1. Finds the CSV file in the same directory as the script.
2. Validates that required columns exist.
3. Cleans and converts numeric columns safely.
4. Imputes missing numeric values.
5. Standardizes features so GMM is not biased by scale differences.
6. Fits a Gaussian Mixture Model.
7. Writes cluster assignments and per-cluster probabilities to a new CSV.
8. Prints a concise summary to the console.

This is intended as a robust starting point for experimentation and teaching.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
# These are the numeric radar fields we will use for clustering.
# We intentionally exclude:
# - timestamp: not directly useful for clustering object types in this example
# - target_type: that is a label-like field, and clustering is unsupervised
#
# You can modify this list if your dataset schema changes.
FEATURE_COLUMNS: List[str] = [
    "range_km",
    "velocity_mps",
    "azimuth_deg",
    "elevation_deg",
    "rcs_m2",
    "signal_strength_db",
    "confidence",
]

DEFAULT_INPUT_FILE = "test_military_radar_readings.csv"
DEFAULT_OUTPUT_FILE = "hypothetical_military_radar_readings_with_gmm_clusters.csv"
DEFAULT_COMPONENTS = 6
DEFAULT_RANDOM_STATE = 42
DEFAULT_COVARIANCE_TYPE = "full"
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 5


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    verbose : bool
        If True, emit debug-level logs. Otherwise, info-level logs.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def script_directory() -> Path:
    """Return the directory containing this script.

    Using Path(__file__).resolve().parent ensures the script will look for the
    CSV beside itself, regardless of the current working directory from which
    the user runs the script.
    """
    return Path(__file__).resolve().parent


def validate_input_file(path: Path) -> None:
    """Validate that the input file exists and is a readable CSV.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    PermissionError
        If the file exists but cannot be read.
    ValueError
        If the file extension does not look like a CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, but got: {path.name}")

    # A quick access test. This catches common permission issues early.
    try:
        with path.open("r", encoding="utf-8"):
            pass
    except PermissionError as exc:
        raise PermissionError(f"Input file is not readable: {path}") from exc


def load_csv(path: Path) -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame with robust error handling."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The input CSV is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The input CSV could not be parsed: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"The input CSV could not be decoded as UTF-8: {path}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while loading CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The loaded DataFrame is empty. Nothing to cluster.")

    return df


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Ensure the DataFrame contains the required columns.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            "The following required columns are missing from the input CSV: "
            + ", ".join(missing)
        )


def prepare_feature_matrix(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, Pipeline]:
    """Prepare a clean numeric matrix for GMM.

    Steps:
    1. Copy the relevant columns so we do not mutate the caller's DataFrame.
    2. Coerce values to numeric safely (bad values become NaN).
    3. Impute missing values with the median.
    4. Standardize features so each feature is on a comparable scale.

    Returns
    -------
    feature_df : pd.DataFrame
        The numeric feature subset before scaling.
    transformed_matrix : np.ndarray
        The imputed and standardized matrix ready for GMM.
    preprocessing_pipeline : Pipeline
        The fitted preprocessing pipeline.
    """
    feature_df = df[feature_columns].copy()

    # Convert each feature to numeric. Any problematic value becomes NaN.
    # This is safer than failing immediately on a single malformed cell.
    for col in feature_columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    # If an entire column becomes NaN, the model cannot use it.
    all_nan_columns = [col for col in feature_columns if feature_df[col].isna().all()]
    if all_nan_columns:
        raise ValueError(
            "The following feature columns contain no usable numeric data: "
            + ", ".join(all_nan_columns)
        )

    # Build a preprocessing pipeline.
    # - Median imputation is robust against outliers.
    # - StandardScaler makes features comparable in magnitude.
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        transformed = preprocessing.fit_transform(feature_df)
    except Exception as exc:
        raise RuntimeError(f"Failed during preprocessing: {exc}") from exc

    if transformed.shape[0] < 2:
        raise ValueError("At least 2 rows are required for clustering.")

    return feature_df, transformed, preprocessing


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    covariance_type: str,
    max_iter: int,
    n_init: int,
    random_state: int,
) -> GaussianMixture:
    """Fit a Gaussian Mixture Model with strong validation and warnings capture."""
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")
    if n_components > X.shape[0]:
        raise ValueError(
            "n_components cannot exceed the number of samples. "
            f"Got n_components={n_components}, samples={X.shape[0]}."
        )

    valid_covariance_types = {"full", "tied", "diag", "spherical"}
    if covariance_type not in valid_covariance_types:
        raise ValueError(
            "Invalid covariance_type. Choose one of: "
            + ", ".join(sorted(valid_covariance_types))
        )

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        reg_covar=1e-6,  # helps numerical stability in some edge cases
    )

    # Capture convergence warnings so the user gets explicit feedback.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            model.fit(X)
        except ValueError as exc:
            raise ValueError(f"GMM fitting failed due to invalid data: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected failure during GMM fitting: {exc}") from exc

        for warning in caught_warnings:
            if issubclass(warning.category, ConvergenceWarning):
                logging.warning(
                    "GMM emitted a convergence warning: %s. "
                    "You may want to increase max_iter, adjust n_components, "
                    "or inspect the data quality.",
                    warning.message,
                )

    return model


def attach_results(
    df: pd.DataFrame,
    model: GaussianMixture,
    X: np.ndarray,
) -> pd.DataFrame:
    """Attach cluster labels, probabilities, and scores to the original DataFrame."""
    result_df = df.copy()

    try:
        labels = model.predict(X)
        probabilities = model.predict_proba(X)
        log_likelihood = model.score_samples(X)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate GMM predictions: {exc}") from exc

    result_df["gmm_cluster"] = labels
    result_df["gmm_log_likelihood"] = log_likelihood

    # Add one column per component probability.
    for idx in range(probabilities.shape[1]):
        result_df[f"gmm_prob_cluster_{idx}"] = probabilities[:, idx]

    return result_df


def save_output(df: pd.DataFrame, path: Path) -> None:
    """Save the output DataFrame with careful error handling."""
    try:
        df.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(f"Cannot write output file: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to save output CSV: {exc}") from exc


def print_summary(
    result_df: pd.DataFrame,
    model: GaussianMixture,
    feature_columns: List[str],
) -> None:
    """Print a human-readable summary of the fitted clustering model."""
    print("\n=== Gaussian Mixture Model Summary ===")
    print(f"Samples clustered      : {len(result_df)}")
    print(f"Features used          : {', '.join(feature_columns)}")
    print(f"Number of clusters     : {model.n_components}")
    print(f"Covariance type        : {model.covariance_type}")
    print(f"Model converged        : {model.converged_}")
    print(f"Iterations used        : {model.n_iter_}")
    print(f"Lower bound            : {model.lower_bound_:.6f}")
    print()

    cluster_counts = result_df["gmm_cluster"].value_counts().sort_index()
    print("Cluster counts:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} rows")


# -----------------------------------------------------------------------------
# Main program flow
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The defaults assume the CSV is in the same directory as the script.
    Users can override the filename or model settings if they want.
    """
    parser = argparse.ArgumentParser(
        description="Fit a Gaussian Mixture Model (GMM) to radar CSV data."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=(
            "Name of the input CSV file in the same folder as the script. "
            f"Default: {DEFAULT_INPUT_FILE}"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Name of the output CSV file to write in the same folder as the script. "
            f"Default: {DEFAULT_OUTPUT_FILE}"
        ),
    )
    parser.add_argument(
        "--components",
        type=int,
        default=DEFAULT_COMPONENTS,
        help=f"Number of Gaussian components/clusters. Default: {DEFAULT_COMPONENTS}",
    )
    parser.add_argument(
        "--covariance-type",
        default=DEFAULT_COVARIANCE_TYPE,
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for GaussianMixture. Default: full",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum GMM iterations. Default: {DEFAULT_MAX_ITER}",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
        help=f"Number of random initializations. Default: {DEFAULT_N_INIT}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility. Default: {DEFAULT_RANDOM_STATE}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Process exit code. 0 means success; non-zero indicates failure.
    """
    args = parse_args()
    configure_logging(verbose=args.verbose)

    try:
        base_dir = script_directory()
        input_path = base_dir / args.input
        output_path = base_dir / args.output

        logging.info("Script directory: %s", base_dir)
        logging.info("Input file: %s", input_path)
        logging.info("Output file: %s", output_path)

        validate_input_file(input_path)
        df = load_csv(input_path)
        logging.info("Loaded %d rows and %d columns.", df.shape[0], df.shape[1])

        validate_columns(df, FEATURE_COLUMNS)
        logging.info("Validated required feature columns.")

        _, X, _ = prepare_feature_matrix(df, FEATURE_COLUMNS)
        logging.info("Prepared feature matrix with shape %s.", X.shape)

        model = fit_gmm(
            X=X,
            n_components=args.components,
            covariance_type=args.covariance_type,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        logging.info("Successfully fit Gaussian Mixture Model.")

        result_df = attach_results(df, model, X)
        save_output(result_df, output_path)
        logging.info("Saved clustered output to %s", output_path)

        print_summary(result_df, model, FEATURE_COLUMNS)
        print(f"\nOutput written to: {output_path}")
        return 0

    except FileNotFoundError as exc:
        logging.error("File error: %s", exc)
    except PermissionError as exc:
        logging.error("Permission error: %s", exc)
    except KeyError as exc:
        logging.error("Column validation error: %s", exc)
    except ValueError as exc:
        logging.error("Value/data error: %s", exc)
    except RuntimeError as exc:
        logging.error("Runtime error: %s", exc)
    except Exception as exc:
        # Catch-all to avoid unhandled crashes and provide a useful message.
        logging.exception("Unexpected unhandled error: %s", exc)

    return 1


if __name__ == "__main__":
    sys.exit(main())
