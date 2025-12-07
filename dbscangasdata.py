"""
dbscan_example.py
YOU WILL NEED THIS DATASET: https://www.kaggle.com/datasets/alistairking/natural-gas-usage
A teaching example of how to:
 - load tabular data from a CSV file with pandas
 - preprocess it for clustering
 - run DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
 - handle common errors in a friendly way

This script is written for students, so it includes MANY comments.

USAGE (from the command line):
    python dbscan_example.py --file data.csv --eps 0.5 --min_samples 10

If you just run:
    python dbscan_example.py

it will default to:
    file = "data.csv", eps = 0.5, min_samples = 10
"""

import argparse
import sys
from typing import List

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    We allow the user to change:
      - the CSV file path
      - the DBSCAN parameters eps and min_samples

    Returns
    -------
    argparse.Namespace
        An object with attributes: file, eps, min_samples
    """
    parser = argparse.ArgumentParser(
        description="Run DBSCAN clustering on a CSV file."
    )

    parser.add_argument(
        "--file",
        type=str,
        default="natural_gas/data.csv",
        help="Path to the CSV file (default: data.csv).",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help=(
            "DBSCAN eps parameter (radius of neighborhood). "
            "Roughly: smaller eps → more clusters and more noise. (default: 0.5)"
        ),
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help=(
            "DBSCAN min_samples parameter (minimum points in a dense region). "
            "Roughly: larger min_samples → fewer clusters and more noise. (default: 10)"
        ),
    )

    args = parser.parse_args()

    # Basic validation of arguments with clear error messages.
    if args.eps <= 0:
        parser.error("eps must be > 0. You provided: {}".format(args.eps))

    if args.min_samples < 1:
        parser.error("min_samples must be >= 1. You provided: {}".format(args.min_samples))

    return args


def load_csv_safely(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with error handling.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    SystemExit
        If there is a problem reading the file, we print a friendly message
        and exit the program.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: The file '{filepath}' was not found. "
              "Make sure it exists and the path is correct.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"ERROR: The file '{filepath}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ERROR: The file '{filepath}' could not be parsed as CSV.")
        print("Details:", e)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors.
        print(f"ERROR: An unexpected error occurred while reading '{filepath}':")
        print(e)
        sys.exit(1)

    # Extra safety: check that the DataFrame is not empty.
    if df.empty:
        print(f"ERROR: The file '{filepath}' was read, but it contains no rows.")
        sys.exit(1)

    print(f"Successfully loaded '{filepath}' with {len(df)} rows and {len(df.columns)} columns.")
    return df


def select_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Select and validate the feature columns to be used for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    feature_columns : List[str]
        Column names to be used as features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the selected feature columns,
        with rows containing missing values dropped.

    Raises
    ------
    SystemExit
        If any of the requested feature columns do not exist
        or if there is no valid data after dropping NaNs.
    """
    # Check that all requested columns are present.
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        print("ERROR: The following required columns are missing from the CSV file:")
        for col in missing:
            print("  -", col)
        print("Available columns are:", list(df.columns))
        sys.exit(1)

    # Select just the feature columns.
    features = df[feature_columns].copy()

    # Drop rows with missing values in any of the feature columns.
    before_drop = len(features)
    features = features.dropna()
    after_drop = len(features)

    if after_drop == 0:
        print("ERROR: After dropping rows with missing values in feature columns, "
              "no data remains. Check your CSV or choose different feature columns.")
        sys.exit(1)
    elif after_drop < before_drop:
        print(f"NOTE: Dropped {before_drop - after_drop} rows due to missing feature values.")

    # Ensure that all selected columns are numeric.
    # If not, try to convert them or report an error.
    for col in feature_columns:
        if not np.issubdtype(features[col].dtype, np.number):
            print(f"WARNING: Column '{col}' is not numeric (dtype={features[col].dtype}). "
                  "Attempting to convert it to numeric.")
            try:
                features[col] = pd.to_numeric(features[col], errors="coerce")
            except Exception as e:
                print(f"ERROR: Could not convert column '{col}' to numeric.")
                print("Details:", e)
                sys.exit(1)

    # After conversion, drop any rows that became NaN.
    before_drop2 = len(features)
    features = features.dropna()
    after_drop2 = len(features)

    if after_drop2 == 0:
        print("ERROR: After converting to numeric and dropping NaNs, no data remains.")
        sys.exit(1)
    elif after_drop2 < before_drop2:
        print(f"NOTE: Dropped {before_drop2 - after_drop2} rows "
              f"after converting to numeric and removing NaNs.")

    print(f"Using {after_drop2} rows for clustering with features: {feature_columns}")
    return features


def run_dbscan(features: pd.DataFrame, eps: float, min_samples: int) -> np.ndarray:
    """
    Run the DBSCAN clustering algorithm on the given features.

    Parameters
    ----------
    features : pd.DataFrame
        The feature data (numeric).
    eps : float
        Radius of neighborhood (DBSCAN parameter).
    min_samples : int
        Minimum number of points required to form a dense region (DBSCAN parameter).

    Returns
    -------
    np.ndarray
        Cluster labels for each row in `features`.
        -1 indicates "noise" (points not assigned to any cluster).
    """
    # Step 1: Standardize the data.
    # Many clustering algorithms (including DBSCAN) work better if
    # each feature has similar scale. StandardScaler converts each
    # feature to have mean ~0 and standard deviation ~1.
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.values)
    except Exception as e:
        print("ERROR: Failed to scale features using StandardScaler.")
        print("Details:", e)
        sys.exit(1)

    # Step 2: Run DBSCAN.
    # DBSCAN groups together points that are closely packed together,
    # with "eps" controlling how close points need to be to be neighbors,
    # and "min_samples" controlling how many neighbors are required to form a cluster.
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
    except Exception as e:
        print("ERROR: Failed to run DBSCAN clustering.")
        print("Details:", e)
        sys.exit(1)

    # Basic sanity check: we expect one label per row.
    if len(labels) != len(features):
        print("ERROR: Number of cluster labels does not match number of rows in features.")
        print(f"len(labels) = {len(labels)}, len(features) = {len(features)}")
        sys.exit(1)

    return labels


def summarize_clusters(labels: np.ndarray) -> None:
    """
    Print a simple summary of the DBSCAN clustering result.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels returned by DBSCAN.
    """
    # DBSCAN uses label -1 for "noise" points (points that are not in any cluster).
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\n===== DBSCAN CLUSTER SUMMARY =====")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise points (label = -1): {count}")
        else:
            print(f"Cluster {label}: {count} points")
    print("==================================\n")


def plot_clusters(features: pd.DataFrame, labels: np.ndarray, feature_columns: List[str]) -> None:
    """
    Create a basic scatter plot of two features colored by cluster label.

    NOTE: This is just for visualization and teaching purposes.
    We will use the first two feature columns for the x and y axes.

    Parameters
    ----------
    features : pd.DataFrame
        Feature data used for clustering.
    labels : np.ndarray
        Cluster labels from DBSCAN.
    feature_columns : List[str]
        Names of the feature columns (used for axis labels).
    """
    if len(feature_columns) < 2:
        print("Plotting skipped: need at least two features for a 2D scatter plot.")
        return

    x_col = feature_columns[0]
    y_col = feature_columns[1]

    # We add a try/except so plotting can't crash the whole script.
    try:
        plt.figure(figsize=(8, 6))

        # We convert labels to an array so we can index features by cluster.
        labels_array = np.array(labels)

        unique_labels = np.unique(labels_array)
        for label in unique_labels:
            # For each cluster (including noise), plot its points.
            mask = labels_array == label
            if label == -1:
                # Noise points are usually plotted in a different style.
                plt.scatter(
                    features.loc[mask, x_col],
                    features.loc[mask, y_col],
                    label="Noise (-1)",
                    alpha=0.6,
                    marker="x",
                )
            else:
                plt.scatter(
                    features.loc[mask, x_col],
                    features.loc[mask, y_col],
                    label=f"Cluster {label}",
                    alpha=0.6,
                )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("DBSCAN Clusters")
        plt.legend()
        plt.tight_layout()
        print("Displaying cluster scatter plot (close the window to finish).")
        plt.show()

    except Exception as e:
        print("WARNING: Could not create plot. This may happen in headless environments (e.g., some servers).")
        print("Plot error details:", e)


def main():
    """
    Main driver function:
      1. Parse command line arguments.
      2. Load the CSV file safely.
      3. Choose feature columns (for this dataset: year, month, value).
      4. Run DBSCAN.
      5. Summarize and save results.
      6. Optionally plot clusters.
    """
    args = parse_arguments()

    # 1) Load the data from CSV.
    df = load_csv_safely(args.file)

    # 2) Choose which columns to use as features for clustering.
    #
    # For your attached dataset, we use:
    #   - "year"
    #   - "month"
    #   - "value"  (the numeric measurement)
    #
    # You can change this list if you want to experiment with different features,
    # but make sure they are numeric (or convertible to numeric).
    feature_columns = ["year", "month", "value"]

    # 3) Select and validate feature data.
    features = select_features(df, feature_columns)

    # 4) Run DBSCAN clustering.
    labels = run_dbscan(features, eps=args.eps, min_samples=args.min_samples)

    # 5) Print a summary of the clusters.
    summarize_clusters(labels)

    # 6) Attach the labels back to the original DataFrame.
    # We need to be careful with indexing since we dropped some rows during preprocessing.
    # features.index contains the original row indices that are still in use.
    clustered_df = df.copy()
    clustered_df["cluster_label"] = np.nan  # initialize with NaN
    clustered_df.loc[features.index, "cluster_label"] = labels

    # 7) Save the DataFrame with cluster labels to a new CSV.
    output_file = "clustered_output.csv"
    try:
        clustered_df.to_csv(output_file, index=False)
        print(f"Clustered data (with 'cluster_label' column) saved to '{output_file}'.")
    except Exception as e:
        print("WARNING: Could not save clustered results to CSV.")
        print("Save error details:", e)

    # 8) Optional: plot clusters using the first two feature columns.
    plot_clusters(features, labels, feature_columns)


if __name__ == "__main__":
    main()
