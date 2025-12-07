#!/usr/bin/env python3
"""
kmeans_daily_prices.py
This uses the dataset from https://www.kaggle.com/datasets/joebeachcapital/natural-gas-prices
A simple, teaching-focused script that:
- Loads a CSV file with daily price data.
- Runs K-means clustering on the Price column.
- Adds the cluster labels back to the data.
- Saves the result as a new CSV file.
- Optionally shows a basic plot of the clusters.

Designed for students:
- Lots of comments.
- Basic error handling.
- Clear structure and small functions.

USAGE (from the terminal/command prompt):

    python kmeans_daily_prices.py --input daily_csv.csv --clusters 3

OPTIONS:
    --input    Path to input CSV file (default: daily_csv.csv)
    --output   Path to output CSV file (default: clustered_output.csv)
    --clusters Number of clusters K (default: 3)
    --no-plot  Disable plotting (useful on servers without a GUI)
"""

import sys
import argparse

# We wrap imports that might not exist on a student's machine in try/except.
# This allows us to give a friendly error instead of a mysterious crash.
try:
    import pandas as pd
except ImportError as e:
    print("ERROR: pandas is required to run this script.")
    print("Install it with: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    print("ERROR: numpy is required to run this script.")
    print("Install it with: pip install numpy")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
except ImportError as e:
    print("ERROR: scikit-learn is required to run this script.")
    print("Install it with: pip install scikit-learn")
    sys.exit(1)

# Matplotlib is optional but nice for visualization.
# If it's missing, we simply disable plotting gracefully.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        args: An argparse.Namespace object with attributes:
              - input
              - output
              - clusters
              - no_plot
    """
    parser = argparse.ArgumentParser(
        description="K-means clustering on daily price data."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="daily_csv.csv",
        help="Path to input CSV file (default: daily_csv.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="clustered_output.csv",
        help="Path to output CSV file (default: clustered_output.csv)",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of clusters K for K-means (default: 3)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of clusters (useful on headless systems).",
    )

    args = parser.parse_args()
    return args


def load_data(csv_path):
    """
    Load the CSV file using pandas with basic error handling.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        SystemExit: If the file can't be loaded or is invalid.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: The file '{csv_path}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"ERROR: The file '{csv_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ERROR: Could not parse '{csv_path}': {e}")
        sys.exit(1)
    except Exception as e:
        # Catch-all in case something unexpected happens
        print(f"ERROR: An unexpected error occurred while reading '{csv_path}': {e}")
        sys.exit(1)

    # Basic validation: check that the Price column exists.
    if "Price" not in df.columns:
        print("ERROR: The input file must contain a 'Price' column.")
        print("Columns found:", list(df.columns))
        sys.exit(1)

    return df


def prepare_features(df):
    """
    Prepare the feature matrix for K-means.

    In this simple example, we only use the Price column (1-D clustering).
    We also handle missing or non-numeric values.

    Args:
        df (pd.DataFrame): Original DataFrame with at least a 'Price' column.

    Returns:
        X (np.ndarray): 2D numpy array with shape (n_samples, 1)
                        suitable for scikit-learn's KMeans.
        clean_df (pd.DataFrame): DataFrame with invalid rows removed.
    """
    # Copy to avoid modifying the original DataFrame.
    clean_df = df.copy()

    # Convert the Price column to numeric explicitly, coercing errors to NaN.
    clean_df["Price"] = pd.to_numeric(clean_df["Price"], errors="coerce")

    # Drop rows where Price is NaN (missing or invalid).
    before_rows = len(clean_df)
    clean_df = clean_df.dropna(subset=["Price"])
    after_rows = len(clean_df)

    if after_rows == 0:
        print("ERROR: No valid numeric values found in 'Price' column.")
        sys.exit(1)

    # If some rows were dropped, warn the user.
    dropped = before_rows - after_rows
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} row(s) with invalid or missing 'Price' values.")

    # KMeans expects a 2D array: (n_samples, n_features).
    # Here we have only one feature: Price.
    X = clean_df[["Price"]].values  # shape (n_samples, 1)

    return X, clean_df


def run_kmeans(X, n_clusters):
    """
    Run K-means clustering on the feature matrix X.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to form.

    Returns:
        labels (np.ndarray): Cluster label for each sample.
        centers (np.ndarray): Coordinates of cluster centers.
    """
    # Basic validation: number of clusters must be positive and <= number of samples.
    n_samples = X.shape[0]
    if n_clusters <= 0:
        print("ERROR: Number of clusters must be a positive integer.")
        sys.exit(1)
    if n_clusters > n_samples:
        print(
            f"ERROR: Number of clusters ({n_clusters}) "
            f"cannot exceed number of data points ({n_samples})."
        )
        sys.exit(1)

    # Create the KMeans object.
    # random_state is set for reproducible results.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"  # lets sklearn choose a sensible number of initializations
    )

    # Fit the model and get cluster labels.
    # This is where the actual K-means algorithm runs.
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers


def save_results(df, labels, output_path):
    """
    Save the DataFrame with cluster labels to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame that corresponds to the labels.
        labels (np.ndarray): Cluster labels for each row in df.
        output_path (str): Path to the output CSV file.
    """
    # Add a new column 'Cluster' with the labels.
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    try:
        df_with_clusters.to_csv(output_path, index=False)
        print(f"Clustered data saved to: {output_path}")
    except PermissionError:
        print(f"ERROR: Permission denied when writing to '{output_path}'.")
        print("Close the file if it's open in another program and try again.")
    except Exception as e:
        print(f"ERROR: Could not save results to '{output_path}': {e}")


def print_cluster_summary(labels, centers):
    """
    Print a simple text summary of the clusters.

    Args:
        labels (np.ndarray): Cluster labels for each sample.
        centers (np.ndarray): Cluster centers.
    """
    print("\n===== CLUSTER SUMMARY =====")
    n_clusters = centers.shape[0]
    for cluster_id in range(n_clusters):
        # Count how many points are in this cluster.
        count = (labels == cluster_id).sum()
        center_value = centers[cluster_id, 0]  # since we have only one feature: Price
        print(
            f"Cluster {cluster_id}: "
            f"center Price ≈ {center_value:.2f}, "
            f"number of points = {count}"
        )
    print("===========================\n")


def plot_clusters(df, labels, centers):
    """
    Create a simple scatter plot of index vs Price, colored by cluster.

    If matplotlib is not available or plotting is disabled, this function
    should not be called.

    Args:
        df (pd.DataFrame): DataFrame with at least the 'Price' column.
        labels (np.ndarray): Cluster labels for each row in df.
        centers (np.ndarray): Cluster centers (used to draw horizontal lines).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Plotting is not available because matplotlib is not installed.")
        return

    # x-axis: simple integer index (0, 1, 2, ...)
    x = np.arange(len(df))
    y = df["Price"].values

    plt.figure()
    # Scatter plot: each point has a color corresponding to its cluster label.
    scatter = plt.scatter(x, y, c=labels, alpha=0.7)

    # Draw horizontal lines at the cluster centers for visual reference.
    for center in centers:
        plt.axhline(y=center[0], linestyle="--", linewidth=1)

    plt.title("K-means Clusters on Daily Prices")
    plt.xlabel("Data Point Index")
    plt.ylabel("Price")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()


def main():
    """
    The main driver function:
    - Parse arguments.
    - Load data.
    - Prepare features.
    - Run K-means.
    - Print summary.
    - Save results.
    - Plot (optional).
    """
    args = parse_arguments()

    # Load the data
    df = load_data(args.input)

    # Prepare the features for K-means (just the 'Price' column here)
    X, clean_df = prepare_features(df)

    # Run K-means
    labels, centers = run_kmeans(X, args.clusters)

    # Print a summary so the student sees something immediately.
    print_cluster_summary(labels, centers)

    # Save results to a new CSV
    save_results(clean_df, labels, args.output)

    # Plot the clusters, unless disabled by flag or matplotlib is missing.
    if not args.no_plot and MATPLOTLIB_AVAILABLE:
        plot_clusters(clean_df, labels, centers)
    elif not MATPLOTLIB_AVAILABLE:
        print("NOTE: Skipping plot because matplotlib is not installed.")
    else:
        print("Plotting disabled by --no-plot flag.")


# This is the Pythonic way to ensure that main() only runs when the script
# is executed directly, not when imported as a module.
if __name__ == "__main__":
    main()
