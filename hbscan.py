#!/usr/bin/env python3
"""
hdbscan_naturalgas.py
this uses the data from https://www.kaggle.com/datasets/alexandrepetit881234/natural-gas-consumption-by-zip-code
also remember pip install hdbscan
A teaching-focused script that:
- Loads a CSV file with natural gas consumption by ZIP code.
- Uses HDBSCAN (Hierarchical DBSCAN) to cluster rows based on
  two numeric features:
      * "Consumption (therms)"
      * "Consumption (GJ)"
- Writes the cluster labels (and probabilities) back to a new CSV file.
- Optionally creates a scatter plot of the clusters.

This script is designed for students:
- Lots of comments.
- Basic error handling.
- Simple structure.

USAGE (from the terminal/command prompt):

    python hdbscan_naturalgas.py --input naturalgasbyzip.csv --min-cluster-size 15

OPTIONS:
    --input             Path to input CSV file (default: naturalgasbyzip.csv)
    --output            Path to output CSV file (default: clustered_naturalgas.csv)
    --min-cluster-size  Minimum cluster size for HDBSCAN (default: 10)
    --min-samples       min_samples parameter for HDBSCAN (default: same as min_cluster_size)
    --no-plot           Disable plotting (useful on servers without a GUI)

NOTE:
    HDBSCAN requires the 'hdbscan' package:
        pip install hdbscan

    This script also uses:
        pandas, numpy, matplotlib (for plotting, optional).
"""

import sys
import argparse

# Try importing required libraries with friendly error messages.
try:
    import pandas as pd
except ImportError:
    print("ERROR: The 'pandas' package is required to run this script.")
    print("Install it with: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: The 'numpy' package is required to run this script.")
    print("Install it with: pip install numpy")
    sys.exit(1)

# HDBSCAN is not part of scikit-learn; it's its own package.
try:
    import hdbscan
except ImportError:
    print("ERROR: The 'hdbscan' package is required for this script.")
    print("Install it with: pip install hdbscan")
    sys.exit(1)

# Matplotlib is only needed for plotting. If it is missing, we'll just skip plots.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace with attributes:
            input, output, min_cluster_size, min_samples, no_plot
    """
    parser = argparse.ArgumentParser(
        description="HDBSCAN clustering on natural gas consumption data."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="naturalgasbyzip.csv",
        help="Path to input CSV file (default: naturalgasbyzip.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="clustered_naturalgas.csv",
        help="Path to output CSV file (default: clustered_naturalgas.csv)",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum cluster size for HDBSCAN (default: 10)",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=(
            "min_samples parameter for HDBSCAN. "
            "If not provided, defaults to min_cluster_size."
        ),
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

    Exits the program with a message if something goes wrong.
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
        # Catch-all for unexpected issues.
        print(f"ERROR: An unexpected error occurred while reading '{csv_path}': {e}")
        sys.exit(1)

    # Check that the expected columns are present.
    required_columns = ["Consumption (therms)", "Consumption (GJ)"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("ERROR: The input file is missing required column(s):")
        for col in missing:
            print(f"   - {col}")
        print("Columns found in file:", list(df.columns))
        sys.exit(1)

    return df


def prepare_features(df):
    """
    Prepare the feature matrix for HDBSCAN.

    For this example, we use two numeric features:
        - 'Consumption (therms)'
        - 'Consumption (GJ)'

    Steps:
        - Convert those columns to numeric.
        - Drop rows with invalid (NaN) values.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        X (np.ndarray): 2D array of shape (n_samples, 2)
                        that HDBSCAN can use.
        clean_df (pd.DataFrame): DataFrame after cleaning.
    """
    clean_df = df.copy()

    # Ensure both columns are numeric. Non-numeric values become NaN.
    clean_df["Consumption (therms)"] = pd.to_numeric(
        clean_df["Consumption (therms)"], errors="coerce"
    )
    clean_df["Consumption (GJ)"] = pd.to_numeric(
        clean_df["Consumption (GJ)"], errors="coerce"
    )

    # Drop rows where either feature is NaN.
    before_rows = len(clean_df)
    clean_df = clean_df.dropna(subset=["Consumption (therms)", "Consumption (GJ)"])
    after_rows = len(clean_df)

    if after_rows == 0:
        print(
            "ERROR: No valid numeric values found in "
            "'Consumption (therms)' and 'Consumption (GJ)' columns."
        )
        sys.exit(1)

    dropped = before_rows - after_rows
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} row(s) with invalid or missing values.")

    # Build the feature matrix X: shape (n_samples, 2).
    X = clean_df[["Consumption (therms)", "Consumption (GJ)"]].values

    return X, clean_df


def run_hdbscan(X, min_cluster_size, min_samples=None):
    """
    Run HDBSCAN clustering on the feature matrix X.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        min_cluster_size (int): Minimum size of clusters.
        min_samples (int or None): min_samples parameter for HDBSCAN.

    Returns:
        labels (np.ndarray): Cluster label for each sample.
                             -1 indicates noise (no cluster).
        probabilities (np.ndarray): Strength of membership for each point.
        clusterer (hdbscan.HDBSCAN): The fitted HDBSCAN object.
    """
    # Basic validation of parameters.
    if min_cluster_size <= 0:
        print("ERROR: min_cluster_size must be a positive integer.")
        sys.exit(1)

    # If min_samples is not provided, default to min_cluster_size
    if min_samples is None:
        min_samples = min_cluster_size

    if min_samples <= 0:
        print("ERROR: min_samples must be a positive integer.")
        sys.exit(1)

    # Create the HDBSCAN clusterer object.
    # Note: metric='euclidean' is a common default.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )

    try:
        # fit_predict returns cluster labels for each row in X.
        labels = clusterer.fit_predict(X)
    except ValueError as e:
        # This might happen if there are not enough points for clustering, etc.
        print(f"ERROR: HDBSCAN failed to run: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during HDBSCAN clustering: {e}")
        sys.exit(1)

    # clusterer.probabilities_ gives the "strength" of each point's cluster assignment.
    probabilities = clusterer.probabilities_

    return labels, probabilities, clusterer


def save_results(df, labels, probabilities, output_path):
    """
    Save the DataFrame with cluster results to a CSV file.

    Args:
        df (pd.DataFrame): Cleaned DataFrame that corresponds to labels.
        labels (np.ndarray): Cluster labels for each row in df.
        probabilities (np.ndarray): Membership strength for each row.
        output_path (str): Path to the output CSV file.
    """
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels
    df_with_clusters["Cluster_Probability"] = probabilities

    try:
        df_with_clusters.to_csv(output_path, index=False)
        print(f"Clustered data saved to: {output_path}")
    except PermissionError:
        print(f"ERROR: Permission denied when writing to '{output_path}'.")
        print("If the file is open in another program, close it and try again.")
    except Exception as e:
        print(f"ERROR: Could not save results to '{output_path}': {e}")


def print_cluster_summary(labels):
    """
    Print a simple summary of clusters found by HDBSCAN.

    Args:
        labels (np.ndarray): Cluster labels for each data point.
                             -1 indicates noise.
    """
    print("\n===== HDBSCAN CLUSTER SUMMARY =====")
    unique_labels, counts = np.unique(labels, return_counts=True)

    for lbl, cnt in zip(unique_labels, counts):
        if lbl == -1:
            # Label -1 is the conventional "noise" label in HDBSCAN/DBSCAN.
            print(f"Noise points (label = -1): {cnt}")
        else:
            print(f"Cluster {lbl}: {cnt} point(s)")

    print("===================================\n")


def plot_clusters(df, labels):
    """
    Create a scatter plot of 'Consumption (therms)' vs 'Consumption (GJ)'
    colored by cluster label.

    Noise points (label = -1) will be shown as a separate color.

    Args:
        df (pd.DataFrame): DataFrame with at least the two consumption columns.
        labels (np.ndarray): Cluster labels for each row.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Plotting is not available because matplotlib is not installed.")
        return

    # Extract the features for plotting.
    x = df["Consumption (therms)"].values
    y = df["Consumption (GJ)"].values

    # Create a scatter plot where the color (c) is the cluster label.
    plt.figure()
    scatter = plt.scatter(x, y, c=labels, alpha=0.7)

    plt.title("HDBSCAN Clusters: Consumption (therms) vs Consumption (GJ)")
    plt.xlabel("Consumption (therms)")
    plt.ylabel("Consumption (GJ)")
    plt.colorbar(scatter, label="Cluster Label (-1 = Noise)")
    plt.tight_layout()
    plt.show()


def main():
    """
    Main driver function:
    - Parse arguments.
    - Load data.
    - Prepare features.
    - Run HDBSCAN.
    - Print summary.
    - Save results.
    - Plot (optional).
    """
    args = parse_arguments()

    # Load the data from the CSV file.
    df = load_data(args.input)

    # Prepare the features for HDBSCAN (two numeric columns).
    X, clean_df = prepare_features(df)

    # Run HDBSCAN clustering.
    labels, probabilities, clusterer = run_hdbscan(
        X,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Print summary of how many points in each cluster (and noise).
    print_cluster_summary(labels)

    # Save the results to a new CSV file.
    save_results(clean_df, labels, probabilities, args.output)

    # Plot the clusters unless plotting is disabled or matplotlib is missing.
    if not args.no_plot and MATPLOTLIB_AVAILABLE:
        plot_clusters(clean_df, labels)
    elif not MATPLOTLIB_AVAILABLE:
        print("NOTE: Skipping plot because matplotlib is not installed.")
    else:
        print("Plotting disabled by --no-plot flag.")


# Standard Python convention: only run main() when this file is executed directly.
if __name__ == "__main__":
    main()
