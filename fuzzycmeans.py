"""
Fuzzy C-Means Clustering Example on Natural Gas Consumption Data
This uses the data from https://www.kaggle.com/datasets/alexandrepetit881234/natural-gas-consumption-by-zip-code
This script is meant as a TEACHING AID for students.

It demonstrates:
- Loading and inspecting a CSV dataset
- Basic data cleaning and feature selection
- Scaling numeric features
- Applying Fuzzy C-Means clustering (via scikit-fuzzy)
- Visualizing the resulting clusters and membership levels
- Using error handling and informative messages

REQUIREMENTS (install via pip if needed):
    pip install numpy pandas matplotlib scikit-fuzzy scikit-learn
"""

import os
import sys

# --- Error handling for missing third-party packages -------------------------
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed. Please install it with 'pip install numpy'.")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed. Please install it with 'pip install pandas'.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed. Please install it with 'pip install matplotlib'.")
    sys.exit(1)

try:
    import skfuzzy as fuzz
except ImportError:
    print(
        "ERROR: scikit-fuzzy (skfuzzy) is not installed.\n"
        "Install it with: pip install scikit-fuzzy"
    )
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print(
        "ERROR: scikit-learn is not installed.\n"
        "Install it with: pip install scikit-learn"
    )
    sys.exit(1)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the natural gas dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    ValueError:
        If the file exists but is empty or not a valid CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"The file '{csv_path}' is empty.") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"The file '{csv_path}' is not a valid CSV.") from e

    if df.empty:
        raise ValueError(f"The file '{csv_path}' has no rows of data.")

    return df


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Select and prepare numeric features for clustering.

    For this dataset we will use:
        - 'Consumption (therms)'
        - 'Consumption (GJ)'
        - 'Latitude'
        - 'Longitude'

    We:
    - Check that required columns exist.
    - Drop rows with missing values in those columns.
    - Scale the features to have mean 0 and unit variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.

    Returns
    -------
    data_scaled : np.ndarray
        2D array of shape (n_samples, n_features) with scaled data.
    """
    # List of columns we plan to use as numerical features
    required_cols = [
        "Consumption (therms)",
        "Consumption (GJ)",
        "Latitude",
        "Longitude",
    ]

    # Check that all required columns are present
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"The following required columns are missing from the dataset: {missing}"
        )

    # Select only these columns
    feature_df = df[required_cols]

    # Drop rows with missing values in any of these columns
    # (Alternative: impute missing values. For teaching, dropping is simpler.)
    feature_df = feature_df.dropna()
    if feature_df.empty:
        raise ValueError(
            "After dropping missing values, no rows remain. "
            "You may need to adjust which columns are used, or handle NaNs differently."
        )

    # Convert to NumPy array
    X = feature_df.values

    # Scale the data so that each feature has mean 0 and std 1.
    # This is important for many clustering algorithms.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def run_fuzzy_cmeans(X: np.ndarray, n_clusters: int = 3):
    """
    Run Fuzzy C-Means clustering on the feature matrix X.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters (c) for Fuzzy C-Means.

    Returns
    -------
    cntr : np.ndarray
        Cluster centers, shape (n_clusters, n_features).
    u : np.ndarray
        Final fuzzy membership array, shape (n_clusters, n_samples).
    fpc : float
        Fuzzy partition coefficient (higher is generally better).
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    n_samples, n_features = X.shape
    if n_samples < n_clusters:
        raise ValueError(
            "Number of samples is less than number of clusters. "
            "Reduce 'n_clusters' or use more data."
        )

    # Fuzzy C-Means expects data in shape (n_features, n_samples),
    # so we need to transpose X.
    data = X.T

    # Fuzzy C-Means parameters:
    #   m: fuzziness exponent (usually in [1.5, 2.5]; 2.0 is common)
    #   error: stopping criterion (minimum improvement between iterations)
    #   maxiter: maximum number of iterations
    m = 2.0
    error = 0.005
    maxiter = 1000

    # Run Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
    )

    return cntr, u, fpc


def visualize_clusters(X: np.ndarray, u: np.ndarray, n_clusters: int):
    """
    Visualize the clustering results.

    We create:
      1. A scatter plot of two features (for teaching we take feature 0 vs 1)
         with points colored by their "hard" cluster assignment
         (i.e. the cluster with highest membership).
      2. A membership bar chart for the first few samples to show fuzzy membership.

    Parameters
    ----------
    X : np.ndarray
        Scaled data, shape (n_samples, n_features).
    u : np.ndarray
        Membership matrix, shape (n_clusters, n_samples).
    n_clusters : int
        Number of clusters.
    """
    # Hard cluster assignment: pick the cluster with highest membership for each data point.
    cluster_labels = np.argmax(u, axis=0)  # shape (n_samples,)

    # --- Plot 1: Scatter plot for two features -------------------------------
    # For visualization, we use the first two features in X: X[:, 0] vs X[:, 1]
    # (e.g., scaled 'Consumption (therms)' and 'Consumption (GJ)')
    if X.shape[1] < 2:
        print("Not enough features (need at least 2) for scatter plot. Skipping.")
    else:
        plt.figure()
        for cluster_idx in range(n_clusters):
            # Select data points where the hard label matches this cluster
            mask = cluster_labels == cluster_idx
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                label=f"Cluster {cluster_idx}",
                alpha=0.6,
            )

        plt.title("Fuzzy C-Means Clustering (Hard Labels)")
        plt.xlabel("Feature 0 (scaled)")  # e.g., Consumption (therms)
        plt.ylabel("Feature 1 (scaled)")  # e.g., Consumption (GJ)
        plt.legend()
        plt.grid(True)

    # --- Plot 2: Membership bar chart for first N samples --------------------
    # Pick only the first few samples so the plot stays readable.
    N = min(20, X.shape[0])  # use up to 20 points
    indices = np.arange(N)

    plt.figure()
    bottom = np.zeros(N)  # keep track of where to stack next bar segment

    for cluster_idx in range(n_clusters):
        # Membership of the first N points to this cluster
        memberships = u[cluster_idx, :N]
        plt.bar(
            indices,
            memberships,
            bottom=bottom,
            label=f"Cluster {cluster_idx}",
            width=0.8,
        )
        # Update bottom for stacked bar chart
        bottom += memberships

    plt.title("Fuzzy Memberships for First Few Samples")
    plt.xlabel("Sample index")
    plt.ylabel("Membership degree")
    plt.xticks(indices)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y")

    # Show all plots
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function:
    - Parse arguments (optional)
    - Load data
    - Prepare features
    - Run Fuzzy C-Means
    - Visualize results
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fuzzy C-Means on natural gas consumption data."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="naturalgasbyzip.csv",
        help="Path to the CSV file (default: naturalgasbyzip.csv)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of clusters for Fuzzy C-Means (default: 3)",
    )

    args = parser.parse_args()

    # Load dataset
    try:
        df = load_dataset(args.csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    # Show basic information about the dataset
    print("Dataset loaded successfully.")
    print("First few rows:")
    print(df.head())
    print("\nColumn names:", list(df.columns))

    # Prepare numeric features for clustering
    try:
        X_scaled = prepare_features(df)
    except (KeyError, ValueError) as e:
        print(f"ERROR preparing features: {e}")
        sys.exit(1)

    print(f"\nPrepared feature matrix with shape: {X_scaled.shape}")

    # Run Fuzzy C-Means
    try:
        cntr, u, fpc = run_fuzzy_cmeans(X_scaled, n_clusters=args.clusters)
    except ValueError as e:
        print(f"ERROR during clustering: {e}")
        sys.exit(1)

    print("\nFuzzy C-Means clustering complete.")
    print(f"Cluster centers (in scaled feature space):\n{cntr}")
    print(f"\nFuzzy Partition Coefficient (FPC): {fpc:.4f}")
    print(
        "Note: FPC is between 0 and 1; values closer to 1 generally "
        "indicate better defined clusters."
    )

    # Visualize results
    visualize_clusters(X_scaled, u, n_clusters=args.clusters)


if __name__ == "__main__":
    main()
