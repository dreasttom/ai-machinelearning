"""
OPTICS (Ordering Points To Identify the Clustering Structure)
-------------------------------------------------------------

Educational implementation of the OPTICS clustering algorithm using
the "naturalgasbyzip.csv" dataset (or another user-provided CSV).

This script:
    - Loads a CSV file.
    - Lets the user choose numeric feature columns interactively
      (or via command-line arguments).
    - Prompts for / accepts eps and MinPts parameters.
    - Runs OPTICS (implemented from scratch).
    - Produces a reachability plot.

The code is heavily commented for students and includes robust error handling.
"""

import os
import sys
import math
import heapq      # For the priority queue used in the algorithm
import argparse   # For command-line argument parsing

from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# ----------------------------------------------------------------------

def load_dataset(
    file_path: str,
    feature_columns: List[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load the dataset from CSV and return a NumPy feature matrix.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    feature_columns : list of str, optional
        List of column names to use as features. If None, all numeric
        columns will be used.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    df : pd.DataFrame
        Original DataFrame (for reference / debugging).

    Raises
    ------
    FileNotFoundError
        If the CSV file is not found.
    ValueError
        If no usable numeric columns are found or data is invalid.
    """

    # Check if the file exists before attempting to load
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading CSV: {e}")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty; no data to cluster.")

    # If the user did not specify feature columns, use all numeric columns
    if feature_columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError(
                "No numeric columns found in dataset; cannot perform clustering."
            )
        feature_columns = list(numeric_df.columns)
    else:
        # Validate that requested columns exist
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following requested feature columns are missing: {missing_cols}"
            )

        # Select only numeric columns among the requested ones
        non_numeric = [
            c for c in feature_columns
            if not np.issubdtype(df[c].dtype, np.number)
        ]
        if non_numeric:
            raise ValueError(
                f"The following requested feature columns are not numeric: {non_numeric}"
            )

        numeric_df = df[feature_columns]

    # Handle missing values: simple imputation by column mean
    X = numeric_df.copy()
    X = X.fillna(X.mean(numeric_only=True))

    if X.isnull().any().any():
        # If still NaNs after imputation, fail early
        raise ValueError("NaN values remain after imputation; check data.")

    X_values = X.to_numpy(dtype=float)

    if X_values.shape[0] < 2:
        raise ValueError("Need at least 2 data points to run OPTICS.")

    return X_values, df


# ----------------------------------------------------------------------
# 2. CORE OPTICS IMPLEMENTATION
# ----------------------------------------------------------------------

def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for a dataset X.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    distances : np.ndarray
        Pairwise distance matrix of shape (n_samples, n_samples).
    """
    # Broadcasting: (n,1,d) - (1,n,d) -> (n,n,d) then square, sum, sqrt
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances


def neighbors(dist_matrix: np.ndarray, idx: int, eps: float) -> np.ndarray:
    """
    Find indices of all points within distance eps of point idx.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix (n_samples, n_samples).
    idx : int
        Index of the reference point.
    eps : float
        Neighborhood radius.

    Returns
    -------
    np.ndarray
        Array of neighbor indices (including idx itself).
    """
    return np.where(dist_matrix[idx] <= eps)[0]


def compute_core_distance(
    dist_matrix: np.ndarray,
    idx: int,
    eps: float,
    min_pts: int
) -> float:
    """
    Compute the core-distance for a point.

    Definition (simplified):
        - Let N_eps(p) be the set of neighbors within radius eps (including p).
        - If |N_eps(p)| < min_pts, core-distance is undefined (NaN).
        - Else, core-distance = distance to the min_pts-th nearest neighbor.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix.
    idx : int
        Index of the point p.
    eps : float
        Neighborhood radius.
    min_pts : int
        Minimum number of points to be a core point.

    Returns
    -------
    float
        Core-distance value, or NaN if undefined.
    """
    # Get the distances from point idx to all others
    distances = dist_matrix[idx]

    # Only neighbors within eps are considered
    # (including the point itself at distance 0)
    neigh_mask = distances <= eps
    neigh_distances = distances[neigh_mask]

    if len(neigh_distances) < min_pts:
        # Not enough neighbors -> not a core point
        return math.nan

    # Sort neighbor distances and take the min_pts-th smallest
    sorted_distances = np.sort(neigh_distances)
    core_dist = float(sorted_distances[min_pts - 1])

    return core_dist


def update_seeds(
    dist_matrix: np.ndarray,
    point_idx: int,
    neighbors_idx: np.ndarray,
    core_distances: np.ndarray,
    reachability_distances: np.ndarray,
    processed: np.ndarray,
    order_seeds: List[Tuple[float, int]]
) -> None:
    """
    Update the reachability distance of neighbors of a core point
    and push them into the priority queue (order_seeds).

    Parameters
    ----------
    dist_matrix : np.ndarray
        Precomputed distance matrix.
    point_idx : int
        Index of the core point p.
    neighbors_idx : np.ndarray
        Indices of neighbors of p.
    core_distances : np.ndarray
        Array of core distances for all points.
    reachability_distances : np.ndarray
        Array of reachability distances for all points (NaN = undefined).
    processed : np.ndarray
        Boolean array indicating whether each point has been processed.
    order_seeds : list of (float, int)
        Min-heap (priority queue) of seeds; elements are (reachability_distance, index).
    """
    for o in neighbors_idx:
        if processed[o]:
            # Already processed; skip
            continue

        # Reachability distance defined as:
        #    reachability(o) = max(core_distance(p), distance(p, o))
        new_reach_dist = max(core_distances[point_idx], dist_matrix[point_idx, o])

        if math.isnan(reachability_distances[o]):
            # No reachability distance yet -> set it
            reachability_distances[o] = new_reach_dist
            heapq.heappush(order_seeds, (new_reach_dist, o))
        else:
            # Only update if we found a smaller (better) reachability distance
            if new_reach_dist < reachability_distances[o]:
                reachability_distances[o] = new_reach_dist
                # We push again; duplicates with larger keys will be ignored
                # when popped because 'processed' will be True by then.
                heapq.heappush(order_seeds, (new_reach_dist, o))


def optics(
    X: np.ndarray,
    eps: float,
    min_pts: int
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Run the OPTICS clustering algorithm on a feature matrix X.

    Note:
        - This implementation focuses on computing the ordering and reachability
          distances. It does NOT extract clusters; instead you can interpret
          the reachability plot manually or implement a separate extraction step.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    eps : float
        Maximum radius for neighborhood queries.
    min_pts : int
        Minimum number of points to be considered a core point.

    Returns
    -------
    ordering : list of int
        The order in which points are processed.
    reachability_distances : np.ndarray
        Reachability distance for each point (NaN if undefined).
    core_distances : np.ndarray
        Core distance for each point (NaN if undefined).

    Raises
    ------
    ValueError
        If parameters are invalid.
    """

    n_samples = X.shape[0]

    # Basic parameter validation
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if min_pts < 2:
        raise ValueError("min_pts should be at least 2.")
    if min_pts > n_samples:
        raise ValueError("min_pts cannot be larger than the number of samples.")

    # Precompute distance matrix (O(n^2) memory/time)
    dist_matrix = compute_distance_matrix(X)

    # Initialize arrays
    processed = np.zeros(n_samples, dtype=bool)
    reachability_distances = np.full(n_samples, np.nan)
    core_distances = np.full(n_samples, np.nan)

    ordering: List[int] = []

    # Loop over all points; expand each unprocessed point
    for p in range(n_samples):
        if processed[p]:
            continue

        # Neighborhood of p
        N = neighbors(dist_matrix, p, eps)

        processed[p] = True
        ordering.append(p)

        # Compute core-distance of p
        core_distances[p] = compute_core_distance(dist_matrix, p, eps, min_pts)

        if math.isnan(core_distances[p]):
            # Not a core point -> cannot expand cluster from p
            continue

        # Initialize priority queue (min-heap) of seeds
        order_seeds: List[Tuple[float, int]] = []

        # Update reachability distances for neighbors
        update_seeds(
            dist_matrix,
            p,
            N,
            core_distances,
            reachability_distances,
            processed,
            order_seeds
        )

        # Process the seeds in order of increasing reachability distance
        while order_seeds:
            # Pop the point with smallest reachability distance
            _, q = heapq.heappop(order_seeds)

            if processed[q]:
                # Could be a stale entry in the heap; ignore
                continue

            Nq = neighbors(dist_matrix, q, eps)
            processed[q] = True
            ordering.append(q)

            # Compute core-distance of q
            core_distances[q] = compute_core_distance(dist_matrix, q, eps, min_pts)

            if not math.isnan(core_distances[q]):
                # If q is a core point, update its neighbors
                update_seeds(
                    dist_matrix,
                    q,
                    Nq,
                    core_distances,
                    reachability_distances,
                    processed,
                    order_seeds
                )

    return ordering, reachability_distances, core_distances


# ----------------------------------------------------------------------
# 3. VISUALIZATION: REACHABILITY PLOT
# ----------------------------------------------------------------------

def plot_reachability(
    ordering: List[int],
    reachability_distances: np.ndarray,
    output_path: str = "reachability_plot.png"
) -> None:
    """
    Create a reachability plot using the OPTICS ordering.

    Parameters
    ----------
    ordering : list of int
        The OPTICS ordering of point indices.
    reachability_distances : np.ndarray
        Reachability distances for each point.
    output_path : str
        File path where the plot image will be saved.
    """
    # Reorder reachability distances according to the OPTICS ordering
    ordered_reachability = [
        reachability_distances[idx] for idx in ordering
    ]

    # Create the figure
    plt.figure(figsize=(10, 5))

    # Plot reachability distance vs. ordering index
    plt.plot(range(len(ordering)), ordered_reachability, marker='.', linestyle='-')

    plt.xlabel("Point order (OPTICS ordering)")
    plt.ylabel("Reachability distance")
    plt.title("OPTICS Reachability Plot")

    plt.tight_layout()

    # Save figure to disk
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Reachability plot saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save plot to '{output_path}': {e}")
    finally:
        plt.close()


# ----------------------------------------------------------------------
# 4. INTERACTIVE HELPERS
# ----------------------------------------------------------------------

def choose_features_interactively(df: pd.DataFrame,
                                  default_features: List[str] = None) -> List[str]:
    """
    Let the user choose numeric feature columns interactively.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    default_features : list of str, optional
        A suggested default subset of numeric columns. If all of them are
        present and numeric, the user can accept them by pressing Enter.

    Returns
    -------
    feature_columns : list of str
        The chosen feature column names.

    Raises
    ------
    ValueError
        If no numeric columns are available.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns available in the dataset.")

    print("\nAvailable numeric columns:")
    for i, col in enumerate(numeric_cols):
        print(f"  [{i}] {col}")

    # Check if default features are valid (exist and numeric)
    valid_default = False
    if default_features is not None:
        valid_default = all(col in numeric_cols for col in default_features)

    if valid_default:
        print("\nPress ENTER to use the default features:")
        print("  " + ", ".join(default_features))
    else:
        print("\nPress ENTER to use ALL numeric columns.")

    print("Or enter comma-separated indices (e.g., 0,1,2) to choose specific features.")

    while True:
        user_input = input("Your choice: ").strip()

        # User pressed Enter
        if user_input == "":
            if valid_default:
                print(f"Using default features: {default_features}")
                return default_features
            else:
                print(f"Using all numeric columns: {numeric_cols}")
                return numeric_cols

        # Try to parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in user_input.split(",") if x.strip() != ""]
        except ValueError:
            print("Invalid input. Please enter comma-separated integer indices or press ENTER.")
            continue

        if not indices:
            print("No indices provided. Try again or press ENTER for default.")
            continue

        # Check index range
        if any(i < 0 or i >= len(numeric_cols) for i in indices):
            print("One or more indices are out of range. Please try again.")
            continue

        chosen = [numeric_cols[i] for i in indices]
        print(f"Using chosen features: {chosen}")
        return chosen


def prompt_float(prompt_text: str, default: float, min_val: float = None) -> float:
    """
    Prompt the user for a float value with a default and optional minimum.

    Parameters
    ----------
    prompt_text : str
        Text to display to the user.
    default : float
        Default value if the user presses Enter.
    min_val : float, optional
        Minimum allowed value (inclusive). If None, any float is allowed.

    Returns
    -------
    value : float
        The chosen or default float value.
    """
    while True:
        user_input = input(f"{prompt_text} [{default}]: ").strip()
        if user_input == "":
            value = default
        else:
            try:
                value = float(user_input)
            except ValueError:
                print("Invalid number. Please try again.")
                continue

        if min_val is not None and value < min_val:
            print(f"Value must be at least {min_val}. Please try again.")
            continue

        return value


def prompt_int(prompt_text: str, default: int, min_val: int = None) -> int:
    """
    Prompt the user for an integer value with a default and optional minimum.

    Parameters
    ----------
    prompt_text : str
        Text to display to the user.
    default : int
        Default value if the user presses Enter.
    min_val : int, optional
        Minimum allowed value (inclusive). If None, any int is allowed.

    Returns
    -------
    value : int
        The chosen or default integer value.
    """
    while True:
        user_input = input(f"{prompt_text} [{default}]: ").strip()
        if user_input == "":
            value = default
        else:
            try:
                value = int(user_input)
            except ValueError:
                print("Invalid integer. Please try again.")
                continue

        if min_val is not None and value < min_val:
            print(f"Value must be at least {min_val}. Please try again.")
            continue

        return value


# ----------------------------------------------------------------------
# 5. MAIN SCRIPT ENTRY POINT
# ----------------------------------------------------------------------

def main():
    """
    Main function:
        - Parses command-line arguments.
        - Loads the dataset.
        - Lets the user choose feature columns (if not provided).
        - Lets the user choose eps and MinPts (if not provided).
        - Runs OPTICS.
        - Generates a reachability plot.
    """

    parser = argparse.ArgumentParser(
        description="Educational OPTICS implementation with interactive options."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="naturalgasbyzip.csv",
        help="Path to CSV file (default: naturalgasbyzip.csv)."
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of numeric feature columns to use. "
             "If omitted, you will be prompted interactively."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Neighborhood radius (eps). If omitted, you will be prompted."
    )
    parser.add_argument(
        "--min-pts",
        type=int,
        default=None,
        help="Minimum number of points for a core point (MinPts). "
             "If omitted, you will be prompted."
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="reachability_plot.png",
        help="Output path for the reachability plot PNG."
    )

    args = parser.parse_args()

    csv_path = args.csv_path

    # First, try to load the dataset (this also checks the file)
    try:
        # Here feature_columns=None -> uses all numeric columns as a first pass
        X_initial, df = load_dataset(csv_path, feature_columns=None)
        print(f"Loaded dataset from '{csv_path}' with shape: {X_initial.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Decide feature columns: either from CLI or interactive
    if args.features is not None:
        # User provided a comma-separated list of features
        feature_cols = [col.strip() for col in args.features.split(",") if col.strip()]

        print(f"Using features from command line: {feature_cols}")
    else:
        # Interactive feature selection
        default_features = ["Consumption (therms)", "Consumption (GJ)"]
        feature_cols = choose_features_interactively(df, default_features=default_features)

    # Now load dataset again, this time with the chosen feature columns
    try:
        X, df = load_dataset(csv_path, feature_columns=feature_cols)
        print(f"\nFinal feature matrix shape: {X.shape}")
        print(f"Feature columns: {feature_cols}")
    except Exception as e:
        print(f"Error preparing feature matrix with chosen columns: {e}")
        sys.exit(1)

    # Decide eps and min_pts: either from CLI or interactive
    # Provide defaults that worked reasonably for the earlier version
    default_eps = 5000.0
    default_min_pts = 5

    if args.eps is not None:
        if args.eps <= 0:
            print("Error: --eps must be > 0.")
            sys.exit(1)
        eps = args.eps
    else:
        print("\nChoose OPTICS parameter eps (neighborhood radius).")
        eps = prompt_float("Enter eps", default=default_eps, min_val=1e-9)

    if args.min_pts is not None:
        if args.min_pts < 2:
            print("Error: --min-pts must be at least 2.")
            sys.exit(1)
        min_pts = args.min_pts
    else:
        print("\nChoose OPTICS parameter MinPts (minimum number of points for a core point).")
        min_pts = prompt_int("Enter MinPts", default=default_min_pts, min_val=2)

    print(f"\nRunning OPTICS with eps={eps}, MinPts={min_pts} ...")

    # Run the OPTICS algorithm
    try:
        ordering, reachability, core_distances = optics(X, eps=eps, min_pts=min_pts)
    except Exception as e:
        print(f"Error running OPTICS algorithm: {e}")
        sys.exit(1)

    # Print some basic info about the results
    print("\nOPTICS completed.")
    print(f"Number of points processed: {len(ordering)}")
    print("First 10 points in ordering:", ordering[:10])
    print("First 10 reachability distances:", reachability[ordering[:10]])

    # Produce reachability plot
    plot_reachability(ordering, reachability, output_path=args.plot_path)


if __name__ == "__main__":
    main()
