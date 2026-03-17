#!/usr/bin/env python3
"""
Deep Embedded Clustering (DEC) for the attached US Navy JSON dataset.

This script is intentionally verbose, heavily commented, and defensive.
It is designed to work out of the box when placed in the same directory as:
    us_navy.json
    This json file is a subset of https://www.kaggle.com/datasets/queyrusi/the-warship-dataset

Main features
-------------
1. Loads a JSON dataset of ship records.
2. Automatically detects numeric vs categorical columns.
3. Preprocesses data into a dense numeric feature matrix.
4. Trains an autoencoder to learn a latent representation.
5. Runs Deep Embedded Clustering (DEC) on the latent space.
6. Saves cluster assignments and optional visualizations.
7. Includes robust error handling and detailed console logging.

Example usage
-------------
python dec_us_navy.py
python dec_us_navy.py --data ./us_navy.json --clusters 8 --epochs 100
python dec_us_navy.py --no-visuals

Dependencies
------------
- numpy
- pandas
- scikit-learn
- matplotlib
- torch

Install example:
    pip install numpy pandas scikit-learn matplotlib torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------
# We perform imports in a guarded block so the user gets a clean, actionable
# message instead of a raw stack trace if a dependency is missing.
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    missing_name = getattr(exc, "name", "a required package")
    print(
        f"[ERROR] Missing dependency: {missing_name}\n"
        "Please install required packages, for example:\n"
        "    pip install numpy pandas scikit-learn matplotlib torch",
        file=sys.stderr,
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
# Structured logging helps users understand what phase the script is in.
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("dec")


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data classes for clean configuration passing
# -----------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration bundle for DEC training."""

    n_clusters: int = 8
    latent_dim: int = 10
    hidden_dims: Tuple[int, int] = (256, 128)
    pretrain_epochs: int = 75
    dec_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    update_interval: int = 5
    tol: float = 1e-3
    seed: int = 42


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------
def resolve_default_dataset_path() -> Path:
    """
    Return the default dataset path.

    The script defaults to a file named `us_navy.json` in the same directory
    as this script, which matches the user's requested behavior.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir / "us_navy.json"



def load_json_dataset(path: Path) -> pd.DataFrame:
    """
    Load a JSON dataset defensively.

    Expected formats supported:
    - A JSON array of objects.
    - A newline-delimited JSON file is *not* assumed here.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Provided dataset path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file '{path}': {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to open/read file '{path}': {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            "Expected the JSON file to contain a top-level list of records."
        )

    if len(data) == 0:
        raise ValueError("The JSON dataset is empty; there is nothing to cluster.")

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("The JSON dataset produced an empty DataFrame.")

    return df


# -----------------------------------------------------------------------------
# Preprocessing utilities
# -----------------------------------------------------------------------------
def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across scikit-learn versions.

    Newer versions use `sparse_output=False`; older versions use `sparse=False`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def preprocess_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    Clean and transform the DataFrame into a dense numeric matrix.

    Strategy:
    - Preserve original DataFrame for later reporting.
    - Treat object/string/category columns as categorical.
    - Treat numeric columns as numeric.
    - Fill missing categorical values with a literal token.
    - Fill missing numeric values with the column median.
    - Scale numeric values and one-hot encode categorical values.

    Returns
    -------
    X : np.ndarray
        Dense numeric feature matrix for modeling.
    cleaned_df : pd.DataFrame
        Cleaned copy of the original data for downstream use.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline for inspection if needed.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty; preprocessing cannot proceed.")

    cleaned_df = df.copy()

    # Drop columns that are completely empty.
    all_null_columns = [col for col in cleaned_df.columns if cleaned_df[col].isna().all()]
    if all_null_columns:
        logger.warning("Dropping entirely empty columns: %s", all_null_columns)
        cleaned_df = cleaned_df.drop(columns=all_null_columns)

    if cleaned_df.empty or cleaned_df.shape[1] == 0:
        raise ValueError("All columns were empty after cleaning.")

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cleaned_df.columns if c not in numeric_cols]

    # Fill missing values conservatively.
    for col in numeric_cols:
        if cleaned_df[col].isna().any():
            median_value = cleaned_df[col].median()
            # If the numeric column is all NaN after all, fall back to 0.
            if pd.isna(median_value):
                median_value = 0.0
            cleaned_df[col] = cleaned_df[col].fillna(median_value)

    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].astype(str).fillna("<MISSING>")
        # Replace blank-like strings with a stable marker.
        cleaned_df[col] = cleaned_df[col].replace({"": "<BLANK>", "nan": "<MISSING>"})

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("scaler", StandardScaler())]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(steps=[("onehot", make_one_hot_encoder())]),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable columns were found for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers)

    try:
        X = preprocessor.fit_transform(cleaned_df)
    except Exception as exc:
        raise RuntimeError(f"Failed during feature preprocessing: {exc}") from exc

    # Ensure a dense float32 array for PyTorch.
    try:
        X = np.asarray(X, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert features to float32 ndarray: {exc}") from exc

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix; got shape {X.shape}")

    if X.shape[0] < 2:
        raise ValueError("At least 2 rows are required for clustering.")

    if X.shape[1] < 2:
        logger.warning(
            "Feature matrix has only %d feature(s). DEC may work poorly with such low dimensionality.",
            X.shape[1],
        )

    return X, cleaned_df, preprocessor


# -----------------------------------------------------------------------------
# PyTorch model definitions
# -----------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    """
    Simple fully connected autoencoder.

    The encoder maps the input into a lower-dimensional latent representation.
    The decoder reconstructs the input from that latent representation.
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int], latent_dim: int):
        super().__init__()

        h1, h2 = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class DECModel(nn.Module):
    """
    Deep Embedded Clustering model.

    DEC keeps an autoencoder encoder and a set of learnable cluster centers.
    It computes soft assignments using a Student's t-distribution.
    """

    def __init__(self, autoencoder: AutoEncoder, n_clusters: int, latent_dim: int):
        super().__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Cluster centers are learned during DEC fine-tuning.
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent embedding from the autoencoder encoder."""
        return self.autoencoder.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, latent embedding, and soft cluster assignments."""
        x_hat, z = self.autoencoder(x)
        q = self.soft_assign(z)
        return x_hat, z, q

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments using the DEC Student t-distribution.

        q_ij = (1 + ||z_i - mu_j||^2)^(-1) / normalization
        """
        # Compute pairwise squared distances between latent points and centers.
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq)
        q = q ** ((1.0 + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


# -----------------------------------------------------------------------------
# DEC objective helpers
# -----------------------------------------------------------------------------
def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the DEC target distribution.

    This sharpens confident assignments and re-balances cluster frequencies.
    """
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    return weight / torch.sum(weight, dim=1, keepdim=True)



def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence used by DEC between target p and soft assignments q."""
    eps = 1e-10
    return torch.mean(torch.sum(p * torch.log((p + eps) / (q + eps)), dim=1))


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def batch_iterator(X: np.ndarray, batch_size: int):
    """Yield mini-batches from a numpy array."""
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx]



def pretrain_autoencoder(
    model: AutoEncoder,
    X: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> List[float]:
    """Train the autoencoder on reconstruction loss."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch in batch_iterator(X, batch_size):
            batch_tensor = torch.from_numpy(batch).to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_tensor)
            loss = criterion(recon, batch_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info("Pretrain epoch %d/%d - reconstruction loss: %.6f", epoch, epochs, avg_loss)

    return losses



def initialize_cluster_centers(
    dec_model: DECModel,
    X: np.ndarray,
    device: torch.device,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """
    Run KMeans on latent embeddings to initialize DEC cluster centers.

    Returns the initial cluster labels from KMeans.
    """
    dec_model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        latent = dec_model.encode(X_tensor).cpu().numpy()

    if latent.shape[0] < n_clusters:
        raise ValueError(
            f"Number of clusters ({n_clusters}) exceeds number of samples ({latent.shape[0]})."
        )

    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        y_pred = kmeans.fit_predict(latent)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize KMeans cluster centers: {exc}") from exc

    with torch.no_grad():
        dec_model.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )

    return y_pred



def train_dec(
    dec_model: DECModel,
    X: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    update_interval: int,
    tol: float,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Train DEC using KL divergence and occasional target distribution updates.

    Returns
    -------
    final_labels : np.ndarray
        Final hard cluster labels.
    kl_history : List[float]
        Per-epoch DEC KL losses.
    recon_history : List[float]
        Per-epoch reconstruction losses during DEC fine-tuning.
    """
    optimizer = optim.Adam(dec_model.parameters(), lr=learning_rate)
    recon_criterion = nn.MSELoss()

    X_tensor = torch.from_numpy(X).to(device)
    previous_labels: Optional[np.ndarray] = None
    kl_history: List[float] = []
    recon_history: List[float] = []

    for epoch in range(1, epochs + 1):
        dec_model.eval()
        with torch.no_grad():
            _, _, q_full = dec_model(X_tensor)
            p_full = target_distribution(q_full)
            current_labels = torch.argmax(q_full, dim=1).cpu().numpy()

        # Early stopping if labels stabilize.
        if previous_labels is not None:
            delta = np.mean(current_labels != previous_labels)
            if epoch > 1 and delta < tol:
                logger.info(
                    "DEC early stopping at epoch %d because label change ratio %.6f < tol %.6f",
                    epoch,
                    delta,
                    tol,
                )
                break
        previous_labels = current_labels.copy()

        dec_model.train()
        epoch_kl = 0.0
        epoch_recon = 0.0
        n_batches = 0

        # Mini-batch update using the precomputed target distribution.
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            batch_idx = indices[start:end]

            batch_x = X_tensor[batch_idx]
            batch_p = p_full[batch_idx].detach()

            optimizer.zero_grad()
            x_hat, _, q_batch = dec_model(batch_x)

            # Standard DEC uses KL divergence. We keep a small reconstruction term
            # to help stabilize training on small real-world datasets.
            kl_loss = kl_divergence(batch_p, q_batch)
            recon_loss = recon_criterion(x_hat, batch_x)
            loss = kl_loss + 0.1 * recon_loss

            loss.backward()
            optimizer.step()

            epoch_kl += float(kl_loss.item())
            epoch_recon += float(recon_loss.item())
            n_batches += 1

        avg_kl = epoch_kl / max(1, n_batches)
        avg_recon = epoch_recon / max(1, n_batches)
        kl_history.append(avg_kl)
        recon_history.append(avg_recon)

        if epoch == 1 or epoch % max(1, update_interval) == 0 or epoch == epochs:
            logger.info(
                "DEC epoch %d/%d - KL loss: %.6f - recon loss: %.6f",
                epoch,
                epochs,
                avg_kl,
                avg_recon,
            )

    # Final label inference.
    dec_model.eval()
    with torch.no_grad():
        _, _, q_final = dec_model(X_tensor)
        final_labels = torch.argmax(q_final, dim=1).cpu().numpy()

    return final_labels, kl_history, recon_history


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def safe_reduce_for_plot(embeddings: np.ndarray, seed: int) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.

    Strategy:
    - If already 2D, return as is.
    - If dimensionality > 2 and sample count is small, use PCA first.
    - For very small datasets, PCA is more stable than t-SNE.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D for plotting; got {embeddings.shape}")

    if embeddings.shape[1] == 2:
        return embeddings

    if embeddings.shape[0] < 5:
        # Tiny datasets are better served by PCA.
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)

    # For moderately sized data, t-SNE can be more visually interesting.
    # We fall back to PCA if t-SNE fails for any reason.
    try:
        perplexity = min(30, max(2, embeddings.shape[0] // 10))
        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(embeddings)
    except Exception:
        logger.warning("t-SNE failed; falling back to PCA for visualization.")
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)



def save_training_curves(
    pretrain_losses: List[float],
    dec_kl_losses: List[float],
    dec_recon_losses: List[float],
    output_dir: Path,
) -> None:
    """Save line plots for training diagnostics."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(pretrain_losses, label="Autoencoder pretrain reconstruction loss")
        plt.plot(
            np.arange(len(pretrain_losses), len(pretrain_losses) + len(dec_kl_losses)),
            dec_kl_losses,
            label="DEC KL loss",
        )
        plt.plot(
            np.arange(len(pretrain_losses), len(pretrain_losses) + len(dec_recon_losses)),
            dec_recon_losses,
            label="DEC reconstruction loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("DEC Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=200)
        plt.close()
    except Exception as exc:
        logger.warning("Failed to save training curves: %s", exc)



def save_cluster_plot(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a 2D scatter plot of clustered embeddings."""
    try:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, s=40)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(title)
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    except Exception as exc:
        logger.warning("Failed to save cluster plot '%s': %s", output_path.name, exc)



def save_cluster_count_plot(labels: np.ndarray, output_path: Path) -> None:
    """Save a simple bar chart of cluster membership counts."""
    try:
        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(8, 5))
        plt.bar(unique.astype(str), counts)
        plt.xlabel("Cluster")
        plt.ylabel("Number of records")
        plt.title("Cluster Membership Counts")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    except Exception as exc:
        logger.warning("Failed to save cluster count plot: %s", exc)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def save_clustered_results(df: pd.DataFrame, labels: np.ndarray, output_dir: Path) -> Path:
    """Write the original data plus cluster labels to CSV."""
    if len(df) != len(labels):
        raise ValueError("Row count mismatch between DataFrame and predicted labels.")

    result_df = df.copy()
    result_df["cluster"] = labels
    output_path = output_dir / "clustered_output.csv"
    result_df.to_csv(output_path, index=False)
    return output_path



def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Produce a light-weight summary table for each cluster.

    For interpretability, the summary includes:
    - cluster id
    - size
    - most common class/type/port if those columns exist
    """
    result = df.copy()
    result["cluster"] = labels

    possible_summary_cols = [c for c in ["class", "type", "port", "pays"] if c in result.columns]

    rows = []
    for cluster_id, group in result.groupby("cluster"):
        row: Dict[str, object] = {
            "cluster": int(cluster_id),
            "size": int(len(group)),
        }
        for col in possible_summary_cols:
            try:
                mode_series = group[col].mode(dropna=True)
                row[f"top_{col}"] = mode_series.iloc[0] if not mode_series.empty else "N/A"
            except Exception:
                row[f"top_{col}"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full DEC workflow and return a process exit code."""
    set_seed(args.seed)

    data_path = Path(args.data).resolve() if args.data else resolve_default_dataset_path()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_path.parent / "dec_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using dataset: %s", data_path)
    logger.info("Saving outputs to: %s", output_dir)

    # -----------------------------
    # Step 1: Load data
    # -----------------------------
    df = load_json_dataset(data_path)
    logger.info("Loaded dataset with %d rows and %d columns.", df.shape[0], df.shape[1])

    # Guard against unrealistic clustering requests.
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2.")
    if args.clusters > len(df):
        raise ValueError(
            f"--clusters ({args.clusters}) cannot exceed number of rows ({len(df)})."
        )

    # -----------------------------
    # Step 2: Preprocess
    # -----------------------------
    X, cleaned_df, _ = preprocess_dataframe(df)
    logger.info("Feature matrix shape after preprocessing: %s", X.shape)

    # Adjust latent dimension if the request exceeds sensible bounds.
    latent_dim = min(args.latent_dim, max(2, X.shape[1] - 1))
    if latent_dim != args.latent_dim:
        logger.warning(
            "Requested latent_dim=%d adjusted to %d based on feature dimensionality.",
            args.latent_dim,
            latent_dim,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    logger.info("Using device: %s", device)

    config = TrainingConfig(
        n_clusters=args.clusters,
        latent_dim=latent_dim,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        pretrain_epochs=args.pretrain_epochs,
        dec_epochs=args.dec_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        update_interval=args.update_interval,
        tol=args.tol,
        seed=args.seed,
    )

    # -----------------------------
    # Step 3: Build models
    # -----------------------------
    autoencoder = AutoEncoder(
        input_dim=X.shape[1],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    ).to(device)

    dec_model = DECModel(
        autoencoder=autoencoder,
        n_clusters=config.n_clusters,
        latent_dim=config.latent_dim,
    ).to(device)

    # -----------------------------
    # Step 4: Pretrain autoencoder
    # -----------------------------
    logger.info("Starting autoencoder pretraining...")
    pretrain_losses = pretrain_autoencoder(
        model=autoencoder,
        X=X,
        device=device,
        epochs=config.pretrain_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )

    # -----------------------------
    # Step 5: Initialize DEC centers
    # -----------------------------
    logger.info("Initializing cluster centers with KMeans on latent embeddings...")
    _ = initialize_cluster_centers(
        dec_model=dec_model,
        X=X,
        device=device,
        n_clusters=config.n_clusters,
        seed=config.seed,
    )

    # -----------------------------
    # Step 6: DEC fine-tuning
    # -----------------------------
    logger.info("Starting DEC fine-tuning...")
    labels, dec_kl_losses, dec_recon_losses = train_dec(
        dec_model=dec_model,
        X=X,
        device=device,
        epochs=config.dec_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        update_interval=config.update_interval,
        tol=config.tol,
    )

    # -----------------------------
    # Step 7: Save outputs
    # -----------------------------
    csv_path = save_clustered_results(cleaned_df, labels, output_dir)
    summary_df = summarize_clusters(cleaned_df, labels)
    summary_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info("Saved clustered output to: %s", csv_path)
    logger.info("Saved cluster summary to: %s", summary_path)

    # -----------------------------
    # Step 8: Metrics and visuals
    # -----------------------------
    try:
        with torch.no_grad():
            latent = dec_model.encode(torch.from_numpy(X).to(device)).cpu().numpy()

        # Silhouette score is useful but only valid when there is more than one cluster
        # and fewer clusters than samples.
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
            sil = silhouette_score(latent, labels)
            logger.info("Silhouette score on latent space: %.4f", sil)
        else:
            logger.warning("Silhouette score skipped because valid cluster structure was not present.")

        if not args.no_visuals:
            latent_2d = safe_reduce_for_plot(latent, seed=args.seed)
            save_cluster_plot(
                embeddings_2d=latent_2d,
                labels=labels,
                output_path=output_dir / "latent_clusters.png",
                title="DEC Clusters in 2D Latent Space",
            )
            save_cluster_count_plot(labels, output_dir / "cluster_counts.png")
            save_training_curves(pretrain_losses, dec_kl_losses, dec_recon_losses, output_dir)
            logger.info("Saved visualization files to: %s", output_dir)
    except Exception as exc:
        logger.warning("Metrics/visualization step encountered an issue: %s", exc)

    # -----------------------------
    # Step 9: Console summary
    # -----------------------------
    print("\n=== DEC Cluster Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nClustered CSV: {csv_path}")
    if not args.no_visuals:
        print(f"Visual outputs folder: {output_dir}")

    return 0


# -----------------------------------------------------------------------------
# CLI setup
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Deep Embedded Clustering (DEC) on a JSON dataset."
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the JSON dataset. Defaults to us_navy.json in the same folder as this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where outputs will be written. Defaults to ./dec_output next to the dataset.",
    )
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters.")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimension size.")
    parser.add_argument("--hidden-dim-1", type=int, default=256, help="First hidden layer size.")
    parser.add_argument("--hidden-dim-2", type=int, default=128, help="Second hidden layer size.")
    parser.add_argument("--pretrain-epochs", type=int, default=75, help="Autoencoder pretraining epochs.")
    parser.add_argument("--dec-epochs", type=int, default=100, help="DEC fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--update-interval", type=int, default=5, help="How often to print DEC progress.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Early stopping tolerance on label changes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--no-visuals", action="store_true", help="Disable plot generation.")

    return parser


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> int:
    """
    Main entry point with top-level exception handling.

    This is deliberately broad so the script fails gracefully and provides a
    useful debugging trail instead of silently crashing.
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.error("Execution interrupted by user.")
        return 130
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        logger.debug("Full traceback:\n%s", traceback.format_exc())
        print("\nDetailed traceback for debugging:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
