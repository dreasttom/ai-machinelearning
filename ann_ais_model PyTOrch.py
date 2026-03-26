"""
ann_ais_model.py

A robust, heavily commented PyTorch-based Artificial Neural Network (ANN)
training script for the AIS_2023_MINI.csv dataset.

Assumptions
-----------
- The CSV file is named "AIS_2023_MINI.csv"
- The CSV file is in the same folder as this script
- The final column is the target label
- The dataset may contain a fake/example first row like: A,B,C,...
- The dataset may contain mixed data types: numeric, categorical, datetime strings,
  and common missing-value markers.

What this script does
---------------------
1. Loads the CSV safely without trusting headers.
2. Removes a fake/example header row if it exists.
3. Normalizes common missing-value markers.
4. Converts datetime-like columns to numeric timestamps.
5. Converts numeric-like text columns to numeric dtype where appropriate.
6. Separates features and target.
7. Builds a preprocessing pipeline for numeric + categorical data.
8. Encodes labels.
9. Trains a PyTorch ANN with early stopping.
10. Evaluates the model using accuracy, confusion matrix, and classification report.
11. Saves the model checkpoint, preprocessing pipeline, and label mapping to disk.

Outputs
-------
- ais_ann_model.pt                 -> PyTorch checkpoint
- ais_ann_preprocessor.pkl         -> fitted scikit-learn preprocessor
- ais_ann_label_mapping.json       -> class names / mappings

Install requirements (example)
------------------------------
pip install torch pandas numpy scikit-learn joblib
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as import_error:
    raise ImportError(
        "PyTorch could not be imported. Install it with: pip install torch"
    ) from import_error


# =============================================================================
# Utility helpers
# =============================================================================

def print_section(title: str) -> None:
    """Print a clean visual section divider to the console."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def set_reproducible_seed(seed: int = 42) -> None:
    """
    Set random seeds where practical so runs are more reproducible.

    Note:
    Perfect reproducibility is not always guaranteed across every machine,
    OS, or PyTorch backend, but this improves consistency.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Ask PyTorch to use deterministic algorithms when possible.
        # If an operation is unsupported, PyTorch may fall back depending on version.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older versions of PyTorch may not support this fully.
            pass

        # cuDNN determinism settings (only relevant on CUDA systems).
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except Exception as exc:
        raise RuntimeError(f"Failed to set random seed: {exc}") from exc



def get_device() -> torch.device:
    """
    Choose the best available PyTorch device.

    Returns
    -------
    torch.device
        'cuda' if available, otherwise 'cpu'.
    """
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:
        raise RuntimeError(f"Unable to determine PyTorch device: {exc}") from exc



def resolve_dataset_path(filename: str = "AIS_2023_MINI.csv") -> Path:
    """Resolve the dataset path relative to the script's folder."""
    try:
        script_dir = Path(__file__).resolve().parent
        csv_path = script_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                f"Make sure '{filename}' is in the same folder as this script."
            )
        if not csv_path.is_file():
            raise FileNotFoundError(f"Expected a file but found something else: {csv_path}")

        return csv_path
    except Exception as exc:
        raise FileNotFoundError(f"Unable to resolve dataset path: {exc}") from exc


# =============================================================================
# Data loading and cleaning
# =============================================================================

def load_csv_without_assumptions(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV without assuming it has a valid header row.

    Why this matters:
    Some datasets do not have a real header, and letting pandas treat the first
    row as a header can silently corrupt the dataset structure.
    """
    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            dtype=str,
            keep_default_na=True,
            na_values=[
                "", " ", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null",
                "None", "none", "?"
            ],
        )

        if df.empty:
            raise ValueError("The CSV file loaded successfully but contains no rows.")

        # Assign safe synthetic column names.
        df.columns = [f"column_{i}" for i in range(df.shape[1])]
        return df

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV file could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"The CSV file encoding could not be decoded: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Unexpected error while reading CSV: {exc}") from exc



def remove_fake_header_row_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a fake/example first row such as A,B,C,... if present.

    This dataset family has sometimes included a first row that looks like an
    example header rather than real data. We detect that pattern heuristically.
    """
    try:
        if df.empty:
            return df

        first_row = df.iloc[0].astype(str).fillna("")
        short_alpha_count = 0

        for value in first_row:
            cleaned = value.strip()
            if cleaned.isalpha() and len(cleaned) <= 3:
                short_alpha_count += 1

        # If most columns in row 0 are short alphabetic tokens, treat it as fake.
        if short_alpha_count >= max(3, int(len(first_row) * 0.7)):
            print("Detected a fake/example header row. Removing row 0.")
            return df.iloc[1:].reset_index(drop=True)

        return df

    except Exception as exc:
        raise ValueError(f"Error while checking/removing fake header row: {exc}") from exc



def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common text placeholders for missing values into np.nan.

    Many real-world CSVs store missing values using text tokens rather than true
    blanks, and those tokens can break numeric conversion or model training.
    """
    try:
        cleaned = df.copy()
        replacement_map = {
            "": np.nan,
            " ": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
            "n/a": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
            "NULL": np.nan,
            "null": np.nan,
            "None": np.nan,
            "none": np.nan,
            "?": np.nan,
        }

        for col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            cleaned[col] = cleaned[col].replace(replacement_map)

        return cleaned
    except Exception as exc:
        raise ValueError(f"Failed to normalize missing markers: {exc}") from exc



def convert_datetime_like_columns(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Convert datetime-like string columns into numeric UNIX timestamps.

    To avoid unnecessary parsing work, the function first checks whether a column
    looks plausibly datetime-like (for example containing '-', ':', '/', or 'T').
    This keeps the script faster and avoids noisy pandas warnings.
    """
    try:
        converted = df.copy()

        for col in converted.columns:
            series = converted[col]
            if series.dtype != object:
                continue

            non_null = series.dropna()
            if non_null.empty:
                continue

            # Quick pattern gate: do not attempt expensive datetime parsing unless
            # the sample values look somewhat date/time-like.
            sample_as_text = non_null.astype(str).head(25)
            looks_datetime_like = sample_as_text.str.contains(r"[-/:T]", regex=True).mean() >= 0.50
            if not looks_datetime_like:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                parsed = pd.to_datetime(non_null, errors="coerce")
            success_ratio = parsed.notna().mean()

            if success_ratio >= threshold:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    parsed_full = pd.to_datetime(converted[col], errors="coerce")

                # Convert ns since epoch to seconds for better numerical scale.
                converted[col] = parsed_full.astype("int64") / 1e9
                converted.loc[parsed_full.isna(), col] = np.nan
                print(f"Converted datetime-like column to numeric timestamp: {col}")

        return converted
    except Exception as exc:
        raise ValueError(f"Failed during datetime conversion: {exc}") from exc



def coerce_numeric_like_columns(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Convert string columns that are mostly numeric into numeric dtype.

    This prevents hidden string values from leaking into numeric preprocessing,
    which is a common cause of model-training failures.
    """
    try:
        converted = df.copy()

        for col in converted.columns:
            if pd.api.types.is_numeric_dtype(converted[col]):
                continue

            non_null = converted[col].dropna()
            if non_null.empty:
                continue

            coerced = pd.to_numeric(non_null, errors="coerce")
            success_ratio = coerced.notna().mean()

            if success_ratio >= threshold:
                converted[col] = pd.to_numeric(converted[col], errors="coerce")
                print(f"Converted numeric-like column to numeric: {col}")

        return converted
    except Exception as exc:
        raise ValueError(f"Failed while coercing numeric-like columns: {exc}") from exc



def clean_target_column(y: pd.Series) -> pd.Series:
    """Clean and normalize the target label column."""
    try:
        y_clean = y.copy()
        y_clean = y_clean.apply(lambda x: x.strip() if isinstance(x, str) else x)
        y_clean = y_clean.replace({
            "": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
            "NULL": np.nan,
            "None": np.nan,
            "?": np.nan,
        })
        return y_clean
    except Exception as exc:
        raise ValueError(f"Failed to clean target column: {exc}") from exc



def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into features and target using the last column as target.

    Raises clear validation errors if the dataset is not suitable for
    supervised classification.
    """
    try:
        if df.shape[1] < 2:
            raise ValueError(
                "The dataset must contain at least one feature column and one target column."
            )

        X = df.iloc[:, :-1].copy()
        y = clean_target_column(df.iloc[:, -1].copy())

        # Drop rows whose target label is missing.
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("No usable rows remain after removing missing target values.")

        unique_classes = pd.Series(y).nunique(dropna=True)
        if unique_classes < 2:
            raise ValueError(
                "The target column contains fewer than 2 classes after cleaning. "
                "A classifier cannot be trained."
            )

        return X, y
    except Exception as exc:
        raise ValueError(f"Failed to prepare features and target: {exc}") from exc


# =============================================================================
# Preprocessing
# =============================================================================

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a mixed-type preprocessing pipeline.

    Numeric columns:
    - median imputation
    - standard scaling

    Categorical columns:
    - most-frequent imputation
    - one-hot encoding
    """
    try:
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if not numeric_features and not categorical_features:
            raise ValueError("No features were detected after preprocessing inspection.")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor, numeric_features, categorical_features

    except TypeError:
        # Older scikit-learn versions use sparse=False instead of sparse_output=False.
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor, numeric_features, categorical_features

    except Exception as exc:
        raise ValueError(f"Failed to build preprocessor: {exc}") from exc


# =============================================================================
# PyTorch ANN model
# =============================================================================

class TabularANN(nn.Module):
    """
    A simple fully connected neural network for tabular classification.

    For binary classification, the network outputs one logit.
    For multiclass classification, it outputs one logit per class.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"Invalid input_dim: {input_dim}")
        if num_classes < 2:
            raise ValueError(f"Invalid num_classes: {num_classes}")

        output_dim = 1 if num_classes == 2 else num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)



def build_model(input_dim: int, num_classes: int, device: torch.device) -> TabularANN:
    """Instantiate the ANN and move it to the requested device."""
    try:
        model = TabularANN(input_dim=input_dim, num_classes=num_classes)
        model.to(device)
        return model
    except Exception as exc:
        raise ValueError(f"Failed to build ANN model: {exc}") from exc



def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convert NumPy arrays into PyTorch DataLoaders.

    DataLoaders provide batching and shuffling support for efficient training.
    """
    try:
        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("Training and validation features must be 2D arrays.")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        # Binary classification uses float targets for BCEWithLogitsLoss.
        # Multiclass classification uses int64 targets for CrossEntropyLoss.
        y_train_tensor = torch.tensor(y_train)
        y_val_tensor = torch.tensor(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    except Exception as exc:
        raise RuntimeError(f"Failed to create DataLoaders: {exc}") from exc



def train_model(
    model: TabularANN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    patience: int = 5,
) -> Dict[str, List[float]]:
    """
    Train the PyTorch model with early stopping.

    Early stopping monitors validation loss and restores the best model weights
    seen during training.
    """
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion: nn.Module

        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            # -----------------------------
            # Training phase
            # -----------------------------
            model.train()
            running_train_loss = 0.0
            train_examples = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)

                if num_classes == 2:
                    batch_y = batch_y.float().view(-1, 1)
                    loss = criterion(logits, batch_y)
                else:
                    batch_y = batch_y.long()
                    loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                batch_size_actual = batch_x.size(0)
                running_train_loss += loss.item() * batch_size_actual
                train_examples += batch_size_actual

            train_loss = running_train_loss / max(train_examples, 1)

            # -----------------------------
            # Validation phase
            # -----------------------------
            model.eval()
            running_val_loss = 0.0
            val_examples = 0
            all_val_preds: List[int] = []
            all_val_true: List[int] = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)

                    if num_classes == 2:
                        batch_y_float = batch_y.float().view(-1, 1)
                        loss = criterion(logits, batch_y_float)
                        preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
                    else:
                        batch_y_long = batch_y.long()
                        loss = criterion(logits, batch_y_long)
                        preds = torch.argmax(logits, dim=1)

                    batch_size_actual = batch_x.size(0)
                    running_val_loss += loss.item() * batch_size_actual
                    val_examples += batch_size_actual

                    all_val_preds.extend(preds.cpu().numpy().astype(int).tolist())
                    all_val_true.extend(batch_y.cpu().numpy().astype(int).tolist())

            val_loss = running_val_loss / max(val_examples, 1)
            val_accuracy = accuracy_score(all_val_true, all_val_preds) if all_val_true else 0.0

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_accuracy))

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_accuracy={val_accuracy:.6f}"
            )

            # -----------------------------
            # Early stopping check
            # -----------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})."
                )
                break

        if best_state is None:
            raise RuntimeError("Training completed, but no valid best model state was captured.")

        model.load_state_dict(best_state)
        return history

    except Exception as exc:
        raise RuntimeError(f"Failed during PyTorch training: {exc}") from exc



def predict_classes(model: TabularANN, X: np.ndarray, num_classes: int, device: torch.device) -> np.ndarray:
    """Generate class predictions from a trained model."""
    try:
        if X.ndim != 2:
            raise ValueError("Prediction features must be a 2D array.")

        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            logits = model(x_tensor)

            if num_classes == 2:
                preds = (torch.sigmoid(logits).view(-1) >= 0.5).long()
            else:
                preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy().astype(int)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate predictions: {exc}") from exc


# =============================================================================
# Training orchestration
# =============================================================================

def train_and_evaluate(df: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    """
    Train the PyTorch ANN and return training artifacts and evaluation metrics.

    This function centralizes the full ML workflow so errors can be caught and
    reported with full traceback context.
    """
    try:
        print_section("Preparing Features and Target")
        X, y = prepare_features_and_target(df)

        print(f"Rows after target cleaning: {len(X)}")
        print(f"Feature columns: {X.shape[1]}")
        print(f"Target classes (raw): {sorted(pd.Series(y).astype(str).unique().tolist())}")

        print_section("Encoding Target Labels")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
        class_names = label_encoder.classes_.tolist()
        num_classes = len(class_names)
        print(f"Encoded classes: {class_names}")

        print_section("Train/Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows:  {len(X_test)}")

        print_section("Building Preprocessor")
        preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        print_section("Fitting Preprocessor")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # The PyTorch model expects dense float32 arrays.
        X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
        X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

        if X_train_processed.size == 0:
            raise RuntimeError("Processed training feature matrix is empty.")
        if np.isnan(X_train_processed).all():
            raise RuntimeError("All processed training values are NaN after preprocessing.")

        # Create a validation split from the training portion only.
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_processed,
            y_train,
            test_size=0.20,
            random_state=42,
            stratify=y_train,
        )

        # Convert targets to the right dtype for the chosen loss function.
        if num_classes == 2:
            y_train_final = np.asarray(y_train_final, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            y_test_model = np.asarray(y_test, dtype=np.int64)
        else:
            y_train_final = np.asarray(y_train_final, dtype=np.int64)
            y_val = np.asarray(y_val, dtype=np.int64)
            y_test_model = np.asarray(y_test, dtype=np.int64)

        input_dim = X_train_processed.shape[1]
        print(f"Processed input dimension: {input_dim}")

        print_section("Preparing PyTorch DataLoaders")
        train_loader, val_loader = make_dataloaders(
            X_train=X_train_final,
            y_train=y_train_final,
            X_val=X_val,
            y_val=y_val,
            batch_size=32,
        )

        print_section("Building PyTorch ANN Model")
        model = build_model(input_dim=input_dim, num_classes=num_classes, device=device)
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Using device: {device}")

        print_section("Training Model")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            epochs=30,
            learning_rate=1e-3,
            patience=5,
        )

        print_section("Evaluating Model")
        y_pred = predict_classes(model, X_test_processed, num_classes=num_classes, device=device)

        accuracy = accuracy_score(y_test_model, y_pred)
        cm = confusion_matrix(y_test_model, y_pred)
        report = classification_report(
            y_test_model,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.6f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)

        return {
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "history": history,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report,
            "class_names": class_names,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "device": str(device),
        }

    except Exception as exc:
        raise RuntimeError(
            "Unexpected model training error: "
            f"{exc}\n\nFull traceback:\n{traceback.format_exc()}"
        ) from exc


# =============================================================================
# Saving artifacts
# =============================================================================

def save_artifacts(
    model: TabularANN,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    output_dir: Path,
    input_dim: int,
    num_classes: int,
) -> None:
    """
    Save the PyTorch checkpoint, fitted preprocessor, and label metadata.

    Why we save multiple files:
    - The PyTorch model only knows about already-processed numeric arrays.
    - The preprocessor is needed to transform future raw CSV rows consistently.
    - The label mapping is needed to convert predicted indices back to class names.
    """
    try:
        model_path = output_dir / "ais_ann_model.pt"
        preprocessor_path = output_dir / "ais_ann_preprocessor.pkl"
        mapping_path = output_dir / "ais_ann_label_mapping.json"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "model_class": "TabularANN",
            "pytorch_version": torch.__version__,
        }
        torch.save(checkpoint, model_path)

        joblib.dump(preprocessor, preprocessor_path)

        label_mapping = {
            "classes": label_encoder.classes_.tolist(),
            "class_to_index": {cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)},
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)

        print_section("Artifacts Saved")
        print(f"Model checkpoint saved to: {model_path}")
        print(f"Preprocessor saved to:     {preprocessor_path}")
        print(f"Labels saved to:           {mapping_path}")

    except Exception as exc:
        raise IOError(f"Failed to save artifacts: {exc}") from exc


# =============================================================================
# Main program
# =============================================================================

def main() -> int:
    """Main program entry point."""
    try:
        print_section("Initializing")
        set_reproducible_seed(42)
        device = get_device()
        print(f"PyTorch version: {torch.__version__}")
        print(f"Selected device: {device}")

        print_section("Resolving Dataset Path")
        csv_path = resolve_dataset_path("AIS_2023_MINI.csv")
        print(f"Dataset path: {csv_path}")

        print_section("Loading Dataset")
        df = load_csv_without_assumptions(csv_path)
        print(f"Raw shape: {df.shape}")

        print_section("Cleaning Dataset")
        df = remove_fake_header_row_if_present(df)
        df = normalize_missing_markers(df)
        df = convert_datetime_like_columns(df)
        df = coerce_numeric_like_columns(df)
        print(f"Cleaned shape: {df.shape}")
        print("Column dtypes after cleaning:")
        print(df.dtypes)

        print_section("Training and Evaluation")
        results = train_and_evaluate(df, device=device)

        print_section("Saving Outputs")
        save_artifacts(
            model=results["model"],
            preprocessor=results["preprocessor"],
            label_encoder=results["label_encoder"],
            output_dir=Path(__file__).resolve().parent,
            input_dim=results["input_dim"],
            num_classes=results["num_classes"],
        )

        print_section("Done")
        print("PyTorch ANN training script completed successfully.")
        return 0

    except FileNotFoundError as exc:
        print_section("ERROR: Dataset Not Found")
        print(exc)
        return 1
    except ImportError as exc:
        print_section("ERROR: Missing Dependency")
        print(exc)
        print("Install required packages with:")
        print("pip install torch pandas numpy scikit-learn joblib")
        return 2
    except ValueError as exc:
        print_section("ERROR: Data Validation Problem")
        print(exc)
        return 3
    except RuntimeError as exc:
        print_section("ERROR: Model Training Failure")
        print(exc)
        return 4
    except IOError as exc:
        print_section("ERROR: Output Saving Failure")
        print(exc)
        return 5
    except MemoryError as exc:
        print_section("ERROR: Out of Memory")
        print(
            "The process ran out of memory. Consider reducing model size, "
            "batch size, or dataset size."
        )
        print(exc)
        return 6
    except KeyboardInterrupt:
        print_section("Interrupted")
        print("Training was interrupted by the user.")
        return 130
    except Exception as exc:
        print_section("ERROR: Unexpected Failure")
        print(f"{exc}\n\nTraceback:\n{traceback.format_exc()}")
        return 7


if __name__ == "__main__":
    # Optional CPU-only override if the user wants to avoid CUDA issues.
    # To force CPU execution, uncomment the next line before running:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.exit(main())
