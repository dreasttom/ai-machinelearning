#!/usr/bin/env python3
"""
analyze_radar_signals.py

Purpose
-------
Train and evaluate a small TensorFlow / Keras neural-network classifier on the
radar_signals.csv dataset. The script assumes the CSV file is in the SAME
folder as this script unless a custom path is provided on the command line.

What this script does
---------------------
1. Loads and validates the dataset.
2. Cleans obvious issues (duplicate rows, blank labels, non-numeric values).
3. Prints an exploratory summary so the user can inspect the data.
4. Computes per-class feature means for quick analysis.
5. Uses Leave-One-Out Cross-Validation (LOOCV) with a Keras neural network.
6. Prints a confusion matrix and overall accuracy.
7. Trains one final model on the full cleaned dataset.
8. Saves CSV outputs plus the final Keras model and its metadata.

Why Leave-One-Out?
------------------
The dataset is extremely small. LOOCV lets us evaluate the model on every row
while still training on as much data as possible in each fold.

Important note
--------------
This script REQUIRES TensorFlow (which includes Keras).
If it is not installed, the script exits with a clear message.

Example usage
-------------
python analyze_radar_signals.py
python analyze_radar_signals.py radar_signals.csv
python analyze_radar_signals.py /full/path/to/radar_signals.csv

Recommended installation
------------------------
pip install tensorflow pandas numpy
"""

from __future__ import annotations

import json
import os
import random
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration / constants
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_NAME = "radar_signals.csv"
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, DEFAULT_CSV_NAME)

# Expected columns in the dataset. We fail early if any are missing because
# later code depends on them being present.
REQUIRED_COLUMNS = [
    "pulse_id",
    "frequency_mhz",
    "amplitude_db",
    "pulse_width_us",
    "pri_us",
    "doppler_hz",
    "snr_db",
    "range_km",
    "azimuth_deg",
    "elevation_deg",
    "label",
]

# Numeric feature columns used as model inputs.
NUMERIC_FEATURES = [
    "frequency_mhz",
    "amplitude_db",
    "pulse_width_us",
    "pri_us",
    "doppler_hz",
    "snr_db",
    "range_km",
    "azimuth_deg",
    "elevation_deg",
]

# Reproducibility: we set the same seed for Python, NumPy, and TensorFlow.
RANDOM_SEED = 42

# Keras model training configuration.
EPOCHS_PER_FOLD = 150
EPOCHS_FINAL_MODEL = 200
BATCH_SIZE = 4
LEARNING_RATE = 0.001
PATIENCE = 15


# -----------------------------------------------------------------------------
# Utility / helper functions
# -----------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Pretty-print section headers so console output is easier to read."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def fail(message: str, exit_code: int = 1) -> None:
    """Print a consistent error message and exit with a specific code."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(exit_code)



def import_tensorflow():
    """
    Import TensorFlow only when needed so we can provide a very clear error
    message if it is missing.

    Returns:
        The imported tensorflow module.

    Raises:
        ImportError: if TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed. Install it first with:\n"
            "    pip install tensorflow\n"
            "Then run the script again."
        ) from exc
    except Exception as exc:
        raise ImportError(
            f"TensorFlow could not be imported due to an unexpected error: {exc}"
        ) from exc

    return tf



def set_global_seed(tf_module) -> None:
    """
    Set seeds across the main randomness sources used in this script.

    Why this matters:
    - Neural networks are randomized (initial weights, shuffling, etc.).
    - Fixing seeds makes results much more reproducible.
    """
    try:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        tf_module.random.set_seed(RANDOM_SEED)
    except Exception as exc:
        # This should not normally fail, but we surface a useful error if it does.
        raise RuntimeError(f"Failed to set random seeds: {exc}") from exc



def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file with defensive error handling.

    Raises:
        FileNotFoundError: if the CSV path does not exist.
        ValueError: if the CSV is empty or malformed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at: {csv_path}\n"
            f"Expected the CSV to be in the same folder as this script, or pass a custom path."
        )

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file exists but contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The CSV file could not be parsed as a valid CSV.") from exc
    except Exception as exc:
        raise ValueError(f"Unexpected error while reading CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV loaded, but it contains zero rows.")

    return df



def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Ensure the dataframe contains every required column.

    Raises:
        ValueError: if any required column is missing.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean obvious issues while preserving the original dataset as much as possible.

    Cleaning steps:
    - Remove exact duplicate rows.
    - Strip whitespace from string fields.
    - Drop rows with blank or missing labels.
    - Coerce numeric feature columns to numeric.
    - Drop rows with invalid or missing numeric feature values.

    Returns:
        A cleaned copy of the dataframe.
    """
    cleaned = df.copy()

    # Remove exact duplicate rows because duplicate training examples can distort
    # performance metrics on tiny datasets.
    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    duplicates_removed = before_duplicates - len(cleaned)
    if duplicates_removed > 0:
        print(f"Note: Removed {duplicates_removed} exact duplicate row(s).")

    # Standardize string columns. This helps avoid accidental mismatches such as
    # ' friendly_aircraft' versus 'friendly_aircraft'.
    for column in ["pulse_id", "label"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    # Convert empty strings and stringified NaN values into real missing values,
    # then remove rows that do not have a usable target label.
    cleaned["label"] = cleaned["label"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    before_label_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["label"])
    labels_dropped = before_label_drop - len(cleaned)
    if labels_dropped > 0:
        print(f"Note: Dropped {labels_dropped} row(s) with missing labels.")

    # Convert all numeric feature columns to actual numeric values. If a value
    # cannot be converted (for example a stray string), it becomes NaN.
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Remove rows that still have invalid numeric values after coercion.
    before_numeric_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=NUMERIC_FEATURES)
    invalid_numeric_dropped = before_numeric_drop - len(cleaned)
    if invalid_numeric_dropped > 0:
        print(
            f"Note: Dropped {invalid_numeric_dropped} row(s) due to missing/invalid numeric values."
        )

    # Check for repeated pulse IDs. This may be legitimate in some datasets, so we
    # warn instead of automatically deleting them.
    if cleaned["pulse_id"].duplicated().any():
        duplicate_ids = cleaned.loc[
            cleaned["pulse_id"].duplicated(), "pulse_id"
        ].tolist()
        print(
            "Warning: Duplicate pulse_id values detected: "
            + ", ".join(map(str, duplicate_ids))
        )

    if cleaned.empty:
        raise ValueError("No usable rows remain after cleaning.")

    # Reset the index so downstream code has a clean, predictable integer index.
    cleaned = cleaned.reset_index(drop=True)
    return cleaned



def dataset_overview(df: pd.DataFrame) -> None:
    """Print a readable overview of the cleaned dataset."""
    print_section("DATASET OVERVIEW")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print_section("FIRST 5 ROWS")
    print(df.head().to_string(index=False))

    print_section("LABEL DISTRIBUTION")
    print(df["label"].value_counts(dropna=False).sort_index().to_string())

    print_section("MISSING VALUES BY COLUMN")
    print(df.isna().sum().to_string())



def summarize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a descriptive statistics table for the numeric feature columns.
    """
    summary = df[NUMERIC_FEATURES].describe().T
    summary["variance"] = df[NUMERIC_FEATURES].var()
    summary["missing_count"] = df[NUMERIC_FEATURES].isna().sum()
    return summary



def compute_class_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class means for each numeric feature.

    This is useful for inspection even though the final classifier is a neural
    network instead of a nearest-centroid model.
    """
    return df.groupby("label")[NUMERIC_FEATURES].mean().sort_index()



def encode_labels(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Convert string labels into integer indices for Keras.

    Keras classification models expect numeric targets. We build stable,
    alphabetically sorted label mappings so results remain deterministic.
    """
    unique_labels = sorted(df["label"].unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError(
            "The dataset must contain at least two distinct label classes for classification."
        )

    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label



def zscore_standardize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Standardize training and test data using *only* training-set statistics.

    This is extremely important because using information from the test sample
    to scale the training data would cause data leakage.

    Returns:
        train_scaled_df,
        test_scaled_df,
        scaler_metadata
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    means = train_df[features].mean()
    stds = train_df[features].std(ddof=0).replace(0, 1.0)

    train_scaled[features] = (train_df[features] - means) / stds
    test_scaled[features] = (test_df[features] - means) / stds

    scaler_metadata = {
        "means": {feature: float(means[feature]) for feature in features},
        "stds": {feature: float(stds[feature]) for feature in features},
    }
    return train_scaled, test_scaled, scaler_metadata



def build_keras_model(tf_module, input_dim: int, num_classes: int):
    """
    Build and compile a small feed-forward neural network.

    Design notes:
    - Dense layers are a good fit for small tabular datasets.
    - ReLU is a common activation for hidden layers.
    - Softmax produces class probabilities for multi-class classification.
    - Sparse categorical cross-entropy lets us use integer labels directly.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")

    keras = tf_module.keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def train_model(
    tf_module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """
    Train a Keras model with conservative settings and early stopping.

    We monitor training loss rather than validation loss because the dataset is so
    small that holding out a validation split inside each LOOCV fold would leave
    too few samples for actual training.
    """
    keras = tf_module.keras

    if len(x_train) == 0:
        raise ValueError("Training data is empty; cannot train model.")

    # Batch size should never exceed the number of training examples.
    effective_batch_size = max(1, min(batch_size, len(x_train)))

    model = build_keras_model(
        tf_module=tf_module,
        input_dim=x_train.shape[1],
        num_classes=len(np.unique(y_train)),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=effective_batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    return model, history



def run_leave_one_out_keras(
    df: pd.DataFrame,
    tf_module,
    label_to_index: Dict[str, int],
    index_to_label: Dict[int, str],
) -> pd.DataFrame:
    """
    Run Leave-One-Out Cross-Validation using a TensorFlow / Keras model.

    For each row:
    - use that row as the test sample
    - train on all remaining rows
    - predict the held-out row

    Returns:
        DataFrame with pulse_id, actual label, predicted label, correctness, and
        model confidence for the predicted class.
    """
    results = []

    for test_index in df.index:
        test_df = df.loc[[test_index]].copy()
        train_df = df.drop(index=test_index).copy()

        # Standardize train/test separately using only training statistics.
        train_scaled, test_scaled, _ = zscore_standardize_train_test(
            train_df=train_df,
            test_df=test_df,
            features=NUMERIC_FEATURES,
        )

        x_train = train_scaled[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
        x_test = test_scaled[NUMERIC_FEATURES].to_numpy(dtype=np.float32)

        y_train = train_scaled["label"].map(label_to_index).to_numpy(dtype=np.int32)
        actual_label = test_scaled.iloc[0]["label"]
        actual_index = label_to_index[actual_label]

        try:
            model, history = train_model(
                tf_module=tf_module,
                x_train=x_train,
                y_train=y_train,
                epochs=EPOCHS_PER_FOLD,
                batch_size=BATCH_SIZE,
            )

            probabilities = model.predict(x_test, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_label = index_to_label[predicted_index]
            predicted_confidence = float(probabilities[predicted_index])

            # Final training loss can be useful for quick debugging.
            final_training_loss = float(history.history["loss"][-1])
            final_training_accuracy = float(history.history["accuracy"][-1])

        except Exception as exc:
            # We do not want one bad fold to destroy the entire analysis.
            predicted_label = f"PREDICTION_ERROR: {exc}"
            predicted_confidence = np.nan
            final_training_loss = np.nan
            final_training_accuracy = np.nan
            predicted_index = None

        results.append(
            {
                "pulse_id": test_scaled.iloc[0]["pulse_id"],
                "actual_label": actual_label,
                "actual_index": actual_index,
                "predicted_label": predicted_label,
                "predicted_index": predicted_index,
                "predicted_confidence": predicted_confidence,
                "correct": actual_label == predicted_label,
                "final_training_loss": final_training_loss,
                "final_training_accuracy": final_training_accuracy,
            }
        )

    return pd.DataFrame(results)



def confusion_matrix_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a confusion matrix from actual versus predicted labels.

    We use pandas.crosstab here to avoid needing another dependency.
    """
    return pd.crosstab(
        results_df["actual_label"],
        results_df["predicted_label"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )



def zscore_standardize_full_dataset(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Standardize the full dataset so we can train a final deployable model.

    This differs from the train/test standardization helper because here we are
    intentionally fitting on the entire cleaned dataset to create the final model
    artifact for later reuse.
    """
    scaled = df.copy()
    means = df[features].mean()
    stds = df[features].std(ddof=0).replace(0, 1.0)
    scaled[features] = (df[features] - means) / stds

    scaler_metadata = {
        "means": {feature: float(means[feature]) for feature in features},
        "stds": {feature: float(stds[feature]) for feature in features},
    }
    return scaled, scaler_metadata



def train_final_model(
    df: pd.DataFrame,
    tf_module,
    label_to_index: Dict[str, int],
) -> Tuple[object, Dict[str, Dict[str, float]]]:
    """
    Train one final model on the full cleaned dataset.

    This is separate from cross-validation. The cross-validation estimates
    performance; the final model is the one you would actually save and reuse.
    """
    full_scaled, scaler_metadata = zscore_standardize_full_dataset(
        df=df,
        features=NUMERIC_FEATURES,
    )

    x_full = full_scaled[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    y_full = full_scaled["label"].map(label_to_index).to_numpy(dtype=np.int32)

    model, _ = train_model(
        tf_module=tf_module,
        x_train=x_full,
        y_train=y_full,
        epochs=EPOCHS_FINAL_MODEL,
        batch_size=BATCH_SIZE,
    )
    return model, scaler_metadata



def write_dataframe_csv(df: pd.DataFrame, output_path: str, description: str, index: bool = True) -> None:
    """Safely write a dataframe to CSV and report any errors clearly."""
    try:
        df.to_csv(output_path, index=index)
        print(f"Saved {description}: {output_path}")
    except PermissionError as exc:
        print(
            f"Warning: Could not write {description} due to a permission error: {exc}",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"Warning: OS error while writing {description}: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"Warning: Unexpected error while writing {description}: {exc}",
            file=sys.stderr,
        )



def write_json(data: dict, output_path: str, description: str) -> None:
    """Safely write JSON metadata to disk."""
    try:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=2)
        print(f"Saved {description}: {output_path}")
    except PermissionError as exc:
        print(
            f"Warning: Could not write {description} due to a permission error: {exc}",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"Warning: OS error while writing {description}: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"Warning: Unexpected error while writing {description}: {exc}",
            file=sys.stderr,
        )



def save_keras_model(model, output_path: str) -> None:
    """
    Save the trained Keras model in the native .keras format.
    """
    try:
        model.save(output_path)
        print(f"Saved final Keras model: {output_path}")
    except PermissionError as exc:
        print(
            f"Warning: Could not save final Keras model due to a permission error: {exc}",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"Warning: OS error while saving final Keras model: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"Warning: Unexpected error while saving final Keras model: {exc}",
            file=sys.stderr,
        )



def main() -> None:
    """
    Main program entry point.

    We wrap the workflow in a top-level try/except so the user gets a readable
    failure message plus a traceback for debugging if something unexpected occurs.
    """
    try:
        print_section("RADAR SIGNAL ANALYSIS WITH TENSORFLOW / KERAS")

        # Allow an optional custom CSV path; otherwise default to the CSV in the
        # same folder as the script.
        csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
        print(f"Using dataset: {csv_path}")

        # Import TensorFlow only once we know we actually need it.
        tf_module = import_tensorflow()
        set_global_seed(tf_module)
        print(f"TensorFlow version: {tf_module.__version__}")

        # 1) Load and validate the raw data.
        df = load_dataset(csv_path)
        validate_columns(df, REQUIRED_COLUMNS)

        # 2) Clean the data.
        df = clean_dataset(df)

        # 3) Print summary information for human inspection.
        dataset_overview(df)

        numeric_summary = summarize_numeric_features(df)
        print_section("NUMERIC FEATURE SUMMARY")
        print(numeric_summary.to_string())

        class_centroids = compute_class_centroids(df)
        print_section("PER-CLASS FEATURE MEANS")
        print(class_centroids.to_string())

        # 4) Encode labels once so the mapping is consistent across folds.
        label_to_index, index_to_label = encode_labels(df)
        print_section("LABEL ENCODING")
        for label, index in label_to_index.items():
            print(f"{label!r} -> {index}")

        # 5) Run LOOCV using Keras.
        results_df = run_leave_one_out_keras(
            df=df,
            tf_module=tf_module,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
        )

        print_section("LEAVE-ONE-OUT KERAS PREDICTIONS")
        print(results_df.to_string(index=False))

        accuracy = float(results_df["correct"].mean())
        print_section("LOOCV ACCURACY")
        print(f"Accuracy: {accuracy:.2%}")

        conf_matrix = confusion_matrix_table(results_df)
        print_section("CONFUSION MATRIX")
        print(conf_matrix.to_string())

        # 6) Train one final model on the entire cleaned dataset.
        final_model, scaler_metadata = train_final_model(
            df=df,
            tf_module=tf_module,
            label_to_index=label_to_index,
        )

        # 7) Save all outputs.
        write_dataframe_csv(
            numeric_summary,
            os.path.join(SCRIPT_DIR, "radar_summary_statistics.csv"),
            "summary statistics CSV",
        )
        write_dataframe_csv(
            class_centroids,
            os.path.join(SCRIPT_DIR, "radar_class_centroids.csv"),
            "class centroid CSV",
        )
        write_dataframe_csv(
            results_df.set_index("pulse_id"),
            os.path.join(SCRIPT_DIR, "radar_predictions_leave_one_out_keras.csv"),
            "Keras leave-one-out prediction CSV",
        )
        write_dataframe_csv(
            conf_matrix,
            os.path.join(SCRIPT_DIR, "radar_confusion_matrix_keras.csv"),
            "Keras confusion matrix CSV",
        )

        metadata = {
            "features": NUMERIC_FEATURES,
            "label_to_index": label_to_index,
            "index_to_label": {str(key): value for key, value in index_to_label.items()},
            "scaler": scaler_metadata,
            "random_seed": RANDOM_SEED,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "epochs_final_model": EPOCHS_FINAL_MODEL,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PATIENCE,
        }
        write_json(
            metadata,
            os.path.join(SCRIPT_DIR, "radar_keras_model_metadata.json"),
            "Keras model metadata JSON",
        )
        save_keras_model(
            final_model,
            os.path.join(SCRIPT_DIR, "radar_signal_classifier.keras"),
        )

        print_section("DONE")
        print("TensorFlow / Keras analysis completed successfully.")

    except FileNotFoundError as exc:
        fail(str(exc), exit_code=2)

    except ImportError as exc:
        fail(str(exc), exit_code=4)

    except ValueError as exc:
        fail(str(exc), exit_code=3)

    except KeyboardInterrupt:
        fail("Execution interrupted by user.", exit_code=130)

    except Exception as exc:
        print("An unexpected error occurred.", file=sys.stderr)
        print(f"Error type: {type(exc).__name__}", file=sys.stderr)
        print(f"Error details: {exc}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(99)


if __name__ == "__main__":
    main()
