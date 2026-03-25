#!/usr/bin/env python3
"""
radar_distance_gui.py

A heavily commented Tkinter GUI application for exploring multiple distance
metrics on the same radar_signals.csv dataset created earlier.

The GUI provides a drop-down menu so the user can choose one of these metrics:
    - Euclidean distance
    - Haversine difference
    - Minkowski distance
    - Mahalanobis distance
    - Manhattan distance

Why a GUI?
----------
The user explicitly asked for a "drop down menu", so this script uses Tkinter,
which is included with standard Python installations on many systems.

Important note about "Haversine difference"
-------------------------------------------
The classic haversine formula is normally used for distances between two points
on a sphere using latitude/longitude coordinates.

This radar dataset does NOT contain latitude/longitude. It does contain:
    - azimuth_deg
    - elevation_deg

So, in this script, the "Haversine difference" option computes an angular
separation using:
    - azimuth_deg as a longitude-like angle
    - elevation_deg as a latitude-like angle

That makes it a practical spherical-angle comparison for this dataset, even
though it is not a geodesic over Earth coordinates.

Features
--------
1. Loads and validates radar_signals.csv
2. Provides a GUI drop-down for metric selection
3. Lets the user choose a reference pulse ID
4. Computes distances from the reference pulse to all other pulses
5. Displays nearest-neighbor style results in a table
6. Includes robust error handling and clear user-facing messages
7. Uses heavy comments for readability and maintainability

Usage
-----
    python radar_distance_gui.py

Optional:
    python radar_distance_gui.py --dataset /path/to/radar_signals.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as exc:
    print(
        "ERROR: Tkinter is required for this GUI script but could not be imported.\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import numpy as np
except Exception as exc:
    print(
        "ERROR: NumPy is required for this script but could not be imported.\n"
        f"Details: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# =============================================================================
# Custom exception hierarchy
# =============================================================================
# A small exception hierarchy makes it easier to:
# - distinguish expected user/data issues from unexpected programming issues
# - show cleaner error messages
# - keep the main GUI code readable
class RadarAppError(Exception):
    """Base exception for this application."""


class DatasetError(RadarAppError):
    """Raised when the dataset is missing, malformed, or invalid."""


class DistanceMetricError(RadarAppError):
    """Raised when a distance metric cannot be computed safely."""


class GUIStateError(RadarAppError):
    """Raised when the GUI state is not ready for a requested action."""


# =============================================================================
# Data model
# =============================================================================
# Each row in the CSV becomes one RadarPulse object. A dataclass keeps the code
# explicit, structured, and easy to reason about.
@dataclass
class RadarPulse:
    pulse_id: str
    frequency_mhz: float
    amplitude_db: float
    pulse_width_us: float
    pri_us: float
    doppler_hz: float
    snr_db: float
    range_km: float
    azimuth_deg: float
    elevation_deg: float
    label: str

    def numeric_vector(self) -> np.ndarray:
        """
        Return the full numeric feature vector in a consistent order.

        This vector is used by Euclidean, Minkowski, Manhattan, and Mahalanobis.
        """
        return np.array(
            [
                self.frequency_mhz,
                self.amplitude_db,
                self.pulse_width_us,
                self.pri_us,
                self.doppler_hz,
                self.snr_db,
                self.range_km,
                self.azimuth_deg,
                self.elevation_deg,
            ],
            dtype=float,
        )

    def angular_vector_deg(self) -> Tuple[float, float]:
        """
        Return the angular components for the haversine-style comparison.

        We treat:
            elevation_deg -> latitude-like angle
            azimuth_deg   -> longitude-like angle
        """
        return (self.elevation_deg, self.azimuth_deg)


# =============================================================================
# Dataset schema
# =============================================================================
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

NUMERIC_COLUMNS = [
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

METRIC_OPTIONS = [
    "Euclidean distance",
    "Haversine difference",
    "Minkowski distance",
    "Mahalanobis distance",
    "Manhattan distance",
]


# =============================================================================
# Dataset loading and validation helpers
# =============================================================================
def validate_file_path(path: str) -> None:
    """
    Validate the dataset path before attempting to read it.

    This avoids confusing failures later in the workflow.
    """
    if not path:
        raise DatasetError("No dataset path was provided.")

    if not os.path.exists(path):
        raise DatasetError(f"Dataset file does not exist: {path}")

    if not os.path.isfile(path):
        raise DatasetError(f"Dataset path is not a file: {path}")

    if not os.access(path, os.R_OK):
        raise DatasetError(f"Dataset file is not readable: {path}")


def safe_float(value: object, column_name: str, pulse_id: str) -> float:
    """
    Convert one CSV value to float with strict validation.

    We reject blank strings, invalid text, NaN, and infinities.
    """
    if value is None:
        raise DatasetError(
            f"Missing value in column '{column_name}' for pulse '{pulse_id}'."
        )

    text = str(value).strip()
    if text == "":
        raise DatasetError(
            f"Blank value in column '{column_name}' for pulse '{pulse_id}'."
        )

    try:
        number = float(text)
    except ValueError as exc:
        raise DatasetError(
            f"Non-numeric value '{value}' in column '{column_name}' for pulse '{pulse_id}'."
        ) from exc

    if math.isnan(number) or math.isinf(number):
        raise DatasetError(
            f"Invalid numeric value '{value}' in column '{column_name}' for pulse '{pulse_id}'. "
            "NaN and infinity are not allowed."
        )

    return number


def load_radar_dataset(path: str) -> List[RadarPulse]:
    """
    Load and validate the radar CSV file.

    Returns
    -------
    List[RadarPulse]
        A validated list of RadarPulse records.
    """
    validate_file_path(path)

    pulses: List[RadarPulse] = []
    seen_ids = set()

    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            headers = reader.fieldnames or []
            missing = [col for col in REQUIRED_COLUMNS if col not in headers]
            if missing:
                raise DatasetError(
                    "Dataset is missing required columns: " + ", ".join(missing)
                )

            for row_number, row in enumerate(reader, start=2):
                if row is None:
                    raise DatasetError(f"Encountered an unreadable row at line {row_number}.")

                pulse_id = str(row.get("pulse_id", "")).strip()
                if pulse_id == "":
                    raise DatasetError(f"Missing pulse_id at line {row_number}.")

                if pulse_id in seen_ids:
                    raise DatasetError(
                        f"Duplicate pulse_id '{pulse_id}' found at line {row_number}."
                    )
                seen_ids.add(pulse_id)

                pulse = RadarPulse(
                    pulse_id=pulse_id,
                    frequency_mhz=safe_float(row.get("frequency_mhz"), "frequency_mhz", pulse_id),
                    amplitude_db=safe_float(row.get("amplitude_db"), "amplitude_db", pulse_id),
                    pulse_width_us=safe_float(row.get("pulse_width_us"), "pulse_width_us", pulse_id),
                    pri_us=safe_float(row.get("pri_us"), "pri_us", pulse_id),
                    doppler_hz=safe_float(row.get("doppler_hz"), "doppler_hz", pulse_id),
                    snr_db=safe_float(row.get("snr_db"), "snr_db", pulse_id),
                    range_km=safe_float(row.get("range_km"), "range_km", pulse_id),
                    azimuth_deg=safe_float(row.get("azimuth_deg"), "azimuth_deg", pulse_id),
                    elevation_deg=safe_float(row.get("elevation_deg"), "elevation_deg", pulse_id),
                    label=str(row.get("label", "")).strip() or "unknown",
                )
                pulses.append(pulse)

    except UnicodeDecodeError as exc:
        raise DatasetError("Dataset could not be decoded as UTF-8 text.") from exc
    except csv.Error as exc:
        raise DatasetError(f"CSV parsing failed: {exc}") from exc

    if not pulses:
        raise DatasetError("The dataset contains no data rows.")

    return pulses


# =============================================================================
# Distance metric implementations
# =============================================================================
def validate_numeric_vectors(a: np.ndarray, b: np.ndarray) -> None:
    """
    Validate that two numeric vectors are safe for distance computation.
    """
    if a is None or b is None:
        raise DistanceMetricError("Distance vectors must not be None.")

    if a.size == 0 or b.size == 0:
        raise DistanceMetricError("Distance vectors must not be empty.")

    if a.shape != b.shape:
        raise DistanceMetricError(
            f"Vector shape mismatch: {a.shape} != {b.shape}"
        )

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DistanceMetricError("Vectors contain NaN or infinity values.")


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standard Euclidean distance.

    Formula:
        sqrt(sum((a_i - b_i)^2))
    """
    validate_numeric_vectors(a, b)
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manhattan distance, also called L1 distance.

    Formula:
        sum(|a_i - b_i|)
    """
    validate_numeric_vectors(a, b)
    return float(np.sum(np.abs(a - b)))


def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3.0) -> float:
    """
    Minkowski distance with configurable order p.

    Special cases:
        p = 1 -> Manhattan
        p = 2 -> Euclidean

    Here we default to p = 3, which gives the user a distinct option that is
    not identical to Manhattan or Euclidean.
    """
    validate_numeric_vectors(a, b)

    if not isinstance(p, (int, float)):
        raise DistanceMetricError("Minkowski parameter p must be numeric.")

    if p <= 0:
        raise DistanceMetricError("Minkowski parameter p must be > 0.")

    return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))


def haversine_difference_deg(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
    radius: float = 1.0,
) -> float:
    """
    Haversine-based angular difference.

    Parameters
    ----------
    lat1_deg, lon1_deg, lat2_deg, lon2_deg : float
        Angles in degrees.
    radius : float
        The sphere radius. We default to 1.0 because this dataset does not
        contain Earth lat/lon coordinates, so we return an angular-style
        spherical separation in radius units rather than kilometers.

    Returns
    -------
    float
        Great-circle separation for the angle pair.

    Notes
    -----
    Because this radar dataset does not include geographic latitude/longitude,
    we adapt the haversine formula to:
        elevation_deg -> latitude-like angle
        azimuth_deg   -> longitude-like angle
    """
    values = [lat1_deg, lon1_deg, lat2_deg, lon2_deg, radius]
    if not all(isinstance(v, (int, float)) for v in values):
        raise DistanceMetricError("Haversine inputs must be numeric.")

    if not all(math.isfinite(v) for v in values):
        raise DistanceMetricError("Haversine inputs must be finite numbers.")

    if radius <= 0:
        raise DistanceMetricError("Haversine radius must be > 0.")

    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )

    # Numerical safety:
    # Floating point arithmetic can sometimes push the value just slightly
    # outside the valid [0, 1] range, which would break asin().
    hav = min(1.0, max(0.0, hav))

    central_angle = 2.0 * math.asin(math.sqrt(hav))
    return float(radius * central_angle)


def compute_inverse_covariance(pulses: Sequence[RadarPulse]) -> np.ndarray:
    """
    Compute a covariance inverse for Mahalanobis distance.

    Mahalanobis distance requires an inverse covariance matrix. This can fail if:
        - there are too few rows
        - the covariance matrix is singular
        - the features are perfectly collinear

    Robust handling here is important because Mahalanobis is more fragile than
    simpler distance metrics.
    """
    if not pulses:
        raise DistanceMetricError("Cannot compute covariance from an empty dataset.")

    matrix = np.array([pulse.numeric_vector() for pulse in pulses], dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise DistanceMetricError(
            "At least two pulses are required to compute covariance."
        )

    if not np.isfinite(matrix).all():
        raise DistanceMetricError("Dataset contains invalid numeric values.")

    try:
        covariance = np.cov(matrix, rowvar=False)
    except Exception as exc:
        raise DistanceMetricError(
            f"Failed to compute covariance matrix: {exc}"
        ) from exc

    if covariance.ndim != 2:
        raise DistanceMetricError("Covariance matrix has an invalid shape.")

    try:
        # Use pseudo-inverse rather than strict inverse. This is more robust if
        # the covariance matrix is near-singular or singular.
        inverse_covariance = np.linalg.pinv(covariance)
    except Exception as exc:
        raise DistanceMetricError(
            f"Failed to invert covariance matrix: {exc}"
        ) from exc

    if not np.isfinite(inverse_covariance).all():
        raise DistanceMetricError("Inverse covariance matrix contains invalid values.")

    return inverse_covariance


def mahalanobis_distance(a: np.ndarray, b: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Mahalanobis distance.

    Formula:
        sqrt((a - b)^T * inv_cov * (a - b))

    This accounts for feature covariance, unlike Euclidean distance.
    """
    validate_numeric_vectors(a, b)

    if inv_cov is None:
        raise DistanceMetricError("Inverse covariance matrix was not provided.")

    if inv_cov.ndim != 2 or inv_cov.shape[0] != inv_cov.shape[1]:
        raise DistanceMetricError("Inverse covariance matrix must be square.")

    if inv_cov.shape[0] != a.shape[0]:
        raise DistanceMetricError(
            f"Inverse covariance shape {inv_cov.shape} does not match vector length {a.shape[0]}."
        )

    if not np.isfinite(inv_cov).all():
        raise DistanceMetricError("Inverse covariance contains invalid values.")

    delta = a - b

    try:
        value = float(np.sqrt(delta.T @ inv_cov @ delta))
    except Exception as exc:
        raise DistanceMetricError(
            f"Mahalanobis computation failed: {exc}"
        ) from exc

    if not math.isfinite(value):
        raise DistanceMetricError("Mahalanobis distance produced a non-finite result.")

    return value


# =============================================================================
# Distance orchestration
# =============================================================================
def get_pulse_lookup(pulses: Sequence[RadarPulse]) -> Dict[str, RadarPulse]:
    """
    Create a lookup dictionary keyed by pulse_id.
    """
    return {pulse.pulse_id: pulse for pulse in pulses}


def compute_distances(
    pulses: Sequence[RadarPulse],
    reference_id: str,
    metric_name: str,
) -> List[Tuple[str, str, float]]:
    """
    Compute distances from one reference pulse to all other pulses.

    Returns
    -------
    List of tuples:
        (pulse_id, label, distance)

    The list is sorted ascending by distance.
    """
    if not pulses:
        raise DistanceMetricError("No pulses were provided for distance computation.")

    lookup = get_pulse_lookup(pulses)
    if reference_id not in lookup:
        raise DistanceMetricError(f"Reference pulse_id '{reference_id}' was not found.")

    if metric_name not in METRIC_OPTIONS:
        raise DistanceMetricError(f"Unsupported metric selected: {metric_name}")

    reference = lookup[reference_id]
    ref_vector = reference.numeric_vector()

    inv_cov: Optional[np.ndarray] = None
    if metric_name == "Mahalanobis distance":
        inv_cov = compute_inverse_covariance(pulses)

    results: List[Tuple[str, str, float]] = []

    for pulse in pulses:
        if pulse.pulse_id == reference_id:
            continue

        if metric_name == "Euclidean distance":
            distance = euclidean_distance(ref_vector, pulse.numeric_vector())

        elif metric_name == "Manhattan distance":
            distance = manhattan_distance(ref_vector, pulse.numeric_vector())

        elif metric_name == "Minkowski distance":
            distance = minkowski_distance(ref_vector, pulse.numeric_vector(), p=3.0)

        elif metric_name == "Mahalanobis distance":
            # inv_cov is guaranteed to be set above for this metric.
            distance = mahalanobis_distance(ref_vector, pulse.numeric_vector(), inv_cov)

        elif metric_name == "Haversine difference":
            ref_lat, ref_lon = reference.angular_vector_deg()
            pulse_lat, pulse_lon = pulse.angular_vector_deg()
            distance = haversine_difference_deg(ref_lat, ref_lon, pulse_lat, pulse_lon)

        else:
            # Defensive fallback in case the options list changes in the future.
            raise DistanceMetricError(f"Metric not implemented: {metric_name}")

        results.append((pulse.pulse_id, pulse.label, float(distance)))

    results.sort(key=lambda item: item[2])
    return results


# =============================================================================
# GUI application
# =============================================================================
class RadarDistanceApp:
    """
    Main GUI controller for the radar distance explorer.
    """

    def __init__(self, root: tk.Tk, dataset_path: str) -> None:
        self.root = root
        self.root.title("Radar Distance Explorer")
        self.root.geometry("980x620")

        self.dataset_path = dataset_path
        self.pulses: List[RadarPulse] = []

        # Tkinter variable objects make it easy to bind widget state to values.
        self.metric_var = tk.StringVar(value=METRIC_OPTIONS[0])
        self.reference_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")

        self._build_widgets()
        self._load_dataset_into_gui(self.dataset_path)

    def _build_widgets(self) -> None:
        """
        Create and arrange all GUI widgets.

        Layout goals:
        - keep the controls simple and obvious
        - make the result table easy to scan
        - surface status messages clearly
        """
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        controls.pack(fill="x", expand=False)

        # Dataset row
        ttk.Label(controls, text="Dataset:").grid(row=0, column=0, sticky="w", padx=4, pady=6)

        self.dataset_entry = ttk.Entry(controls, width=90)
        self.dataset_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=6)
        self.dataset_entry.insert(0, self.dataset_path)

        browse_button = ttk.Button(
            controls,
            text="Browse...",
            command=self._browse_dataset,
        )
        browse_button.grid(row=0, column=2, sticky="ew", padx=4, pady=6)

        reload_button = ttk.Button(
            controls,
            text="Reload Dataset",
            command=self._reload_dataset_clicked,
        )
        reload_button.grid(row=0, column=3, sticky="ew", padx=4, pady=6)

        # Reference pulse row
        ttk.Label(controls, text="Reference pulse:").grid(
            row=1, column=0, sticky="w", padx=4, pady=6
        )

        self.reference_combo = ttk.Combobox(
            controls,
            textvariable=self.reference_var,
            state="readonly",
            width=30,
        )
        self.reference_combo.grid(row=1, column=1, sticky="w", padx=4, pady=6)

        # Metric row
        ttk.Label(controls, text="Distance metric:").grid(
            row=2, column=0, sticky="w", padx=4, pady=6
        )

        self.metric_combo = ttk.Combobox(
            controls,
            textvariable=self.metric_var,
            values=METRIC_OPTIONS,
            state="readonly",
            width=30,
        )
        self.metric_combo.grid(row=2, column=1, sticky="w", padx=4, pady=6)

        compute_button = ttk.Button(
            controls,
            text="Compute Distances",
            command=self._compute_clicked,
        )
        compute_button.grid(row=2, column=2, sticky="ew", padx=4, pady=6)

        clear_button = ttk.Button(
            controls,
            text="Clear Results",
            command=self._clear_results,
        )
        clear_button.grid(row=2, column=3, sticky="ew", padx=4, pady=6)

        controls.columnconfigure(1, weight=1)

        # Explanation frame
        info = ttk.LabelFrame(main_frame, text="Metric Notes", padding=10)
        info.pack(fill="x", expand=False, pady=(10, 10))

        info_text = (
            "Euclidean: straight-line distance across all numeric radar features.\n"
            "Haversine difference: angular separation using elevation/azimuth as "
            "latitude/longitude-like angles.\n"
            "Minkowski: uses p=3 for a generalized distance measure.\n"
            "Mahalanobis: accounts for covariance structure across features.\n"
            "Manhattan: sum of absolute feature differences."
        )
        ttk.Label(info, text=info_text, justify="left").pack(anchor="w")

        # Results frame with a treeview table
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        results_frame.pack(fill="both", expand=True)

        columns = ("rank", "pulse_id", "label", "distance")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=18)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("pulse_id", text="Pulse ID")
        self.tree.heading("label", text="Label")
        self.tree.heading("distance", text="Distance")
        self.tree.column("rank", width=70, anchor="center")
        self.tree.column("pulse_id", width=120, anchor="center")
        self.tree.column("label", width=220, anchor="w")
        self.tree.column("distance", width=180, anchor="e")

        scrollbar_y = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        # Status bar at the bottom
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=6,
        )
        status_label.pack(fill="x", expand=False, pady=(10, 0))

    def _browse_dataset(self) -> None:
        """
        Open a file picker and update the dataset path entry.
        """
        chosen = filedialog.askopenfilename(
            title="Select radar dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, chosen)

    def _load_dataset_into_gui(self, dataset_path: str) -> None:
        """
        Load the dataset and refresh the reference-pulse dropdown.
        """
        try:
            self.pulses = load_radar_dataset(dataset_path)

            pulse_ids = [pulse.pulse_id for pulse in self.pulses]
            self.reference_combo["values"] = pulse_ids

            if pulse_ids:
                self.reference_var.set(pulse_ids[0])
            else:
                self.reference_var.set("")

            self.status_var.set(
                f"Loaded {len(self.pulses)} pulses from: {dataset_path}"
            )

        except DatasetError as exc:
            self.pulses = []
            self.reference_combo["values"] = []
            self.reference_var.set("")
            self.status_var.set("Dataset load failed.")
            messagebox.showerror("Dataset Error", str(exc))

        except Exception as exc:
            self.pulses = []
            self.reference_combo["values"] = []
            self.reference_var.set("")
            self.status_var.set("Unexpected dataset load failure.")
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred while loading the dataset:\n\n{exc}",
            )

    def _reload_dataset_clicked(self) -> None:
        """
        Reload the dataset using the current path in the entry widget.
        """
        dataset_path = self.dataset_entry.get().strip()
        self.dataset_path = dataset_path
        self._clear_results()
        self._load_dataset_into_gui(dataset_path)

    def _clear_results(self) -> None:
        """
        Remove all rows from the results table.
        """
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.status_var.set("Results cleared.")

    def _compute_clicked(self) -> None:
        """
        Handle the Compute Distances button.

        This function performs GUI-state validation and then delegates the
        actual math to the metric functions above.
        """
        try:
            if not self.pulses:
                raise GUIStateError("No dataset is loaded. Please load a valid CSV first.")

            reference_id = self.reference_var.get().strip()
            metric_name = self.metric_var.get().strip()

            if reference_id == "":
                raise GUIStateError("Please choose a reference pulse.")

            if metric_name == "":
                raise GUIStateError("Please choose a distance metric.")

            results = compute_distances(self.pulses, reference_id, metric_name)

            self._clear_results()

            for rank, (pulse_id, label, distance) in enumerate(results, start=1):
                self.tree.insert(
                    "",
                    "end",
                    values=(rank, pulse_id, label, f"{distance:.6f}"),
                )

            self.status_var.set(
                f"Computed {len(results)} distances using '{metric_name}' from reference '{reference_id}'."
            )

        except (GUIStateError, DistanceMetricError) as exc:
            messagebox.showerror("Computation Error", str(exc))
            self.status_var.set("Computation failed.")

        except Exception as exc:
            # We provide a concise user-facing message while also capturing the
            # traceback to stderr for debugging if the script is run from a shell.
            traceback.print_exc()
            messagebox.showerror(
                "Unexpected Error",
                f"An unexpected error occurred during distance computation:\n\n{exc}",
            )
            self.status_var.set("Unexpected computation failure.")


# =============================================================================
# CLI/bootstrap helpers
# =============================================================================
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.

    The GUI uses a dataset path, and this helper allows the user to override
    the default file location if needed.
    """
    parser = argparse.ArgumentParser(
        description="Radar signal distance GUI with metric dropdown."
    )

    default_dataset = Path(__file__).resolve().parent / "radar_signals.csv"

    parser.add_argument(
        "--dataset",
        default=str(default_dataset),
        help="Path to the radar_signals CSV dataset.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point for the GUI application.

    Returns explicit exit codes to make behavior predictable in automation.
    """
    argv = list(argv) if argv is not None else sys.argv[1:]

    try:
        args = parse_args(argv)

        root = tk.Tk()
        app = RadarDistanceApp(root, args.dataset)

        # Keep a reference to avoid linter complaints about the object seeming unused.
        _ = app

        root.mainloop()
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        traceback.print_exc()
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
