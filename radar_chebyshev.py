#!/usr/bin/env python3
"""
radar_chebyshev.py

A heavily commented example script that demonstrates how to compute the
Chebyshev distance on a radar-signal-style dataset.

Chebyshev distance between two vectors x and y is defined as:

    D(x, y) = max_i(|x_i - y_i|)

In plain English:
- Look at the absolute difference for each feature
- Take the largest one
- That largest difference is the Chebyshev distance

This metric is useful when you care about the single biggest mismatch
between two observations.

The script supports:
1. Loading a CSV file containing radar pulse features
2. Validating that the file exists and has the expected structure
3. Validating that required columns are present and numeric
4. Handling missing values and malformed rows gracefully
5. Computing:
   - distance from a reference row to every other row
   - pairwise distances between all rows
   - nearest neighbors for a chosen pulse

Usage examples:
    python radar_chebyshev.py --dataset radar_signals.csv
    python radar_chebyshev.py --dataset radar_signals.csv --reference-id P001
    python radar_chebyshev.py --dataset radar_signals.csv --reference-id P003 --top-k 5
    python radar_chebyshev.py --dataset radar_signals.csv --write-output distances.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------
# Using specific exception classes makes error handling more precise and makes
# failures easier to understand for anyone running or maintaining the script.
class RadarDataError(Exception):
    """Base class for radar dataset related errors."""


class DatasetNotFoundError(RadarDataError):
    """Raised when the dataset path does not exist."""


class DatasetFormatError(RadarDataError):
    """Raised when the dataset structure is invalid."""


class DistanceComputationError(Exception):
    """Raised when a distance computation cannot be completed."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
# A dataclass gives us a clean, explicit structure for each radar pulse record.
# The `features()` method returns the numeric values in a consistent order.
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

    def features(self) -> List[float]:
        """
        Return the numeric feature vector in a fixed order.

        A fixed order is crucial because distance metrics assume the i-th value
        in one vector corresponds to the same feature in the other vector.
        """
        return [
            self.frequency_mhz,
            self.amplitude_db,
            self.pulse_width_us,
            self.pri_us,
            self.doppler_hz,
            self.snr_db,
            self.range_km,
            self.azimuth_deg,
            self.elevation_deg,
        ]


# ---------------------------------------------------------------------------
# Expected dataset schema
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def safe_float(value: str, column_name: str, pulse_id: str) -> float:
    """
    Convert a value to float with validation.

    Robust handling matters because CSV files are often edited manually and may
    contain blank strings, invalid text, or values like 'NaN'/'inf' that can
    silently poison downstream calculations.
    """
    if value is None:
        raise DatasetFormatError(
            f"Missing value in column '{column_name}' for pulse '{pulse_id}'."
        )

    cleaned = str(value).strip()
    if cleaned == "":
        raise DatasetFormatError(
            f"Blank value in column '{column_name}' for pulse '{pulse_id}'."
        )

    try:
        number = float(cleaned)
    except ValueError as exc:
        raise DatasetFormatError(
            f"Non-numeric value '{value}' in column '{column_name}' for pulse '{pulse_id}'."
        ) from exc

    # Reject NaN and infinity because they break or distort distance metrics.
    if math.isnan(number) or math.isinf(number):
        raise DatasetFormatError(
            f"Invalid numeric value '{value}' in column '{column_name}' for pulse '{pulse_id}'. "
            "NaN and infinity are not allowed."
        )

    return number


def validate_file_path(path: str) -> None:
    """
    Ensure the dataset path exists, is a file, and is readable.

    Raising descriptive exceptions early prevents confusing failures later.
    """
    if not path:
        raise DatasetNotFoundError("No dataset path was provided.")

    if not os.path.exists(path):
        raise DatasetNotFoundError(f"Dataset file does not exist: {path}")

    if not os.path.isfile(path):
        raise DatasetNotFoundError(f"Dataset path is not a file: {path}")

    if not os.access(path, os.R_OK):
        raise DatasetNotFoundError(f"Dataset file is not readable: {path}")


def validate_headers(headers: Sequence[str]) -> None:
    """
    Validate that the CSV includes all required columns.

    We do not require a strict column order, only that all required fields exist.
    """
    if not headers:
        raise DatasetFormatError("The CSV file has no header row.")

    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        raise DatasetFormatError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )


def load_radar_dataset(path: str) -> List[RadarPulse]:
    """
    Load and validate the radar dataset from CSV.

    Returns a list of RadarPulse objects.
    """
    validate_file_path(path)

    pulses: List[RadarPulse] = []
    seen_ids = set()

    try:
        with open(path, mode="r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            validate_headers(reader.fieldnames or [])

            for row_number, row in enumerate(reader, start=2):
                # Row numbering starts at 2 because row 1 is the header.
                if row is None:
                    raise DatasetFormatError(f"Encountered an unreadable row at line {row_number}.")

                pulse_id = str(row.get("pulse_id", "")).strip()
                if pulse_id == "":
                    raise DatasetFormatError(f"Missing pulse_id at line {row_number}.")

                if pulse_id in seen_ids:
                    raise DatasetFormatError(
                        f"Duplicate pulse_id '{pulse_id}' found at line {row_number}."
                    )
                seen_ids.add(pulse_id)

                try:
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
                except DatasetFormatError:
                    # Re-raise our clean, user-facing format errors unchanged.
                    raise
                except Exception as exc:
                    # Wrap unexpected parsing errors with more context.
                    raise DatasetFormatError(
                        f"Unexpected parsing error at line {row_number} for pulse '{pulse_id}'."
                    ) from exc

                pulses.append(pulse)

    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            "The dataset could not be decoded as UTF-8 text."
        ) from exc
    except csv.Error as exc:
        raise DatasetFormatError(
            f"CSV parsing failed: {exc}"
        ) from exc

    if not pulses:
        raise DatasetFormatError("The dataset contains no data rows.")

    return pulses


def chebyshev_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the Chebyshev distance between two equally-sized numeric vectors.

    Error handling is intentionally strict:
    - The vectors must not be empty
    - They must have the same length
    - Every element must be a finite numeric value
    """
    if vector_a is None or vector_b is None:
        raise DistanceComputationError("Input vectors must not be None.")

    if len(vector_a) == 0 or len(vector_b) == 0:
        raise DistanceComputationError("Input vectors must not be empty.")

    if len(vector_a) != len(vector_b):
        raise DistanceComputationError(
            f"Vector length mismatch: {len(vector_a)} != {len(vector_b)}"
        )

    max_difference = None

    for index, (a, b) in enumerate(zip(vector_a, vector_b)):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise DistanceComputationError(
                f"Non-numeric value encountered at index {index}: {a!r}, {b!r}"
            )

        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            raise DistanceComputationError(
                f"Invalid numeric value encountered at index {index}: {a!r}, {b!r}"
            )

        difference = abs(a - b)

        if max_difference is None or difference > max_difference:
            max_difference = difference

    # This should never remain None because empty vectors are rejected above,
    # but the check keeps the function defensive and future-proof.
    if max_difference is None:
        raise DistanceComputationError("Unable to compute distance from the provided vectors.")

    return float(max_difference)


def compute_distances_from_reference(
    pulses: Sequence[RadarPulse], reference_id: str
) -> List[Tuple[str, float]]:
    """
    Compute Chebyshev distance from the chosen reference pulse to every other pulse.

    Returns a list of (pulse_id, distance) tuples sorted by ascending distance.
    """
    if not pulses:
        raise DistanceComputationError("No radar pulses were provided.")

    reference_lookup: Dict[str, RadarPulse] = {pulse.pulse_id: pulse for pulse in pulses}
    reference_pulse = reference_lookup.get(reference_id)

    if reference_pulse is None:
        valid_ids = ", ".join(sorted(reference_lookup.keys()))
        raise DistanceComputationError(
            f"Reference pulse_id '{reference_id}' was not found. Valid IDs: {valid_ids}"
        )

    results: List[Tuple[str, float]] = []

    for pulse in pulses:
        if pulse.pulse_id == reference_id:
            continue

        distance = chebyshev_distance(reference_pulse.features(), pulse.features())
        results.append((pulse.pulse_id, distance))

    results.sort(key=lambda item: item[1])
    return results


def compute_pairwise_distances(pulses: Sequence[RadarPulse]) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise Chebyshev distances for all unique pulse pairs.

    For N pulses, this produces N*(N-1)/2 distances.
    """
    if not pulses:
        raise DistanceComputationError("No radar pulses were provided.")

    results: List[Tuple[str, str, float]] = []

    for i in range(len(pulses)):
        for j in range(i + 1, len(pulses)):
            pulse_a = pulses[i]
            pulse_b = pulses[j]
            distance = chebyshev_distance(pulse_a.features(), pulse_b.features())
            results.append((pulse_a.pulse_id, pulse_b.pulse_id, distance))

    return results


def write_distance_report(path: str, rows: Iterable[Tuple[str, float]]) -> None:
    """
    Write reference-to-target distances to a CSV file.

    The output schema is intentionally simple so the file is easy to inspect
    manually or ingest into other tools.
    """
    if not path:
        raise ValueError("An output path must be provided.")

    try:
        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pulse_id", "chebyshev_distance"])
            for pulse_id, distance in rows:
                writer.writerow([pulse_id, f"{distance:.6f}"])
    except OSError as exc:
        raise OSError(f"Failed to write output file '{path}': {exc}") from exc


def pretty_print_dataset_summary(pulses: Sequence[RadarPulse]) -> None:
    """
    Print a simple dataset summary to help users confirm the file loaded correctly.
    """
    labels: Dict[str, int] = {}
    for pulse in pulses:
        labels[pulse.label] = labels.get(pulse.label, 0) + 1

    print("=" * 72)
    print("RADAR DATASET SUMMARY")
    print("=" * 72)
    print(f"Total pulses loaded : {len(pulses)}")
    print("Class distribution  :")
    for label, count in sorted(labels.items()):
        print(f"  - {label:<18} {count}")
    print("=" * 72)


def pretty_print_nearest_neighbors(
    reference_id: str, distances: Sequence[Tuple[str, float]], top_k: int
) -> None:
    """
    Print the nearest neighbors to the selected reference pulse.
    """
    print(f"\nNearest neighbors for reference pulse '{reference_id}':")
    print("-" * 72)
    print(f"{'Rank':<6}{'Pulse ID':<12}{'Chebyshev Distance':>22}")
    print("-" * 72)

    for rank, (pulse_id, distance) in enumerate(distances[:top_k], start=1):
        print(f"{rank:<6}{pulse_id:<12}{distance:>22.6f}")

    if not distances:
        print("No comparison pulses were available.")


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """
    Parse command-line arguments with validation.
    """
    parser = argparse.ArgumentParser(
        description="Compute Chebyshev distances on a radar signal dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the radar signal CSV dataset.",
    )
    parser.add_argument(
        "--reference-id",
        default=None,
        help="Pulse ID to use as the reference. Defaults to the first record in the dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest neighbors to display. Must be >= 1.",
    )
    parser.add_argument(
        "--write-output",
        default=None,
        help="Optional path for writing reference distances to CSV.",
    )
    parser.add_argument(
        "--show-pairwise-count",
        action="store_true",
        help="If set, display how many unique pairwise distances were computed.",
    )

    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main program entry point.

    Returning explicit status codes makes the script easier to use in shell
    pipelines, automation, batch jobs, and CI systems.
    """
    argv = argv if argv is not None else sys.argv[1:]

    try:
        args = parse_arguments(argv)
        pulses = load_radar_dataset(args.dataset)

        pretty_print_dataset_summary(pulses)

        reference_id = args.reference_id or pulses[0].pulse_id

        distances = compute_distances_from_reference(pulses, reference_id)
        pretty_print_nearest_neighbors(reference_id, distances, args.top_k)

        if args.show_pairwise_count:
            pairwise = compute_pairwise_distances(pulses)
            print(f"\nUnique pairwise distances computed: {len(pairwise)}")

        if args.write_output:
            write_distance_report(args.write_output, distances)
            print(f"\nDistance report written to: {args.write_output}")

        return 0

    except (DatasetNotFoundError, DatasetFormatError, DistanceComputationError, ValueError) as exc:
        # These are expected, user-facing errors. We print a clean message.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully instead of showing a stack trace.
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        # Last-resort catch for truly unexpected failures.
        # In production systems you might also log the traceback.
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
