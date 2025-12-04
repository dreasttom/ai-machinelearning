#!/usr/bin/env python3
"""
Sentiment analysis demo using the Enron Corpus.

This script:
    - Recursively walks a directory containing Enron email .txt files
    - Loads a sample of emails
    - Uses NLTK's VADER sentiment analyzer to score each email
    - Classifies emails as positive / neutral / negative
    - Prints aggregate statistics and a few labeled examples

Usage:
    python enron_sentiment_demo.py --data_dir /path/to/enron_mail_20110402 you need to download
    the data set https://www.kaggle.com/datasets/wcukierski/enron-email-dataset?resource=download
"""

import argparse
import logging
import os
import random
import sys
from typing import List, Tuple, Dict

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def configure_logging(verbose: bool = False) -> None:
    """
    Configure the logging level and format.

    :param verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sentiment analysis demo on the Enron Corpus using NLTK VADER."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the Enron Corpus (e.g., 'enron_mail_20110402').",
    )
    parser.add_argument(
        "--max_emails",
        type=int,
        default=2000,
        help="Maximum number of emails to sample for analysis (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def ensure_vader_downloaded() -> None:
    """
    Ensure that the NLTK VADER lexicon is downloaded.

    If it is missing, attempt to download it. If download fails,
    raise an informative RuntimeError.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        logging.debug("VADER lexicon already downloaded.")
    except LookupError:
        logging.info("VADER lexicon not found. Downloading via NLTK...")
        try:
            nltk.download("vader_lexicon")
        except Exception as e:
            raise RuntimeError(
                "Failed to download VADER lexicon. "
                "Check your internet connection and try again."
            ) from e


def load_email_paths(root_dir: str, extension: str = ".txt") -> List[str]:
    """
    Recursively find all email files in the given directory.

    :param root_dir: Root directory of the Enron Corpus.
    :param extension: File extension to match (default: '.txt').
    :return: List of email file paths.
    :raises FileNotFoundError: If root_dir does not exist or is not a directory.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Provided data_dir is not a directory: {root_dir}")

    email_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extension):
                full_path = os.path.join(dirpath, filename)
                email_paths.append(full_path)

    logging.info("Found %d email files with extension '%s'.", len(email_paths), extension)
    return email_paths


def read_email_text(path: str) -> str:
    """
    Read a single email file as text.

    :param path: Path to the email file.
    :return: File contents as a string (possibly empty).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (OSError, UnicodeError) as e:
        # Log the error and return an empty string so the caller can skip it.
        logging.debug("Error reading file '%s': %s", path, e)
        return ""


def sample_emails(email_paths: List[str], max_emails: int, seed: int) -> List[str]:
    """
    Randomly sample up to max_emails from the list of email_paths.

    :param email_paths: List of all email file paths.
    :param max_emails: Maximum number of emails to sample.
    :param seed: Random seed for reproducibility.
    :return: List of sampled file paths.
    """
    if not email_paths:
        logging.warning("No email paths provided to sample from.")
        return []

    random.seed(seed)
    if len(email_paths) <= max_emails:
        logging.info(
            "Number of emails (%d) <= max_emails (%d). Using all.",
            len(email_paths),
            max_emails,
        )
        return email_paths

    logging.info("Sampling %d emails from %d total emails.", max_emails, len(email_paths))
    return random.sample(email_paths, max_emails)


def classify_sentiment(score: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    """
    Classify a compound sentiment score as 'positive', 'negative', or 'neutral'.

    :param score: Compound sentiment score from VADER.
    :param pos_threshold: Minimum compound score to be considered positive.
    :param neg_threshold: Maximum compound score to be considered negative.
    :return: One of 'positive', 'negative', or 'neutral'.
    """
    if score >= pos_threshold:
        return "positive"
    if score <= neg_threshold:
        return "negative"
    return "neutral"


def analyze_emails(
    email_texts: List[str],
    analyzer: SentimentIntensityAnalyzer,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Analyze a list of email texts and compute sentiment for each.

    :param email_texts: List of email bodies as strings.
    :param analyzer: NLTK SentimentIntensityAnalyzer instance.
    :return:
        - A list of dicts containing text, scores, and label for each email.
        - A dict with counts of 'positive', 'negative', 'neutral'.
    """
    results = []
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for text in email_texts:
        if not text.strip():
            # Skip empty emails
            continue

        scores = analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        label = classify_sentiment(compound)

        label_counts[label] = label_counts.get(label, 0) + 1

        results.append(
            {
                "text": text,
                "scores": scores,
                "label": label,
            }
        )

    return results, label_counts


def print_summary(label_counts: Dict[str, int], total: int) -> None:
    """
    Print a summary table of sentiment label counts.

    :param label_counts: Dict with keys 'positive', 'negative', 'neutral'.
    :param total: Total number of analyzed emails.
    """
    logging.info("Sentiment analysis summary:")
    print("\n=== Sentiment Distribution ===")
    print(f"Total emails analyzed: {total}")
    for label in ["positive", "neutral", "negative"]:
        count = label_counts.get(label, 0)
        pct = (count / total * 100.0) if total > 0 else 0.0
        print(f"{label.capitalize():8s}: {count:5d} ({pct:5.1f}%)")
    print("==============================\n")


def print_sample_emails(
    analyzed_emails: List[Dict],
    num_examples: int = 5,
) -> None:
    """
    Print a few example emails with their sentiment label and scores.

    :param analyzed_emails: List of analyzed email dicts.
    :param num_examples: Number of examples to print.
    """
    if not analyzed_emails:
        logging.warning("No analyzed emails to display.")
        return

    num_examples = min(num_examples, len(analyzed_emails))
    print(f"=== Showing {num_examples} example emails ===\n")

    # Shuffle to show a variety of sentiment
