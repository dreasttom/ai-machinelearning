"""
missile_knn.py
==============
K-Nearest Neighbor (KNN) classification on the missiles dataset.
uses dataset https://www.kaggle.com/datasets/fanbyprinciple/north-korea-missile-test-database

What this script does:
  1. Loads and cleans the missiles CSV
  2. Engineers numeric features from messy text columns (MASS, LENGTH, DIAMETER)
  3. Encodes categorical features (ORIGIN, PROPELLANT category)
  4. Builds a clean target variable by collapsing the noisy TYPE column into
     broad missile categories
  5. Splits data into train / test sets
  6. Runs a grid search to find the optimal value of K
  7. Evaluates the best model (accuracy, classification report, confusion matrix)
  8. Produces a multi-panel matplotlib figure saved as missile_knn_results.png

Usage:
  pip install pandas scikit-learn matplotlib seaborn
  python missile_knn.py

Author : Claude (Anthropic)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import re
import warnings

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend – works without a display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    from sklearn.decomposition import PCA
except ImportError as exc:
    # Provide a helpful message so the user knows exactly what to install
    print(
        f"\n[ERROR] Required package not found: {exc}\n"
        "Please run:  pip install pandas scikit-learn matplotlib seaborn\n"
    )
    sys.exit(1)

warnings.filterwarnings("ignore")   # suppress minor sklearn/pandas warnings

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = "missiles.csv"       # path to the input CSV
OUTPUT_PNG  = "missile_knn_results.png"  # where to save the plot
RANDOM_SEED = 42                   # reproducibility
TEST_SIZE   = 0.25                 # 25 % of data held out for testing
K_RANGE     = range(1, 31)         # values of K to try during grid search

# ── Colour palette (used throughout the plots) ────────────────────────────────
PALETTE = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "accent1":  "#58a6ff",
    "accent2":  "#f78166",
    "accent3":  "#3fb950",
    "accent4":  "#d2a8ff",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
}

# ── Broad missile-type mapping ────────────────────────────────────────────────
# The raw TYPE column has hundreds of noisy values (Wikipedia artefacts, line
# breaks, reference numbers, etc.).  We map each row to one of 8 broad classes
# using substring matching.  Rows that match nothing go into "Other".
TYPE_MAP = {
    "Ballistic":       ["ballistic", "srbm", "mrbm", "irbm", "icbm", "slbm"],
    "Cruise":          ["cruise", "alcm", "glcm", "slcm", "tomahawk"],
    "Air-to-Air":      ["air-to-air", "atr", "air to air"],
    "Surface-to-Air":  ["surface-to-air", "sam ", "manpads", "anti-aircraft"],
    "Anti-Tank":       ["anti-tank", "atgm", "atm", "atgw"],
    "Anti-Ship":       ["anti-ship", "ashm", "antisurface"],
    "Rocket Artillery":["rocket artiller", "multiple rocket", "mlrs"],
    "Air-to-Surface":  ["air-to-surface", "air to surface", "agm", "aasm",
                        "glide bomb", "precision-guided munition"],
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_first_number(text: str) -> float:
    """
    Extract the first numeric value (integer or decimal) from a messy string.

    Many cells in this dataset look like '1,200 kg' or '≈24–26 m (79–85 ft)'.
    This function strips commas and returns the first float found, or NaN if
    nothing can be extracted.

    Parameters
    ----------
    text : str
        Raw cell value (may be NaN itself)

    Returns
    -------
    float
        First numeric value found, or np.nan
    """
    if pd.isna(text):
        return np.nan
    # Remove commas used as thousands separators  ("1,200" → "1200")
    cleaned = str(text).replace(",", "")
    # Find the first sequence of digits (with optional decimal point)
    match = re.search(r"[\d]+\.?[\d]*", cleaned)
    if match:
        return float(match.group())
    return np.nan


def normalise_type(raw_type: str) -> str:
    """
    Map a raw TYPE string to one of the broad missile categories defined in
    TYPE_MAP.  The matching is case-insensitive substring search.

    Parameters
    ----------
    raw_type : str
        Value from the TYPE column (may contain newlines and noise)

    Returns
    -------
    str
        One of the keys in TYPE_MAP, or "Other"
    """
    if pd.isna(raw_type):
        return "Other"
    lower = str(raw_type).lower().strip()
    for category, keywords in TYPE_MAP.items():
        if any(kw in lower for kw in keywords):
            return category
    return "Other"


def categorise_propellant(prop: str) -> str:
    """
    Collapse the free-text PROPELLANT column into three simple groups:
      'Solid', 'Liquid', or 'Unknown'.

    Parameters
    ----------
    prop : str
        Raw propellant description

    Returns
    -------
    str
        'Solid', 'Liquid', or 'Unknown'
    """
    if pd.isna(prop):
        return "Unknown"
    lower = str(prop).lower()
    if "solid" in lower:
        return "Solid"
    if "liquid" in lower or "jet" in lower or "turbo" in lower:
        return "Liquid"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the missiles CSV with sensible options.

    Parameters
    ----------
    path : str
        File-system path to missiles.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe

    Raises
    ------
    FileNotFoundError  – if the file does not exist
    ValueError         – if the file is empty or missing required columns
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")   # utf-8-sig handles BOM
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found: '{path}'.\n"
            "Make sure missiles.csv is in the same directory as this script."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    # ── Basic sanity checks ───────────────────────────────────────────────────
    if df.empty:
        raise ValueError("The CSV file is empty.")

    required_cols = {"TYPE", "MASS", "LENGTH", "DIAMETER", "ORIGIN", "PROPELLANT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw dataframe into a model-ready feature matrix plus a target
    vector.  All steps are non-destructive – we work on a copy.

    New columns added
    -----------------
    mass_kg      : numeric mass extracted from MASS column
    length_m     : numeric length in metres extracted from LENGTH
    diameter_m   : numeric diameter in metres extracted from DIAMETER
    propellant_enc : ordinal encoding of Solid/Liquid/Unknown
    origin_enc   : label-encoded ORIGIN
    type_label   : broad missile category (our target)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Augmented copy of the dataframe
    """
    df = df.copy()

    # ── Numeric features ──────────────────────────────────────────────────────
    df["mass_kg"]    = df["MASS"].apply(extract_first_number)
    df["length_m"]   = df["LENGTH"].apply(extract_first_number)
    df["diameter_m"] = df["DIAMETER"].apply(extract_first_number)

    # ── Categorical features ──────────────────────────────────────────────────
    # Propellant → Solid / Liquid / Unknown → 0 / 1 / 2
    prop_map = {"Solid": 0, "Liquid": 1, "Unknown": 2}
    df["propellant_cat"] = df["PROPELLANT"].apply(categorise_propellant)
    df["propellant_enc"] = df["propellant_cat"].map(prop_map)

    # Origin: strip whitespace/newlines, then label-encode
    df["ORIGIN_clean"] = (
        df["ORIGIN"]
        .fillna("Unknown")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    le_origin = LabelEncoder()
    df["origin_enc"] = le_origin.fit_transform(df["ORIGIN_clean"])

    # ── Target variable ───────────────────────────────────────────────────────
    df["type_label"] = df["TYPE"].apply(normalise_type)

    print("[INFO] Type distribution after mapping:")
    print(df["type_label"].value_counts().to_string())
    print()

    return df, le_origin


def build_matrices(df: pd.DataFrame):
    """
    Select features and target, drop rows with any NaN in the feature columns,
    and return X (feature matrix) and y (label vector).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : np.ndarray  (n_samples, 5)
    y : np.ndarray  (n_samples,)
    labels : list[str]  – human-readable class names (for plots)

    Raises
    ------
    ValueError if fewer than 20 usable rows remain
    """
    feature_cols = [
        "mass_kg",
        "length_m",
        "diameter_m",
        "propellant_enc",
        "origin_enc",
    ]

    # Drop rows where any feature is missing
    df_clean = df[feature_cols + ["type_label"]].dropna()

    if len(df_clean) < 20:
        raise ValueError(
            f"Only {len(df_clean)} complete rows found after dropping NaNs. "
            "Not enough data to train a KNN model."
        )

    print(f"[INFO] Using {len(df_clean):,} complete rows for modelling "
          f"(dropped {len(df) - len(df_clean)} rows with missing values)")

    X = df_clean[feature_cols].values.astype(float)
    y = df_clean["type_label"].values

    labels = sorted(df_clean["type_label"].unique())
    print(f"[INFO] Classes: {labels}\n")
    return X, y, labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_knn(X_train, y_train, X_test, y_test):
    """
    Perform a grid search over K values and return the best fitted model along
    with the cross-validation accuracy curve.

    We use 5-fold stratified cross-validation on the TRAINING set to choose K,
    then report final accuracy on the held-out TEST set.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : test data

    Returns
    -------
    best_model   : fitted KNeighborsClassifier
    best_k       : int
    cv_scores    : list of mean CV accuracy for each k in K_RANGE
    test_accuracy: float
    """
    cv_scores = []

    print("[INFO] Grid-searching K …")
    for k in K_RANGE:
        try:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights="uniform",   # each neighbour votes equally
                metric="euclidean",  # standard L2 distance
            )
            # 5-fold cross-validation – returns array of 5 scores
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
            cv_scores.append(scores.mean())
        except Exception as exc:
            # If a particular k fails (e.g., k > n_samples), record 0 and move on
            print(f"  [WARN] k={k} failed during CV: {exc}")
            cv_scores.append(0.0)

    # ── Pick the k with highest mean CV accuracy ───────────────────────────────
    best_k = list(K_RANGE)[int(np.argmax(cv_scores))]
    print(f"[INFO] Best K = {best_k}  (CV accuracy = {max(cv_scores):.3f})\n")

    # ── Retrain on the full training set with best_k ───────────────────────────
    best_model = KNeighborsClassifier(n_neighbors=best_k, weights="uniform",
                                      metric="euclidean")
    best_model.fit(X_train, y_train)

    y_pred        = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"[INFO] Test accuracy  : {test_accuracy:.3f}")
    print("\n[INFO] Classification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return best_model, best_k, cv_scores, test_accuracy


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def apply_dark_theme():
    """
    Configure matplotlib's rcParams for a dark-themed, publication-quality look.
    All subsequent figures will inherit these settings.
    """
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["muted"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#21262d",
        "grid.linewidth":    0.6,
        "font.family":       "monospace",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
    })


def plot_results(
    X, y, X_train, X_test, y_test,
    best_model, best_k, cv_scores, test_accuracy, labels, output_path
):
    """
    Create a 2×2 figure with four diagnostic panels:

      Panel A  (top-left)  : K vs CV accuracy – shows how we chose K
      Panel B  (top-right) : Confusion matrix on the test set
      Panel C  (bottom-left): PCA scatter – 2-D view of the feature space,
                              coloured by predicted class
      Panel D  (bottom-right): Per-class precision / recall bar chart

    Parameters
    ----------
    X, y                   : full dataset arrays (used for PCA scatter)
    X_train, X_test, y_test: split arrays
    best_model             : fitted KNeighborsClassifier
    best_k                 : int, optimal K
    cv_scores              : list of CV accuracies for each k
    test_accuracy          : float
    labels                 : list of class name strings
    output_path            : where to write the PNG
    """
    apply_dark_theme()

    # A distinct colour for each class
    class_colours = [
        PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
        PALETTE["accent4"], "#ffa657", "#79c0ff", "#56d364", "#f0883e"
    ]
    colour_map = {lbl: class_colours[i % len(class_colours)]
                  for i, lbl in enumerate(labels)}

    fig = plt.figure(figsize=(16, 12), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"K-Nearest Neighbour  •  Missiles Dataset  •  Best K = {best_k}  "
        f"•  Test Accuracy = {test_accuracy:.1%}",
        fontsize=14, color=PALETTE["text"], y=0.98, weight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── PANEL A : K vs CV Accuracy ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("A  –  K Selection: CV Accuracy vs K")

    k_values   = list(K_RANGE)
    best_score = max(cv_scores)

    ax_a.plot(k_values, cv_scores, color=PALETTE["accent1"],
              linewidth=2, marker="o", markersize=4, zorder=3)

    # Highlight the chosen K with a vertical line
    ax_a.axvline(best_k, color=PALETTE["accent2"], linestyle="--",
                 linewidth=1.5, label=f"Best K = {best_k}")
    ax_a.axhline(best_score, color=PALETTE["muted"], linestyle=":",
                 linewidth=1, alpha=0.6)

    ax_a.scatter([best_k], [best_score], color=PALETTE["accent2"],
                 s=80, zorder=4)
    ax_a.annotate(
        f"K={best_k}\n{best_score:.3f}",
        xy=(best_k, best_score),
        xytext=(best_k + 1.5, best_score - 0.015),
        color=PALETTE["accent2"], fontsize=8,
        arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"], lw=1)
    )

    ax_a.set_xlabel("K  (number of neighbours)")
    ax_a.set_ylabel("Mean CV Accuracy (5-fold)")
    ax_a.legend()
    ax_a.grid(True, axis="y")
    ax_a.set_xlim(min(k_values) - 0.5, max(k_values) + 0.5)

    # ── PANEL B : Confusion Matrix ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("B  –  Confusion Matrix (Test Set)")

    y_pred = best_model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=labels)

    # Normalise each row so colours show recall rather than raw counts
    # Add small epsilon to avoid division by zero for empty classes
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Build a custom dark-blue sequential cmap
    cmap = sns.color_palette("mako", as_cmap=True)
    im   = ax_b.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Tick labels
    ax_b.set_xticks(range(len(labels)))
    ax_b.set_yticks(range(len(labels)))
    short_labels = [l[:8] for l in labels]   # truncate for readability
    ax_b.set_xticklabels(short_labels, rotation=40, ha="right")
    ax_b.set_yticklabels(short_labels)
    ax_b.set_xlabel("Predicted Label")
    ax_b.set_ylabel("True Label")

    # Annotate cells with raw counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            if val > 0:
                ax_b.text(j, i, str(val), ha="center", va="center",
                          fontsize=7,
                          color="white" if cm_norm[i, j] < 0.6 else "#0d1117")

    plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04).set_label(
        "Row-normalised recall", color=PALETTE["muted"])

    # ── PANEL C : PCA 2-D Scatter ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("C  –  PCA Projection (2D) – Predicted Classes")

    # Project full dataset into 2 principal components for visualisation
    try:
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        X_2d = pca.fit_transform(X)

        # Use model to predict labels for every sample
        y_pred_all = best_model.predict(X)

        for lbl in labels:
            mask = y_pred_all == lbl
            ax_c.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colour_map[lbl], label=lbl,
                s=18, alpha=0.65, edgecolors="none"
            )

        var_exp = pca.explained_variance_ratio_
        ax_c.set_xlabel(f"PC1  ({var_exp[0]:.1%} var)")
        ax_c.set_ylabel(f"PC2  ({var_exp[1]:.1%} var)")
        ax_c.legend(loc="upper right", markerscale=1.5,
                    framealpha=0.3, labelspacing=0.3)
    except Exception as exc:
        # PCA might fail for degenerate feature sets – fail gracefully
        ax_c.text(0.5, 0.5, f"PCA failed:\n{exc}",
                  transform=ax_c.transAxes, ha="center", va="center",
                  color=PALETTE["accent2"])

    # ── PANEL D : Per-Class Precision & Recall ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("D  –  Per-Class Precision & Recall")

    from sklearn.metrics import precision_recall_fscore_support

    try:
        prec, rec, _, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, zero_division=0
        )

        x_pos     = np.arange(len(labels))
        bar_width = 0.35

        bars_p = ax_d.bar(x_pos - bar_width / 2, prec, bar_width,
                           label="Precision", color=PALETTE["accent1"],
                           alpha=0.85, edgecolor="none")
        bars_r = ax_d.bar(x_pos + bar_width / 2, rec, bar_width,
                           label="Recall",    color=PALETTE["accent3"],
                           alpha=0.85, edgecolor="none")

        # Annotate bars with support (number of test samples)
        for i, sup in enumerate(support):
            ax_d.text(x_pos[i], 1.02, f"n={sup}", ha="center",
                      fontsize=6.5, color=PALETTE["muted"])

        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels([l[:8] for l in labels], rotation=40, ha="right")
        ax_d.set_ylim(0, 1.12)
        ax_d.set_ylabel("Score")
        ax_d.legend()
        ax_d.grid(True, axis="y", alpha=0.4)

    except Exception as exc:
        ax_d.text(0.5, 0.5, f"Bar chart failed:\n{exc}",
                  transform=ax_d.transAxes, ha="center", va="center",
                  color=PALETTE["accent2"])

    # ── Save ──────────────────────────────────────────────────────────────────
    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"\n[INFO] Figure saved → {output_path}")
    except OSError as exc:
        raise OSError(
            f"Could not write figure to '{output_path}': {exc}"
        ) from exc
    finally:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrates the full pipeline:
      load → engineer → split → scale → train → evaluate → visualise
    """
    print("=" * 60)
    print("  Missile Dataset  –  KNN Classification Pipeline")
    print("=" * 60 + "\n")

    # ── Step 1 : Load ──────────────────────────────────────────────────────────
    try:
        df_raw = load_data(DATA_PATH)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    # ── Step 2 : Feature engineering ──────────────────────────────────────────
    try:
        df_eng, _ = engineer_features(df_raw)
        X, y, labels = build_matrices(df_eng)
    except (ValueError, KeyError) as exc:
        print(f"[FATAL] Feature engineering failed: {exc}")
        sys.exit(1)

    # ── Step 3 : Train / test split ────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,          # keep class proportions in both splits
        )
        print(f"[INFO] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}\n")
    except ValueError as exc:
        # stratify fails when a class has only 1 member – fall back without it
        print(f"  [WARN] Stratified split failed ({exc}), using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

    # ── Step 4 : Feature scaling ───────────────────────────────────────────────
    # KNN is distance-based, so features MUST be on the same scale.
    # We fit the scaler only on the training set to prevent data leakage.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    # Also scale the full dataset (needed for PCA scatter plot)
    X_scaled = scaler.transform(X)

    # ── Step 5 : Train & evaluate ──────────────────────────────────────────────
    try:
        best_model, best_k, cv_scores, test_accuracy = train_knn(
            X_train, y_train, X_test, y_test
        )
    except Exception as exc:
        print(f"[FATAL] Model training failed: {exc}")
        sys.exit(1)

    # ── Step 6 : Plot ──────────────────────────────────────────────────────────
    try:
        plot_results(
            X_scaled, y,
            X_train, X_test, y_test,
            best_model, best_k, cv_scores, test_accuracy,
            labels, OUTPUT_PNG
        )
    except Exception as exc:
        print(f"[ERROR] Plotting failed: {exc}")
        # Non-fatal – the model results were already printed above

    print("\n[DONE] Pipeline complete.")
    print(f"  Best K            : {best_k}")
    print(f"  Test accuracy     : {test_accuracy:.1%}")
    print(f"  Classes modelled  : {', '.join(labels)}")
    print(f"  Plot saved to     : {OUTPUT_PNG}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
