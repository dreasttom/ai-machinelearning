"""
Spherical K-Means Clustering (from scratch, student-friendly) + GUI (Tkinter)
=============================================================================

This script loads a CSV (like the attached naturalgas.csv), turns selected columns into
a numeric feature matrix, then runs **Spherical K-Means**:

- Like regular K-Means, but:
  1) Each data row vector is normalized to have length 1 (unit norm)
  2) Similarity is **cosine similarity** (dot product of unit vectors)
  3) Cluster centroids are also normalized to unit length each update

Why Spherical K-Means?
- Works well when direction matters more than magnitude.
- Common in text clustering (TF-IDF vectors), but can also be used for mixed tabular
  data once encoded as vectors.

GUI Features (for teaching):
- Load a CSV file
- Choose which columns to use as features
- One-hot encode categorical columns automatically
- Choose numeric scaling (standardize or none)
- Set K, max iterations, random seed
- Run clustering and see progress (objective per iteration)
- View 2D PCA projection plot (no sklearn required)
- View cluster sizes
- Save a new CSV with a "cluster" column

Dependencies:
    pip install numpy pandas matplotlib

Run:
    python spherical_kmeans_gui.py

Tip:
- Place naturalgas.csv in the same folder as this script, or click "Load CSV" in the GUI.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================================================================
# 1) Math Utilities (Normalization, PCA, etc.)
# =============================================================================

def row_normalize_unit(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row vector to unit length:
        x_norm = x / ||x||
    This is the key step for spherical k-means.

    eps prevents division by zero if a row becomes all zeros.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def standardize_columns(X: np.ndarray, eps: float = 1e-12):
    """
    Standardize features (columns):
        X_std = (X - mean) / std

    Returns:
        X_std, mean, std
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X - mean) / std, mean, std


def pca_2d(X: np.ndarray):
    """
    Simple PCA to 2D using SVD (no scikit-learn).

    Steps:
    1) Center the data (subtract column means)
    2) Compute SVD: X_centered = U S V^T
    3) First 2 principal directions are rows of V^T (or columns of V)

    Returns:
        X_2d: (N, 2) projection
    """
    # Center (important for PCA)
    Xc = X - X.mean(axis=0, keepdims=True)

    # SVD
    # Xc = U S Vt
    # principal components directions are Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Project onto first two components
    # Components are the first two rows of Vt
    components_2 = Vt[:2].T  # shape (D, 2)
    X_2d = Xc @ components_2  # shape (N, 2)
    return X_2d


# =============================================================================
# 2) Spherical K-Means (from scratch)
# =============================================================================

class SphericalKMeans:
    """
    Spherical K-Means implementation from scratch using NumPy.

    Key differences vs ordinary K-Means:
    - Input rows should be unit normalized
    - Centroids are kept unit normalized
    - Assignment uses cosine similarity (dot product of unit vectors)

    Objective (one common version):
        maximize sum_i cos(x_i, c_{z_i})
    since with unit vectors cos(x, c) == x dot c.
    """

    def __init__(self, k: int = 3, max_iter: int = 50, seed: int = 0):
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

        # learned results
        self.centroids_ = None  # shape (k, D)
        self.labels_ = None     # shape (N,)
        self.objective_history_ = []  # list of objective values per iteration

    def _initialize_centroids(self, X_unit: np.ndarray):
        """
        Initialize centroids by picking k random data points.

        (Teaching note) Other inits exist (k-means++, etc.), but random is easiest.
        """
        rng = np.random.default_rng(self.seed)
        n = X_unit.shape[0]

        if self.k > n:
            raise ValueError("k cannot be larger than number of data points.")

        indices = rng.choice(n, size=self.k, replace=False)
        C = X_unit[indices].copy()

        # Ensure unit norm (they already are if X_unit is unit-normalized)
        C = row_normalize_unit(C)
        return C

    def fit(self, X: np.ndarray):
        """
        Fit spherical k-means.

        X: raw feature matrix (N, D). We'll unit-normalize inside.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (N, D).")

        # 1) Unit-normalize data rows
        X_unit = row_normalize_unit(X)

        # 2) Initialize centroids
        C = self._initialize_centroids(X_unit)

        # 3) Iterate: assign -> update
        self.objective_history_ = []

        for it in range(1, self.max_iter + 1):
            # --- Assignment step ---
            # Cosine similarity for unit vectors is dot product:
            #   sim(i, j) = x_i dot c_j
            # We compute all similarities at once:
            sims = X_unit @ C.T  # shape (N, k)

            labels = np.argmax(sims, axis=1)  # best centroid for each point
            best_sims = sims[np.arange(X_unit.shape[0]), labels]

            # Objective = average (or sum) of best similarities
            obj = float(np.mean(best_sims))
            self.objective_history_.append(obj)

            # --- Update step ---
            # New centroid = mean of assigned unit vectors, then normalize.
            C_new = np.zeros_like(C)

            for j in range(self.k):
                mask = (labels == j)
                if np.any(mask):
                    # Mean direction
                    mean_vec = X_unit[mask].mean(axis=0, keepdims=True)
                    C_new[j] = row_normalize_unit(mean_vec)[0]
                else:
                    # Empty cluster: reinitialize to a random data point
                    # (Teaching note) This can happen when k is large or data is uneven.
                    # A common fix is to "steal" a far-away point, but random is simplest.
                    rng = np.random.default_rng(self.seed + it + j)
                    idx = rng.integers(0, X_unit.shape[0])
                    C_new[j] = X_unit[idx]

            # Check for convergence:
            # If centroids barely move, we can stop early.
            shift = np.linalg.norm(C_new - C)
            C = C_new

            if shift < 1e-6:
                break

        # Save results
        self.centroids_ = C
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        """
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_unit = row_normalize_unit(X)
        sims = X_unit @ self.centroids_.T
        return np.argmax(sims, axis=1)


# =============================================================================
# 3) Feature Building for Tabular CSV (one-hot + numeric scaling)
# =============================================================================

def build_feature_matrix(df: pd.DataFrame,
                         selected_cols,
                         standardize_numeric: bool = True,
                         one_hot_categoricals: bool = True):
    """
    Convert selected columns into a numeric matrix X.

    Strategy (student-friendly and practical):
    - Numeric columns: keep as numeric (and optionally standardize)
    - Categorical columns: one-hot encode (optional)

    Returns:
        X (np.ndarray), feature_names (list[str]), df_clean (pd.DataFrame)
    """
    if not selected_cols:
        raise ValueError("No columns selected.")

    # Work on a copy
    work = df[selected_cols].copy()

    # Handle missing values simply:
    # - For numeric: fill with column mean
    # - For categorical: fill with 'MISSING'
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].astype(float)
            if work[col].isna().any():
                work[col] = work[col].fillna(work[col].mean())
        else:
            work[col] = work[col].astype(str)
            if work[col].isna().any():
                work[col] = work[col].fillna("MISSING")

    # Separate numeric vs categorical
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    cat_cols = [c for c in work.columns if c not in numeric_cols]

    # Start with numeric matrix
    X_parts = []
    feature_names = []

    if numeric_cols:
        X_num = work[numeric_cols].to_numpy(dtype=float)
        if standardize_numeric:
            X_num, _, _ = standardize_columns(X_num)
        X_parts.append(X_num)
        feature_names.extend(numeric_cols)

    # One-hot encode categoricals
    if cat_cols and one_hot_categoricals:
        dummies = pd.get_dummies(work[cat_cols], columns=cat_cols, drop_first=False)
        X_cat = dummies.to_numpy(dtype=float)
        X_parts.append(X_cat)
        feature_names.extend(list(dummies.columns))
    elif cat_cols and not one_hot_categoricals:
        # If categoricals are selected but not one-hot encoded, we can't use them directly.
        # (Teaching choice) We'll raise an error to keep behavior clear.
        raise ValueError("Categorical columns selected but one-hot encoding is OFF. Either enable it or deselect categorical columns.")

    if not X_parts:
        raise ValueError("No usable features were produced. (Did you select only categorical columns while one-hot was disabled?)")

    X = np.concatenate(X_parts, axis=1)
    return X, feature_names, work


# =============================================================================
# 4) GUI Application
# =============================================================================

class SphericalKMeansGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spherical K-Means (from scratch) - Student GUI")
        self.geometry("1250x800")

        # Data & results
        self.df = None
        self.csv_path = None

        self.X = None
        self.feature_names = None

        self.model = None
        self.labels = None

        # Build UI
        self._build_ui()

        # Try auto-load naturalgas.csv if present in current folder
        maybe = "naturalgas.csv"
        if os.path.exists(maybe):
            try:
                self.load_csv(maybe)
            except Exception:
                pass

    # ---------------- UI layout ----------------

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left: file + column selection
        ttk.Label(left, text="Data", font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Button(left, text="Load CSV", command=self.load_csv_dialog).pack(fill=tk.X, pady=4)

        self.file_label = ttk.Label(left, text="No file loaded.")
        self.file_label.pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Select feature columns:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.cols_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, height=12)
        self.cols_listbox.pack(fill=tk.X, pady=4)

        # Preprocessing options
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Preprocessing", font=("Arial", 12, "bold")).pack(anchor="w")

        self.one_hot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="One-hot encode categorical columns", variable=self.one_hot_var).pack(anchor="w")

        self.std_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Standardize numeric columns", variable=self.std_var).pack(anchor="w")

        # K-Means params
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Spherical K-Means Params", font=("Arial", 12, "bold")).pack(anchor="w")

        ttk.Label(left, text="k (number of clusters):").pack(anchor="w")
        self.k_entry = ttk.Entry(left)
        self.k_entry.insert(0, "5")
        self.k_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="max_iter:").pack(anchor="w")
        self.iter_entry = ttk.Entry(left)
        self.iter_entry.insert(0, "30")
        self.iter_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="random seed:").pack(anchor="w")
        self.seed_entry = ttk.Entry(left)
        self.seed_entry.insert(0, "0")
        self.seed_entry.pack(fill=tk.X, pady=2)

        ttk.Button(left, text="Run Spherical K-Means", command=self.run_clustering).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Save CSV with cluster labels", command=self.save_results).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Logs", font=("Arial", 12, "bold")).pack(anchor="w")
        self.log_text = tk.Text(left, height=16, width=44)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Right: tabs (plots + summary)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_plots = ttk.Frame(self.nb, padding=10)
        self.tab_summary = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.tab_plots, text="Plots")
        self.nb.add(self.tab_summary, text="Cluster Summary")

        self._build_plots_tab()
        self._build_summary_tab()

    def _build_plots_tab(self):
        ttk.Label(self.tab_plots, text="Objective + PCA Projection", font=("Arial", 12, "bold")).pack(anchor="w")

        self.fig = Figure(figsize=(8.2, 6.2), dpi=100)
        self.ax_obj = self.fig.add_subplot(211)
        self.ax_pca = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plots)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._refresh_plots()

    def _build_summary_tab(self):
        ttk.Label(self.tab_summary, text="Cluster sizes", font=("Arial", 12, "bold")).pack(anchor="w")

        cols = ("Cluster", "Count")
        self.summary_tree = ttk.Treeview(self.tab_summary, columns=cols, show="headings", height=20)
        for c in cols:
            self.summary_tree.heading(c, text=c)
            self.summary_tree.column(c, width=140, anchor="center")
        self.summary_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    # ---------------- Logging ----------------

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # ---------------- Data Loading ----------------

    def load_csv_dialog(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.load_csv(path)

    def load_csv(self, path: str):
        df = pd.read_csv(path)
        self.df = df
        self.csv_path = path
        self.file_label.configure(text=f"Loaded: {os.path.basename(path)}  ({df.shape[0]} rows, {df.shape[1]} cols)")
        self.log(f"Loaded CSV: {path}")
        self.log(f"Columns: {list(df.columns)}")

        # Populate listbox
        self.cols_listbox.delete(0, tk.END)
        for c in df.columns:
            self.cols_listbox.insert(tk.END, c)

        # Helpful default selection for the natural gas dataset:
        # year, month, value, area-name, process-name (common sense clustering signal)
        defaults = ["year", "month", "value", "area-name", "process-name"]
        for i, c in enumerate(df.columns):
            if c in defaults:
                self.cols_listbox.selection_set(i)

        self.model = None
        self.labels = None
        self.X = None
        self.feature_names = None
        self._refresh_plots()
        self._refresh_summary()

    # ---------------- Feature Building ----------------

    def _get_selected_columns(self):
        if self.df is None:
            raise ValueError("No CSV loaded.")
        idxs = self.cols_listbox.curselection()
        if not idxs:
            return []
        return [self.cols_listbox.get(i) for i in idxs]

    # ---------------- Clustering ----------------

    def run_clustering(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please load a CSV first.")
            return

        try:
            selected_cols = self._get_selected_columns()
            one_hot = bool(self.one_hot_var.get())
            std_num = bool(self.std_var.get())

            k = int(self.k_entry.get().strip())
            max_iter = int(self.iter_entry.get().strip())
            seed = int(self.seed_entry.get().strip())

            if k < 2:
                raise ValueError("k must be at least 2.")
            if max_iter < 1:
                raise ValueError("max_iter must be at least 1.")

            # 1) Build feature matrix X
            self.log("Building feature matrix...")
            X, feat_names, _ = build_feature_matrix(
                self.df,
                selected_cols=selected_cols,
                standardize_numeric=std_num,
                one_hot_categoricals=one_hot
            )
            self.X = X
            self.feature_names = feat_names

            self.log(f"Feature matrix shape: {X.shape} (N={X.shape[0]}, D={X.shape[1]})")
            self.log("Running Spherical K-Means...")
            self.log("Teaching note: we unit-normalize rows so cosine similarity becomes a dot product.")

            # 2) Fit spherical k-means
            model = SphericalKMeans(k=k, max_iter=max_iter, seed=seed)
            model.fit(X)

            self.model = model
            self.labels = model.labels_

            self.log(f"Done. Iterations run: {len(model.objective_history_)}")
            self.log(f"Final objective (mean cosine similarity): {model.objective_history_[-1]:.6f}")

            # 3) Update plots and summary
            self._refresh_plots()
            self._refresh_summary()

        except Exception as e:
            messagebox.showerror("Clustering error", str(e))

    # ---------------- Saving ----------------

    def save_results(self):
        if self.df is None or self.labels is None:
            messagebox.showwarning("No results", "Run clustering first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save clustered CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not out_path:
            return

        df_out = self.df.copy()
        df_out["cluster"] = self.labels
        df_out.to_csv(out_path, index=False)
        self.log(f"Saved results to: {out_path}")

    # ---------------- Plots and Summary ----------------

    def _refresh_plots(self):
        self.ax_obj.clear()
        self.ax_pca.clear()

        # Objective plot
        self.ax_obj.set_title("Spherical K-Means Objective per Iteration")
        self.ax_obj.set_xlabel("Iteration")
        self.ax_obj.set_ylabel("Mean cosine similarity (higher is better)")

        if self.model is not None and self.model.objective_history_:
            self.ax_obj.plot(self.model.objective_history_, marker="o", linewidth=1)

        # PCA projection plot
        self.ax_pca.set_title("2D PCA Projection of Feature Vectors (colored by cluster)")
        self.ax_pca.set_xlabel("PC1")
        self.ax_pca.set_ylabel("PC2")

        if self.X is not None:
            # PCA on unit-normalized data is common for visualization
            X_unit = row_normalize_unit(self.X)
            X2 = pca_2d(X_unit)

            if self.labels is None:
                # no clusters yet: plot all points same style
                self.ax_pca.scatter(X2[:, 0], X2[:, 1], s=10)
            else:
                k = int(np.max(self.labels)) + 1
                for j in range(k):
                    mask = (self.labels == j)
                    self.ax_pca.scatter(X2[mask, 0], X2[mask, 1], s=10, label=f"cluster {j}")
                self.ax_pca.legend(loc="best", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def _refresh_summary(self):
        # Clear
        for row in self.summary_tree.get_children():
            self.summary_tree.delete(row)

        if self.labels is None:
            return

        # Count cluster sizes
        counts = pd.Series(self.labels).value_counts().sort_index()
        for cluster_id, cnt in counts.items():
            self.summary_tree.insert("", "end", values=(int(cluster_id), int(cnt)))


# =============================================================================
# 5) Main
# =============================================================================

if __name__ == "__main__":
    app = SphericalKMeansGUI()
    app.mainloop()
