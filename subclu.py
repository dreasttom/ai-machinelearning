"""
SUBCLU (Subspace Density Clustering) - Student-Friendly + GUI (Tkinter)
======================================================================

This script implements a *teachable* (and reasonably faithful) version of **SUBCLU**:
- SUBCLU discovers **DBSCAN-style density clusters in many subspaces** of a dataset.
- It uses an **Apriori-style candidate generation**:
    If a subspace has clusters, larger subspaces built from it are candidates too.

What this GUI does:
- Load the attached naturalgas.csv (default path: /mnt/data/naturalgas.csv) or any CSV.
- Choose feature columns (numeric and/or categorical).
- Categorical columns can be one-hot encoded.
- Optionally standardize numeric columns (recommended).
- Set DBSCAN parameters (eps, minPts).
- Set maximum subspace dimension to explore.
- Run SUBCLU and see:
    - which subspaces had clusters
    - number of clusters per subspace
    - cluster sizes
- Save a summary CSV of discovered clusters.

Important teaching notes:
- SUBCLU relies on **DBSCAN**, so we implement DBSCAN "from scratch" (no sklearn).
- This implementation is **naive O(n^2)** for neighborhood queries to keep it readable.
  It works best on small/medium datasets or limited feature selections.
- SUBCLU can return *many* clusters/subspaces; use max_subspace_dim to control complexity.

Dependencies:
    pip install numpy pandas matplotlib

Run:
    python subclu_gui.py
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
# 1) Preprocessing helpers (one-hot, standardize, etc.)
# =============================================================================

def standardize_columns(X: np.ndarray, eps: float = 1e-12):
    """
    Standardize each column:
        X_std = (X - mean) / std
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X - mean) / std, mean, std


def build_feature_matrix(df: pd.DataFrame,
                         selected_cols,
                         standardize_numeric: bool = True,
                         one_hot_categoricals: bool = True):
    """
    Convert selected columns into a purely numeric feature matrix.

    Strategy:
    - Numeric columns: keep numeric (optionally standardize)
    - Categorical columns: one-hot encode (optional)

    Returns:
        X (np.ndarray), feature_names (list[str])
    """
    if not selected_cols:
        raise ValueError("No columns selected.")

    work = df[selected_cols].copy()

    # Simple missing-value handling (student-friendly)
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].astype(float)
            if work[col].isna().any():
                work[col] = work[col].fillna(work[col].mean())
        else:
            work[col] = work[col].astype(str)
            if work[col].isna().any():
                work[col] = work[col].fillna("MISSING")

    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    cat_cols = [c for c in work.columns if c not in numeric_cols]

    X_parts = []
    feature_names = []

    if numeric_cols:
        X_num = work[numeric_cols].to_numpy(dtype=float)
        if standardize_numeric:
            X_num, _, _ = standardize_columns(X_num)
        X_parts.append(X_num)
        feature_names.extend(numeric_cols)

    if cat_cols and one_hot_categoricals:
        dummies = pd.get_dummies(work[cat_cols], columns=cat_cols, drop_first=False)
        X_cat = dummies.to_numpy(dtype=float)
        X_parts.append(X_cat)
        feature_names.extend(list(dummies.columns))
    elif cat_cols and not one_hot_categoricals:
        raise ValueError("Categorical columns selected but one-hot encoding is OFF. Either enable one-hot or deselect categoricals.")

    if not X_parts:
        raise ValueError("No usable features created. Check your column selection.")

    X = np.concatenate(X_parts, axis=1)
    return X, feature_names


# =============================================================================
# 2) DBSCAN from scratch (used by SUBCLU)
# =============================================================================

def pairwise_distances(X: np.ndarray):
    """
    Compute full pairwise Euclidean distance matrix (N x N).

    This is O(N^2) memory + time.
    It is simple and great for teaching, but not for huge datasets.

    dist(i,j) = ||X[i] - X[j]||
    """
    # Broadcasting trick:
    # (N,1,D) - (1,N,D) -> (N,N,D), then norm across D
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def dbscan_from_scratch(X: np.ndarray, eps: float, min_pts: int):
    """
    Classic DBSCAN algorithm.

    Definitions:
    - Neighborhood of point i: all points within distance eps.
    - Core point: has at least min_pts points in its neighborhood (including itself).
    - Cluster: connected components of core points + reachable border points.

    Returns:
        labels: (N,) where -1 = noise, 0..k-1 are cluster ids
    """
    N = X.shape[0]
    dist = pairwise_distances(X)  # (N,N)
    neighbors = [np.where(dist[i] <= eps)[0] for i in range(N)]

    labels = np.full(N, -1, dtype=int)   # start with all noise
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True

        Ni = neighbors[i]
        if len(Ni) < min_pts:
            # Not a core point => remains noise for now (could later become border)
            continue

        # Start a new cluster
        labels[i] = cluster_id

        # Expand the cluster using a queue (breadth-first style)
        queue = list(Ni)
        q_index = 0
        while q_index < len(queue):
            j = queue[q_index]
            q_index += 1

            if not visited[j]:
                visited[j] = True
                Nj = neighbors[j]
                if len(Nj) >= min_pts:
                    # If j is a core point, add its neighbors to the queue
                    for p in Nj:
                        if p not in queue:
                            queue.append(p)

            # Assign cluster label if unassigned
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


# =============================================================================
# 3) SUBCLU (teachable implementation)
# =============================================================================

def subspace_key(subspace):
    """
    Represent a subspace (tuple of feature indices) as a stable key.
    """
    return tuple(sorted(subspace))


def all_k_minus_1_subspaces(subspace):
    """
    Return all (k-1)-subspaces of a given k-subspace.
    Example: (0,2,5) -> [(2,5), (0,5), (0,2)]
    """
    s = list(subspace)
    out = []
    for i in range(len(s)):
        subset = s[:i] + s[i+1:]
        out.append(tuple(subset))
    return out


def generate_candidates(prev_frequent, k):
    """
    Apriori candidate generation:
    - Join pairs of (k-1)-subspaces that share first k-2 elements.
    - Prune if any (k-1)-subset of candidate isn't in prev_frequent.

    prev_frequent: set of tuples (each length k-1)
    returns: set of tuples (each length k)
    """
    prev_list = sorted(list(prev_frequent))
    candidates = set()

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            a = prev_list[i]
            b = prev_list[j]

            # Join rule: first k-2 items must match
            if a[:k-2] == b[:k-2]:
                cand = tuple(sorted(set(a) | set(b)))
                if len(cand) != k:
                    continue

                # Prune: all (k-1)-subspaces must be frequent
                subsets = all_k_minus_1_subspaces(cand)
                if all(tuple(sorted(ss)) in prev_frequent for ss in subsets):
                    candidates.add(cand)
            else:
                # Because list is sorted, if prefixes differ we can break early
                break

    return candidates


class SUBCLU:
    """
    SUBCLU discovers density clusters in many subspaces.

    High-level SUBCLU idea (teaching version):
    1) Find DBSCAN clusters in all 1D subspaces (each feature alone).
       Keep only subspaces where at least 1 cluster exists (not all noise).
    2) Grow to 2D, 3D, ... using Apriori.
    3) For a k-dimensional candidate subspace S:
       - We restrict the points to those that already showed "cluster potential"
         in its (k-1) subsets (this is the SUBCLU speed/consistency idea).
       - Run DBSCAN in that subspace on those candidate points.
       - Keep if clusters exist.

    Output:
      clusters[subspace] = list of clusters
      where each cluster is a list/array of point indices.
    """

    def __init__(self, eps: float, min_pts: int, max_dim: int = 3):
        self.eps = float(eps)
        self.min_pts = int(min_pts)
        self.max_dim = int(max_dim)

        self.clusters_ = {}      # subspace(tuple) -> list of np arrays of indices
        self.frequent_ = set()   # subspaces with clusters

    @staticmethod
    def _labels_to_clusters(labels):
        """
        Convert DBSCAN labels into clusters of indices.
        Ignore noise label -1.
        """
        clusters = []
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            idx = np.where(labels == cid)[0]
            if len(idx) > 0:
                clusters.append(idx)
        return clusters

    def fit(self, X: np.ndarray):
        """
        Run SUBCLU on X (N x D).
        """
        N, D = X.shape
        self.clusters_.clear()
        self.frequent_.clear()

        # ---- Step 1: 1D subspaces ----
        for d in range(D):
            sub = (d,)
            Xs = X[:, [d]]  # 1D column
            labels = dbscan_from_scratch(Xs, self.eps, self.min_pts)
            clusters = self._labels_to_clusters(labels)
            if clusters:
                self.frequent_.add(sub)
                self.clusters_[sub] = clusters

        # ---- Step 2: grow subspaces k=2..max_dim ----
        k = 2
        prev_frequent = set(sorted(self.frequent_))
        while k <= self.max_dim and prev_frequent:
            candidates = generate_candidates(prev_frequent, k)

            new_frequent = set()

            for sub in sorted(candidates):
                # Candidate pruning via "cluster potential":
                # For each (k-1)-subset, collect points that are in any cluster in that subset.
                # Then intersect these sets across all (k-1)-subsets.
                subsets = all_k_minus_1_subspaces(sub)

                point_sets = []
                valid = True
                for ss in subsets:
                    ss = subspace_key(ss)
                    if ss not in self.clusters_:
                        valid = False
                        break
                    pts = np.concatenate(self.clusters_[ss])  # union of cluster points in that subset
                    point_sets.append(set(pts.tolist()))

                if not valid or not point_sets:
                    continue

                # Intersect points across subsets
                candidate_points = set.intersection(*point_sets)
                if len(candidate_points) < self.min_pts:
                    continue  # cannot form a DBSCAN core anyway

                cand_idx = np.array(sorted(candidate_points), dtype=int)
                Xs = X[cand_idx][:, list(sub)]  # data restricted to points and subspace

                labels = dbscan_from_scratch(Xs, self.eps, self.min_pts)
                clusters_local = self._labels_to_clusters(labels)

                # Convert cluster indices from local (0..len(cand_idx)-1) to global point indices
                clusters_global = [cand_idx[c] for c in clusters_local]

                if clusters_global:
                    new_frequent.add(sub)
                    self.clusters_[sub] = clusters_global

            prev_frequent = new_frequent
            self.frequent_ |= new_frequent
            k += 1

        return self


# =============================================================================
# 4) GUI Application
# =============================================================================

class SUBCLU_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SUBCLU (Subspace Density Clustering) - Student GUI")
        self.geometry("1280x820")

        self.df = None
        self.csv_path = None

        self.X = None
        self.feature_names = None

        self.model = None
        self.results = {}  # subspace -> list of clusters (each is array of indices)

        self._build_ui()

        # Auto-load attached file if available in the environment
        default_path = "/mnt/data/naturalgas.csv"
        if os.path.exists(default_path):
            try:
                self.load_csv(default_path)
            except Exception:
                pass

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---- Left panel: data + parameters ----
        ttk.Label(left, text="Data", font=("Arial", 12, "bold")).pack(anchor="w")
        ttk.Button(left, text="Load CSV", command=self.load_csv_dialog).pack(fill=tk.X, pady=4)
        self.file_label = ttk.Label(left, text="No file loaded.")
        self.file_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Select columns to build features:", font=("Arial", 11, "bold")).pack(anchor="w")
        self.cols_list = tk.Listbox(left, selectmode=tk.MULTIPLE, height=12)
        self.cols_list.pack(fill=tk.X, pady=4)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Preprocessing", font=("Arial", 12, "bold")).pack(anchor="w")

        self.one_hot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="One-hot encode categorical columns", variable=self.one_hot_var).pack(anchor="w")

        self.std_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Standardize numeric columns", variable=self.std_var).pack(anchor="w")

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="DBSCAN Parameters", font=("Arial", 12, "bold")).pack(anchor="w")

        ttk.Label(left, text="eps (radius):").pack(anchor="w")
        self.eps_entry = ttk.Entry(left)
        self.eps_entry.insert(0, "0.8")
        self.eps_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="minPts:").pack(anchor="w")
        self.minpts_entry = ttk.Entry(left)
        self.minpts_entry.insert(0, "5")
        self.minpts_entry.pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="SUBCLU Controls", font=("Arial", 12, "bold")).pack(anchor="w")

        ttk.Label(left, text="Max subspace dimension:").pack(anchor="w")
        self.maxdim_entry = ttk.Entry(left)
        self.maxdim_entry.insert(0, "3")
        self.maxdim_entry.pack(fill=tk.X, pady=2)

        ttk.Button(left, text="Run SUBCLU", command=self.run_subclu).pack(fill=tk.X, pady=(10, 2))
        ttk.Button(left, text="Save summary CSV", command=self.save_summary_csv).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Logs", font=("Arial", 12, "bold")).pack(anchor="w")
        self.log_text = tk.Text(left, height=18, width=46)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ---- Right panel: tabs ----
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_summary = ttk.Frame(self.nb, padding=10)
        self.tab_detail = ttk.Frame(self.nb, padding=10)
        self.tab_plot = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_summary, text="Subspace Summary")
        self.nb.add(self.tab_detail, text="Cluster Detail")
        self.nb.add(self.tab_plot, text="Quick Plot (2D subspaces)")

        self._build_summary_tab()
        self._build_detail_tab()
        self._build_plot_tab()

    def _build_summary_tab(self):
        ttk.Label(self.tab_summary, text="Subspaces with clusters discovered by SUBCLU", font=("Arial", 12, "bold")).pack(anchor="w")

        cols = ("Subspace", "Dim", "#clusters", "Total clustered pts")
        self.summary_tree = ttk.Treeview(self.tab_summary, columns=cols, show="headings", height=20)
        for c in cols:
            self.summary_tree.heading(c, text=c)
            self.summary_tree.column(c, width=200, anchor="center")
        self.summary_tree.column("Subspace", width=420, anchor="w")

        self.summary_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.summary_tree.bind("<<TreeviewSelect>>", self.on_select_subspace)

    def _build_detail_tab(self):
        top = ttk.Frame(self.tab_detail)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Selected subspace:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.selected_subspace_var = tk.StringVar(value="(none)")
        ttk.Label(top, textvariable=self.selected_subspace_var).pack(side=tk.LEFT, padx=8)

        cols = ("Cluster ID", "Size")
        self.cluster_tree = ttk.Treeview(self.tab_detail, columns=cols, show="headings", height=20)
        for c in cols:
            self.cluster_tree.heading(c, text=c)
            self.cluster_tree.column(c, width=180, anchor="center")
        self.cluster_tree.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_plot_tab(self):
        ttk.Label(self.tab_plot, text="2D subspace scatter (first two dims of chosen subspace)", font=("Arial", 12, "bold")).pack(anchor="w")

        self.fig = Figure(figsize=(8.0, 6.2), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        note = (
            "Teaching note:\n"
            "- SUBCLU finds clusters in many dimensions.\n"
            "- If a subspace has >2 dims, we plot only the first two for a quick view.\n"
            "- If your chosen feature set is high-dimensional, this plot is only a partial glimpse."
        )
        ttk.Label(self.tab_plot, text=note, justify="left").pack(anchor="w", pady=(8, 0))

    # ---------------- Logging ----------------

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # ---------------- CSV loading ----------------

    def load_csv_dialog(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.load_csv(path)

    def load_csv(self, path):
        df = pd.read_csv(path)
        self.df = df
        self.csv_path = path
        self.file_label.configure(text=f"Loaded: {os.path.basename(path)}  ({df.shape[0]} rows, {df.shape[1]} cols)")
        self.log(f"Loaded CSV: {path}")

        # Populate columns list
        self.cols_list.delete(0, tk.END)
        for c in df.columns:
            self.cols_list.insert(tk.END, c)

        # Helpful default selection for naturalgas-like data
        defaults = ["year", "month", "value", "area-name", "process-name"]
        for i, c in enumerate(df.columns):
            if c in defaults:
                self.cols_list.selection_set(i)

        # Clear previous results
        self.results = {}
        self._refresh_summary()
        self._refresh_detail(None)
        self._refresh_plot(None)

    def _get_selected_cols(self):
        idxs = self.cols_list.curselection()
        return [self.cols_list.get(i) for i in idxs]

    # ---------------- SUBCLU run ----------------

    def run_subclu(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return

        try:
            selected_cols = self._get_selected_cols()
            if not selected_cols:
                raise ValueError("Select at least one column.")

            eps = float(self.eps_entry.get().strip())
            min_pts = int(self.minpts_entry.get().strip())
            max_dim = int(self.maxdim_entry.get().strip())

            if eps <= 0:
                raise ValueError("eps must be > 0.")
            if min_pts < 2:
                raise ValueError("minPts should be >= 2 (common values are 4..10).")
            if max_dim < 1:
                raise ValueError("max subspace dimension must be >= 1.")

            self.log("Building feature matrix...")
            X, feature_names = build_feature_matrix(
                self.df,
                selected_cols=selected_cols,
                standardize_numeric=bool(self.std_var.get()),
                one_hot_categoricals=bool(self.one_hot_var.get())
            )
            self.X = X
            self.feature_names = feature_names

            D = X.shape[1]
            if max_dim > D:
                self.log(f"Note: max_dim={max_dim} > #features={D}. Using max_dim={D}.")
                max_dim = D

            self.log(f"Feature matrix: N={X.shape[0]}, D={X.shape[1]}")
            self.log("Running SUBCLU (this may take time if N or D is large)...")

            model = SUBCLU(eps=eps, min_pts=min_pts, max_dim=max_dim)
            model.fit(X)

            self.model = model
            self.results = model.clusters_

            self.log(f"Done. Subspaces with clusters: {len(self.results)}")
            if len(self.results) == 0:
                self.log("No clusters found. Try increasing eps or lowering minPts, or select fewer columns.")

            self._refresh_summary()
            self._refresh_detail(None)
            self._refresh_plot(None)

        except Exception as e:
            messagebox.showerror("SUBCLU error", str(e))

    # ---------------- UI refresh methods ----------------

    def _refresh_summary(self):
        for row in self.summary_tree.get_children():
            self.summary_tree.delete(row)

        if not self.results:
            return

        # Build summary rows
        for subspace, clusters in sorted(self.results.items(), key=lambda kv: (len(kv[0]), kv[0])):
            dim = len(subspace)
            n_clusters = len(clusters)
            total_pts = int(sum(len(c) for c in clusters))

            # Convert indices -> readable feature names
            name = self.subspace_to_name(subspace)
            self.summary_tree.insert("", "end", values=(name, dim, n_clusters, total_pts))

    def subspace_to_name(self, subspace):
        # subspace is indices into self.feature_names
        if self.feature_names is None:
            return str(subspace)
        return "(" + ", ".join(self.feature_names[i] for i in subspace) + ")"

    def on_select_subspace(self, event):
        sel = self.summary_tree.selection()
        if not sel or not self.results:
            return

        # We stored "Subspace" as a string name. To map back, we search.
        chosen_name = self.summary_tree.item(sel[0], "values")[0]

        # Find the subspace tuple that matches this name
        chosen_sub = None
        for sub in self.results.keys():
            if self.subspace_to_name(sub) == chosen_name:
                chosen_sub = sub
                break

        self._refresh_detail(chosen_sub)
        self._refresh_plot(chosen_sub)

    def _refresh_detail(self, subspace):
        for row in self.cluster_tree.get_children():
            self.cluster_tree.delete(row)

        if subspace is None:
            self.selected_subspace_var.set("(none)")
            return

        self.selected_subspace_var.set(self.subspace_to_name(subspace))
        clusters = self.results.get(subspace, [])
        for cid, c in enumerate(clusters):
            self.cluster_tree.insert("", "end", values=(cid, int(len(c))))

    def _refresh_plot(self, subspace):
        self.ax.clear()
        self.ax.set_title("No subspace selected yet.")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")

        if subspace is None or self.X is None or self.feature_names is None:
            self.canvas.draw()
            return

        clusters = self.results.get(subspace, [])
        if not clusters:
            self.canvas.draw()
            return

        # Decide which 2 dimensions to plot:
        dims = list(subspace)
        if len(dims) == 1:
            # Plot 1D on x-axis and zeros on y
            x_idx = dims[0]
            x = self.X[:, x_idx]
            self.ax.set_title(f"1D subspace: {self.feature_names[x_idx]}")
            self.ax.scatter(x, np.zeros_like(x), s=10)
            self.ax.set_xlabel(self.feature_names[x_idx])
            self.ax.set_ylabel("(constant)")
            self.canvas.draw()
            return

        x_idx, y_idx = dims[0], dims[1]
        self.ax.set_title(f"Subspace (showing first 2 dims): {self.feature_names[x_idx]} vs {self.feature_names[y_idx]}")
        self.ax.set_xlabel(self.feature_names[x_idx])
        self.ax.set_ylabel(self.feature_names[y_idx])

        # Build a label array for points in clusters; noise as -1
        labels = np.full(self.X.shape[0], -1, dtype=int)
        for cid, c in enumerate(clusters):
            labels[c] = cid

        # Plot each cluster
        for cid in range(len(clusters)):
            mask = labels == cid
            self.ax.scatter(self.X[mask, x_idx], self.X[mask, y_idx], s=12, label=f"cluster {cid}")

        # Plot noise points lightly
        noise_mask = labels == -1
        if np.any(noise_mask):
            self.ax.scatter(self.X[noise_mask, x_idx], self.X[noise_mask, y_idx], s=8, alpha=0.25, label="noise")

        self.ax.legend(loc="best", fontsize=8)
        self.canvas.draw()

    # ---------------- Saving summary ----------------

    def save_summary_csv(self):
        if not self.results:
            messagebox.showwarning("No results", "Run SUBCLU first.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save SUBCLU summary CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not out_path:
            return

        rows = []
        for subspace, clusters in sorted(self.results.items(), key=lambda kv: (len(kv[0]), kv[0])):
            subspace_name = self.subspace_to_name(subspace)
            dim = len(subspace)
            for cid, c in enumerate(clusters):
                rows.append({
                    "subspace": subspace_name,
                    "dimension": dim,
                    "cluster_id": cid,
                    "cluster_size": int(len(c)),
                    "point_indices": " ".join(map(str, c.tolist()))
                })

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.log(f"Saved summary CSV: {out_path}")


# =============================================================================
# 5) Main
# =============================================================================

if __name__ == "__main__":
    app = SUBCLU_GUI()
    app.mainloop()
