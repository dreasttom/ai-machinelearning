"""
DENCLUE (DENsity-based CLUstEring) — Educational, heavily-commented implementation + GUI

What this script includes:
- A from-scratch, student-friendly DENCLUE implementation using Gaussian kernels.
- Gradient-ascent "hill climbing" to find density attractors.
- Merging nearby attractors into clusters.
- A Tkinter GUI with:
  - synthetic data generation (blobs / rings / moons-ish),
  - optional CSV loading,
  - interactive parameters,
  - matplotlib visualization embedded in the window,
  - clustering summary + noise count.

Install notes:
- Requires: numpy, matplotlib
  pip install numpy matplotlib

CSV format:
- Two columns (x,y). Header is allowed. Extra columns are ignored.
"""

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
#                    DENCLUE CORE (EDUCATIONAL)
# ============================================================

def gaussian_kernel(u2: np.ndarray) -> np.ndarray:
    """
    Gaussian kernel value for squared distances u^2.
    We use exp(-u^2 / 2). Any missing constant factor cancels out in gradients.
    """
    return np.exp(-0.5 * u2)


def kde_density_and_gradient(X: np.ndarray, x: np.ndarray, sigma: float) -> tuple[float, np.ndarray]:
    """
    Compute KDE density estimate f(x) and its gradient ∇f(x) under Gaussian kernels.

    KDE idea (informal):
      f(x) = sum_i K( ||x - X_i|| / sigma )

    For Gaussian kernel K(t) = exp(-t^2/2), gradient is:
      ∇f(x) = sum_i [ K_i * (-(x - X_i) / sigma^2) ]

    We'll return:
      - density f(x) (not normalized by constants, fine for clustering)
      - gradient ∇f(x)
    """
    diffs = x[None, :] - X  # shape: (n, d)
    u2 = np.sum((diffs / sigma) ** 2, axis=1)  # squared normalized distances
    w = gaussian_kernel(u2)  # weights, shape: (n,)

    density = float(np.sum(w))
    # Gradient: sum_i w_i * (-(x - X_i) / sigma^2)
    grad = -np.sum(w[:, None] * diffs, axis=0) / (sigma ** 2)

    return density, grad


def hill_climb_to_attractor(
    X: np.ndarray,
    x0: np.ndarray,
    sigma: float,
    step: float,
    grad_tol: float,
    max_iter: int
) -> tuple[np.ndarray, float, int]:
    """
    Move x along gradient ascent on KDE density until convergence (density attractor).

    Parameters:
      X        : data points (n,d)
      x0       : starting point (d,)
      sigma    : kernel bandwidth
      step     : learning rate / step size for gradient ascent
      grad_tol : stop when gradient magnitude is small
      max_iter : safety cap

    Returns:
      attractor: converged point
      density  : density at attractor
      iters    : number of iterations used
    """
    x = x0.astype(float).copy()

    for it in range(max_iter):
        density, grad = kde_density_and_gradient(X, x, sigma)
        gnorm = np.linalg.norm(grad)

        # If gradient is tiny, we're near a local maximum / stationary point.
        if gnorm < grad_tol:
            return x, density, it + 1

        # Gradient ascent update
        x = x + step * grad

    # If we hit max_iter, return what we have (still useful for teaching).
    density, _ = kde_density_and_gradient(X, x, sigma)
    return x, density, max_iter


def denclue(
    X: np.ndarray,
    sigma: float = 1.0,
    step: float = 0.1,
    grad_tol: float = 1e-3,
    max_iter: int = 200,
    min_density: float = 10.0,
    merge_radius: float = 0.5
) -> dict:
    """
    DENCLUE clustering (simplified educational version).

    High-level sketch:
    1) For each data point x_i, run hill-climbing to find its density attractor a_i.
    2) If the attractor density is below min_density => label as NOISE.
    3) Merge attractors that are close (within merge_radius) into the same cluster.
    4) Points inherit the cluster label of their attractor.

    Returns a dictionary with:
      labels         : (n,) integer labels (-1 for noise)
      attractors     : (n,2) attractor per point
      attractor_ids  : (n,) id after merging
      attractor_info : list of dicts (cluster center, size, avg density)
    """
    n, d = X.shape

    # --- Step 1: find attractor for each point ---
    attractors = np.zeros_like(X, dtype=float)
    densities = np.zeros(n, dtype=float)
    iters_used = np.zeros(n, dtype=int)

    for i in range(n):
        a, dens, iters = hill_climb_to_attractor(
            X=X,
            x0=X[i],
            sigma=sigma,
            step=step,
            grad_tol=grad_tol,
            max_iter=max_iter
        )
        attractors[i] = a
        densities[i] = dens
        iters_used[i] = iters

    # --- Step 2: mark low-density points as noise (temporarily) ---
    is_noise = densities < min_density

    # --- Step 3: merge attractors into clusters ---
    # We'll do a simple incremental "prototype" merge:
    # Keep a list of cluster centers; each attractor joins the first within merge_radius.
    # Otherwise it creates a new cluster.
    centers = []
    members = []  # list of lists of indices

    attractor_cluster = np.full(n, -1, dtype=int)

    for i in range(n):
        if is_noise[i]:
            continue

        a = attractors[i]
        assigned = False

        for c_idx, c in enumerate(centers):
            if np.linalg.norm(a - c) <= merge_radius:
                attractor_cluster[i] = c_idx
                members[c_idx].append(i)
                assigned = True
                break

        if not assigned:
            centers.append(a.copy())
            members.append([i])
            attractor_cluster[i] = len(centers) - 1

    # Recompute centers as mean of member attractors (helps stability)
    for c_idx in range(len(centers)):
        idxs = members[c_idx]
        centers[c_idx] = np.mean(attractors[idxs], axis=0)

    # --- Step 4: label points by their cluster, noise = -1 ---
    labels = np.full(n, -1, dtype=int)
    for i in range(n):
        if is_noise[i]:
            labels[i] = -1
        else:
            labels[i] = attractor_cluster[i]

    # Build some friendly summary info
    attractor_info = []
    for c_idx in range(len(centers)):
        idxs = np.where(labels == c_idx)[0]
        attractor_info.append({
            "cluster": c_idx,
            "center": centers[c_idx],
            "count": int(len(idxs)),
            "avg_density": float(np.mean(densities[idxs])) if len(idxs) else 0.0,
            "avg_iters": float(np.mean(iters_used[idxs])) if len(idxs) else 0.0
        })

    return {
        "labels": labels,
        "attractors": attractors,
        "densities": densities,
        "iters_used": iters_used,
        "attractor_ids": attractor_cluster,
        "attractor_info": attractor_info
    }


# ============================================================
#                    DATA GENERATION HELPERS
# ============================================================

def make_blobs(n=400, centers=3, spread=0.8, seed=0) -> np.ndarray:
    """
    Simple blob generator without scikit-learn.
    """
    rng = np.random.default_rng(seed)
    # Pick random center locations
    C = rng.uniform(-5, 5, size=(centers, 2))
    # Assign each point a center
    which = rng.integers(0, centers, size=n)
    X = C[which] + rng.normal(0, spread, size=(n, 2))
    return X


def make_rings(n=500, rings=2, noise=0.08, seed=0) -> np.ndarray:
    """
    Concentric rings. Useful for showing that DENCLUE can handle non-convex shapes.
    """
    rng = np.random.default_rng(seed)
    X = []
    for r in range(1, rings + 1):
        angles = rng.uniform(0, 2 * math.pi, size=n // rings)
        radius = r * 2.0 + rng.normal(0, 0.15, size=angles.shape[0])
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.column_stack([x, y]) + rng.normal(0, noise, size=(angles.shape[0], 2))
        X.append(pts)
    return np.vstack(X)


def make_two_moons_like(n=500, noise=0.08, seed=0) -> np.ndarray:
    """
    A moons-like dataset (approximation, no sklearn).
    """
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    t1 = rng.uniform(0, math.pi, size=n1)
    t2 = rng.uniform(0, math.pi, size=n2)

    moon1 = np.column_stack([np.cos(t1), np.sin(t1)])
    moon2 = np.column_stack([1 - np.cos(t2), 0.5 - np.sin(t2)])

    X = np.vstack([moon1, moon2])
    X = X * 3.0 + rng.normal(0, noise, size=X.shape)
    return X


def load_csv_xy(path: str) -> np.ndarray:
    """
    Load a CSV containing at least two columns (x,y).
    - Allows header.
    - Ignores extra columns.
    """
    data = np.genfromtxt(path, delimiter=",", dtype=float, invalid_raise=False)
    if data.ndim == 1:
        # Single row
        data = data[None, :]

    # If header caused NaNs in first row, try skipping first row
    if np.any(np.isnan(data[0, :2])):
        data = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1, invalid_raise=False)
        if data.ndim == 1:
            data = data[None, :]

    if data.shape[1] < 2:
        raise ValueError("CSV must have at least 2 numeric columns (x,y).")

    X = data[:, :2]
    # Drop rows with NaN in first two columns
    X = X[~np.any(np.isnan(X), axis=1)]
    if len(X) == 0:
        raise ValueError("No valid numeric (x,y) rows found in the CSV.")
    return X


# ============================================================
#                           GUI APP
# ============================================================

class DenclueGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DENCLUE Clustering (Educational) — Tkinter GUI")
        self.geometry("1150x720")

        # Current dataset
        self.X = make_blobs(n=450, centers=3, spread=0.9, seed=0)
        self.result = None

        # ---- Layout: left control panel + right plot panel ----
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.ctrl = ttk.Frame(self, padding=10)
        self.ctrl.grid(row=0, column=0, sticky="nsw")
        self.ctrl.columnconfigure(0, weight=1)

        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self._build_controls()
        self._build_plot()

        # Initial draw
        self._draw_points()

    def _build_controls(self):
        # Dataset section
        ds_lab = ttk.LabelFrame(self.ctrl, text="Dataset", padding=10)
        ds_lab.grid(row=0, column=0, sticky="ew")
        ds_lab.columnconfigure(1, weight=1)

        self.dataset_type = tk.StringVar(value="Blobs")

        ttk.Label(ds_lab, text="Type:").grid(row=0, column=0, sticky="w")
        ds_combo = ttk.Combobox(ds_lab, textvariable=self.dataset_type,
                                values=["Blobs", "Rings", "Moons-like"], state="readonly", width=14)
        ds_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Button(ds_lab, text="Generate", command=self.on_generate).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(ds_lab, text="Load CSV (x,y)", command=self.on_load_csv).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # Parameter section
        p_lab = ttk.LabelFrame(self.ctrl, text="DENCLUE Parameters", padding=10)
        p_lab.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for c in range(2):
            p_lab.columnconfigure(c, weight=1)

        # NOTE: DENCLUE has multiple parameterizations in literature.
        # Here we expose the most intuitive for students:
        #   sigma        = kernel bandwidth (smoothness)
        #   step         = gradient-ascent step size
        #   grad_tol     = convergence tolerance
        #   max_iter     = iteration cap
        #   min_density  = density threshold (noise filter)
        #   merge_radius = how close attractors must be to merge into one cluster

        self.var_sigma = tk.DoubleVar(value=1.2)
        self.var_step = tk.DoubleVar(value=0.12)
        self.var_grad_tol = tk.DoubleVar(value=1e-3)
        self.var_max_iter = tk.IntVar(value=200)
        self.var_min_density = tk.DoubleVar(value=25.0)
        self.var_merge_radius = tk.DoubleVar(value=0.7)

        self._add_labeled_entry(p_lab, "sigma (bandwidth)", self.var_sigma, 0)
        self._add_labeled_entry(p_lab, "step (learning rate)", self.var_step, 1)
        self._add_labeled_entry(p_lab, "grad_tol (stop)", self.var_grad_tol, 2)
        self._add_labeled_entry(p_lab, "max_iter", self.var_max_iter, 3)
        self._add_labeled_entry(p_lab, "min_density (noise)", self.var_min_density, 4)
        self._add_labeled_entry(p_lab, "merge_radius", self.var_merge_radius, 5)

        ttk.Button(self.ctrl, text="Run DENCLUE", command=self.on_run, width=20).grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(self.ctrl, text="Show Attractors", command=self.on_toggle_attractors).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(self.ctrl, text="Reset View", command=self._draw_points).grid(row=4, column=0, sticky="ew", pady=(6, 0))

        # Output / explanation box
        out_lab = ttk.LabelFrame(self.ctrl, text="Output", padding=10)
        out_lab.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        self.ctrl.rowconfigure(5, weight=1)

        self.output = tk.Text(out_lab, height=18, wrap="word")
        self.output.pack(fill="both", expand=True)

        self.show_attractors = False
        self._write_output(
            "Tip: Start with Blobs. If you see too many tiny clusters, increase merge_radius or min_density.\n"
            "If everything becomes one cluster, decrease sigma or merge_radius.\n"
        )

    def _add_labeled_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        e = ttk.Entry(parent, textvariable=var, width=12)
        e.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)

    def _build_plot(self):
        # Matplotlib figure embedded into Tkinter
        self.fig = Figure(figsize=(7.5, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Data (unclustered)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _write_output(self, text):
        self.output.insert("end", text + "\n")
        self.output.see("end")

    # -------------------- Actions --------------------

    def on_generate(self):
        kind = self.dataset_type.get()
        if kind == "Blobs":
            self.X = make_blobs(n=500, centers=4, spread=0.85, seed=0)
        elif kind == "Rings":
            self.X = make_rings(n=520, rings=3, noise=0.10, seed=1)
        else:
            self.X = make_two_moons_like(n=520, noise=0.12, seed=2)

        self.result = None
        self.show_attractors = False
        self._write_output(f"Generated dataset: {kind}  (n={len(self.X)})")
        self._draw_points()

    def on_load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file with x,y columns",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.X = load_csv_xy(path)
            self.result = None
            self.show_attractors = False
            self._write_output(f"Loaded CSV: {path}  (n={len(self.X)})")
            self._draw_points()
        except Exception as e:
            messagebox.showerror("CSV Load Error", str(e))

    def on_run(self):
        if self.X is None or len(self.X) == 0:
            messagebox.showwarning("No data", "Generate or load a dataset first.")
            return

        # Pull parameters from GUI
        try:
            sigma = float(self.var_sigma.get())
            step = float(self.var_step.get())
            grad_tol = float(self.var_grad_tol.get())
            max_iter = int(self.var_max_iter.get())
            min_density = float(self.var_min_density.get())
            merge_radius = float(self.var_merge_radius.get())

            if sigma <= 0 or step <= 0 or grad_tol <= 0 or max_iter <= 0 or merge_radius <= 0:
                raise ValueError("All parameters must be positive.")
        except Exception as e:
            messagebox.showerror("Parameter error", f"Invalid parameter value.\n\n{e}")
            return

        self._write_output("Running DENCLUE... (hill-climbing each point)")
        self.result = denclue(
            X=self.X,
            sigma=sigma,
            step=step,
            grad_tol=grad_tol,
            max_iter=max_iter,
            min_density=min_density,
            merge_radius=merge_radius
        )

        labels = self.result["labels"]
        n_noise = int(np.sum(labels == -1))
        n_clusters = int(labels.max() + 1) if np.any(labels >= 0) else 0

        self._write_output(f"Done. clusters={n_clusters}, noise={n_noise}")
        for info in self.result["attractor_info"]:
            c = info["cluster"]
            center = info["center"]
            self._write_output(
                f"  Cluster {c}: count={info['count']}, "
                f"avg_density={info['avg_density']:.2f}, "
                f"avg_iters={info['avg_iters']:.1f}, "
                f"center≈({center[0]:.2f},{center[1]:.2f})"
            )

        self._draw_clusters()

    def on_toggle_attractors(self):
        self.show_attractors = not self.show_attractors
        if self.result is None:
            self._draw_points()
        else:
            self._draw_clusters()

    # -------------------- Drawing --------------------

    def _draw_points(self):
        self.ax.clear()
        self.ax.scatter(self.X[:, 0], self.X[:, 1], s=18)
        self.ax.set_title("Data (unclustered)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw()

    def _draw_clusters(self):
        self.ax.clear()

        labels = self.result["labels"]
        X = self.X

        # Plot noise first
        noise_idx = (labels == -1)
        if np.any(noise_idx):
            self.ax.scatter(X[noise_idx, 0], X[noise_idx, 1], s=18, marker="x", label="noise")

        # Plot each cluster
        clusters = sorted(set(labels.tolist()) - {-1})
        for c in clusters:
            idx = (labels == c)
            self.ax.scatter(X[idx, 0], X[idx, 1], s=18, label=f"cluster {c}")

        # Optionally show attractors
        if self.show_attractors:
            A = self.result["attractors"]
            # thin lines from point -> attractor can be very dense, so keep it lightweight:
            # We'll just plot attractor points (small dots)
            self.ax.scatter(A[:, 0], A[:, 1], s=8, alpha=0.7, label="attractors")

        n_clusters = len(clusters)
        n_noise = int(np.sum(labels == -1))
        self.ax.set_title(f"DENCLUE Result — clusters={n_clusters}, noise={n_noise}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best", fontsize=9)
        self.canvas.draw()


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    app = DenclueGUI()
    app.mainloop()
