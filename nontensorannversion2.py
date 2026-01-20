"""
Neural Network From Scratch (NO TensorFlow / PyTorch)
====================================================

This version includes:
1) Multiple activation choices (ReLU / Sigmoid / Tanh / LeakyReLU)
2) Visual Backprop tab (shapes + gradient norms per layer)
3) Step-by-step single-batch mode:
   - "Sample New Batch"
   - "Forward Step"  (computes logits/probs/loss and stores cache)
   - "Backprop Step" (computes gradients and updates Visual Backprop table)
   - "Update Params" (applies gradient descent step using the stored gradients)
   - "Reset Step State"

Student idea:
- Training normally feels like a black box.
- Step Mode makes the pipeline explicit: forward -> backward -> update.

Requirements:
    pip install numpy matplotlib

Run:
    python nn_gui_step_mode.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
# 1) Utility Functions
# ============================================================

def one_hot(y_int, num_classes):
    y_int = y_int.astype(int).ravel()
    out = np.zeros((y_int.size, num_classes), dtype=float)
    out[np.arange(y_int.size), y_int] = 1.0
    return out

def train_test_split(X, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def softmax(z):
    # stable softmax
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs, y_onehot):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))

def accuracy_from_logits(logits, y_true_int):
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true_int))


# ============================================================
# 2) Activations + derivatives
# ============================================================

def relu(z): return np.maximum(0.0, z)
def relu_grad(z): return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_grad(z):
    a = sigmoid(z)
    return a * (1.0 - a)

def tanh(z): return np.tanh(z)
def tanh_grad(z):
    a = np.tanh(z)
    return 1.0 - a * a

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

ACTIVATIONS = {
    "ReLU": {
        "f": relu,
        "df": relu_grad,
        "notes": "ReLU(z)=max(0,z). Fast, common. Risk: 'dead' neurons if stuck negative."
    },
    "Sigmoid": {
        "f": sigmoid,
        "df": sigmoid_grad,
        "notes": "Sigmoid squashes to (0,1). Can saturate -> vanishing gradients."
    },
    "Tanh": {
        "f": tanh,
        "df": tanh_grad,
        "notes": "Tanh squashes to (-1,1). Often better than sigmoid; can still vanish."
    },
    "LeakyReLU": {
        "f": lambda z: leaky_relu(z, alpha=0.01),
        "df": lambda z: leaky_relu_grad(z, alpha=0.01),
        "notes": "LeakyReLU keeps small slope for negative z, reducing dead ReLU risk."
    }
}


# ============================================================
# 3) MLP Model
# ============================================================

class MLP:
    """
    MLP for multi-class classification:
        X -> Dense -> Act -> ... -> Dense -> Softmax
    """

    def __init__(self, input_dim, hidden_layers, output_dim, activation_name="ReLU", seed=0):
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation_name}")

        self.rng = np.random.default_rng(seed)
        self.hidden_layers = list(hidden_layers)
        self.act_name = activation_name
        self.act_f = ACTIVATIONS[activation_name]["f"]
        self.act_df = ACTIVATIONS[activation_name]["df"]

        layer_sizes = [input_dim] + self.hidden_layers + [output_dim]
        self.W = []
        self.b = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            is_hidden_layer = i < (len(layer_sizes) - 2)

            # Simple initialization heuristic:
            # - ReLU-like: He scale
            # - Sigmoid/Tanh hidden: slightly smaller
            if is_hidden_layer and activation_name in ("Sigmoid", "Tanh"):
                scale = np.sqrt(1.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / fan_in)

            w = self.rng.normal(0.0, scale, size=(fan_in, layer_sizes[i + 1]))
            bias = np.zeros((1, layer_sizes[i + 1]), dtype=float)
            self.W.append(w)
            self.b.append(bias)

    def forward(self, X):
        A = X
        cache = {"A": [X], "Z": []}

        # hidden layers
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_f(Z)
            cache["Z"].append(Z)
            cache["A"].append(A)

        # output logits
        Z_last = A @ self.W[-1] + self.b[-1]
        cache["Z"].append(Z_last)
        return Z_last, cache

    def backward(self, y_onehot, cache):
        """
        Softmax + cross-entropy gives:
            dZ_last = (softmax(logits) - y) / N
        """
        N = cache["A"][0].shape[0]
        logits = cache["Z"][-1]
        probs = softmax(logits)
        dZ = (probs - y_onehot) / N

        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        bp_layers = []

        # last layer
        A_prev = cache["A"][-1]
        dW[-1] = A_prev.T @ dZ
        db[-1] = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.W[-1].T

        bp_layers.append({
            "layer": len(self.W) - 1,
            "W_shape": self.W[-1].shape,
            "b_shape": self.b[-1].shape,
            "Z_shape": cache["Z"][-1].shape,
            "A_prev_shape": A_prev.shape,
            "A_shape": "(logits only)",
            "dZ_norm": float(np.linalg.norm(dZ)),
            "dW_norm": float(np.linalg.norm(dW[-1])),
            "db_norm": float(np.linalg.norm(db[-1]))
        })

        # hidden layers reverse
        for layer in reversed(range(len(self.W) - 1)):
            Z = cache["Z"][layer]
            dZ = dA_prev * self.act_df(Z)

            A_prev = cache["A"][layer]
            A_cur = cache["A"][layer + 1]

            dW[layer] = A_prev.T @ dZ
            db[layer] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA_prev = dZ @ self.W[layer].T

            bp_layers.append({
                "layer": layer,
                "W_shape": self.W[layer].shape,
                "b_shape": self.b[layer].shape,
                "Z_shape": Z.shape,
                "A_prev_shape": A_prev.shape,
                "A_shape": A_cur.shape,
                "dZ_norm": float(np.linalg.norm(dZ)),
                "dW_norm": float(np.linalg.norm(dW[layer])),
                "db_norm": float(np.linalg.norm(db[layer]))
            })

        bp_layers.reverse()
        bp_info = {
            "activation": self.act_name,
            "notes": ACTIVATIONS[self.act_name]["notes"],
            "layers": bp_layers
        }
        return dW, db, bp_info

    def step(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict_logits(self, X):
        logits, _ = self.forward(X)
        return logits

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)


# ============================================================
# 4) GUI App
# ============================================================

class NeuralNetGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NN From Scratch (NumPy) - Activations + Visual Backprop + Step Mode")
        self.geometry("1280x820")

        # data
        self.X = None
        self.y = None
        self.num_classes = None

        # split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_oh = None
        self.y_test_oh = None

        # model
        self.model = None

        # history
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

        # latest backprop info
        self.latest_bp_info = None

        # ---- Step-mode state (single-batch pipeline) ----
        self.step_batch_X = None
        self.step_batch_y_int = None
        self.step_batch_y_oh = None

        self.step_logits = None
        self.step_probs = None
        self.step_loss = None
        self.step_cache = None

        self.step_dW = None
        self.step_db = None
        self.step_bp_info = None

        self.step_update_count = 0

        self._build_ui()

    # ---------------- UI building ----------------

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # LEFT: controls
        ttk.Label(left, text="Dataset", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Button(left, text="Generate XOR (2D)", command=self.generate_xor).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Load CSV (last col = label)", command=self.load_csv).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Network", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Hidden layers (e.g., 8,8):").pack(anchor="w")
        self.hidden_entry = ttk.Entry(left)
        self.hidden_entry.insert(0, "8,8")
        self.hidden_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Hidden activation:").pack(anchor="w")
        self.act_var = tk.StringVar(value="ReLU")
        self.act_combo = ttk.Combobox(left, textvariable=self.act_var, state="readonly",
                                      values=list(ACTIVATIONS.keys()))
        self.act_combo.pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Explain activation", command=self.explain_activation).pack(fill=tk.X, pady=(2, 6))

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Training (Normal Mode)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Learning rate:").pack(anchor="w")
        self.lr_entry = ttk.Entry(left)
        self.lr_entry.insert(0, "0.05")
        self.lr_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Epochs:").pack(anchor="w")
        self.epochs_entry = ttk.Entry(left)
        self.epochs_entry.insert(0, "300")
        self.epochs_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Batch size:").pack(anchor="w")
        self.batch_entry = ttk.Entry(left)
        self.batch_entry.insert(0, "32")
        self.batch_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Test split ratio:").pack(anchor="w")
        self.test_ratio_entry = ttk.Entry(left)
        self.test_ratio_entry.insert(0, "0.25")
        self.test_ratio_entry.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Backprop panel update (epochs):").pack(anchor="w")
        self.bp_every_entry = ttk.Entry(left)
        self.bp_every_entry.insert(0, "10")
        self.bp_every_entry.pack(fill=tk.X, pady=2)

        self.show_bp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Show Visual Backprop updates", variable=self.show_bp_var).pack(anchor="w", pady=(2, 8))

        ttk.Button(left, text="Build Model", command=self.build_model).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Train (Normal)", command=self.train_normal).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Evaluate", command=self.evaluate).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Logs", font=("Arial", 12, "bold")).pack(anchor="w")
        self.log_text = tk.Text(left, height=18, width=46)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # RIGHT: tabs
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_plots = ttk.Frame(self.nb, padding=10)
        self.tab_bp = ttk.Frame(self.nb, padding=10)
        self.tab_step = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_plots, text="Plots")
        self.nb.add(self.tab_bp, text="Visual Backprop")
        self.nb.add(self.tab_step, text="Step Mode (Single Batch)")

        self._build_plots_tab()
        self._build_backprop_tab()
        self._build_step_tab()

        self._refresh_plots()
        self._refresh_step_buttons()

    def _build_plots_tab(self):
        ttk.Label(self.tab_plots, text="Training Loss + 2D Visualization", font=("Arial", 12, "bold")).pack(anchor="w")
        self.fig = Figure(figsize=(8.2, 6.2), dpi=100)
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_vis = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plots)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_backprop_tab(self):
        top = ttk.Frame(self.tab_bp)
        top.pack(fill=tk.X)

        ttk.Label(top, text="What is Backprop doing?", font=("Arial", 12, "bold")).pack(anchor="w")

        expl = (
            "Forward: Z = A_prev·W + b, A = activation(Z)\n"
            "Output: logits -> softmax -> probabilities\n"
            "Loss: cross-entropy(probs, y)\n\n"
            "Backward (chain rule):\n"
            "  dW = A_prevᵀ·dZ\n"
            "  db = sum(dZ)\n"
            "  dA_prev = dZ·Wᵀ\n"
            "  dZ_prev = dA_prev * activation'(Z_prev)\n\n"
            "This panel shows SHAPES and gradient norms per layer.\n"
            "Tiny norms -> vanishing gradients. Huge norms -> exploding gradients."
        )
        ttk.Label(top, text=expl, justify="left").pack(anchor="w", pady=(4, 8))

        self.bp_act_notes = ttk.Label(top, text="Activation notes appear after Build/Train/Step Backprop.", justify="left")
        self.bp_act_notes.pack(anchor="w", pady=(0, 8))

        cols = ("Layer", "W shape", "b shape", "Z shape", "A_prev shape", "A shape", "||dZ||", "||dW||", "||db||")
        self.bp_tree = ttk.Treeview(self.tab_bp, columns=cols, show="headings", height=14)
        for c in cols:
            self.bp_tree.heading(c, text=c)
            self.bp_tree.column(c, width=130, anchor="center")
        self.bp_tree.column("Layer", width=70)
        self.bp_tree.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self.tab_bp, text="Deeper explanation popup", command=self.backprop_help_popup).pack(anchor="e", pady=(8, 0))

    def _build_step_tab(self):
        """
        Step Mode layout:
        - Top explanation
        - Controls row: Sample Batch, Forward, Backprop, Update Params, Reset
        - Status area with current step stats
        - Optional: show first few predictions for batch
        """
        ttk.Label(self.tab_step, text="Step Mode: One Mini-batch at a Time", font=("Arial", 12, "bold")).pack(anchor="w")

        expl = (
            "Use this to teach what training REALLY does.\n"
            "Typical training loop is:\n"
            "  (1) sample batch  -> (2) forward -> (3) loss -> (4) backprop -> (5) update params\n\n"
            "Buttons below let you run each part separately.\n"
            "Tip: After 'Backprop Step', switch to 'Visual Backprop' tab to see shapes + gradient sizes."
        )
        ttk.Label(self.tab_step, text=expl, justify="left").pack(anchor="w", pady=(6, 12))

        ctrl = ttk.Frame(self.tab_step)
        ctrl.pack(fill=tk.X, pady=(0, 10))

        self.step_shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Shuffle when sampling batch", variable=self.step_shuffle_var).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(ctrl, text="Step batch size (uses Batch size field on left):").pack(side=tk.LEFT)

        btns = ttk.Frame(self.tab_step)
        btns.pack(fill=tk.X, pady=(0, 10))

        self.btn_sample = ttk.Button(btns, text="Sample New Batch", command=self.step_sample_batch)
        self.btn_forward = ttk.Button(btns, text="Forward Step", command=self.step_forward)
        self.btn_backprop = ttk.Button(btns, text="Backprop Step", command=self.step_backprop)
        self.btn_update = ttk.Button(btns, text="Update Params", command=self.step_update_params)
        self.btn_reset = ttk.Button(btns, text="Reset Step State", command=self.step_reset_state)

        self.btn_sample.pack(side=tk.LEFT, padx=4)
        self.btn_forward.pack(side=tk.LEFT, padx=4)
        self.btn_backprop.pack(side=tk.LEFT, padx=4)
        self.btn_update.pack(side=tk.LEFT, padx=4)
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        # Step status / info panel
        status = ttk.LabelFrame(self.tab_step, text="Current Step Status", padding=10)
        status.pack(fill=tk.X, pady=(0, 10))

        self.step_status_var = tk.StringVar(value="No step batch sampled yet.")
        ttk.Label(status, textvariable=self.step_status_var, justify="left").pack(anchor="w")

        # A small text box to show quick batch predictions (helps learning softmax)
        pred_frame = ttk.LabelFrame(self.tab_step, text="Batch Snapshot (first 10 rows)", padding=10)
        pred_frame.pack(fill=tk.BOTH, expand=True)

        self.step_pred_text = tk.Text(pred_frame, height=14)
        self.step_pred_text.pack(fill=tk.BOTH, expand=True)

    # ---------------- small helpers ----------------

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _parse_hidden_layers(self):
        txt = self.hidden_entry.get().strip()
        if txt == "":
            return []
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError("Hidden layers must be integers separated by commas (e.g., 8,8).")

    def _get_lr(self):
        lr = float(self.lr_entry.get().strip())
        if not (0 < lr <= 10):
            raise ValueError("Learning rate should be > 0 and not huge (try 0.001 to 0.5).")
        return lr

    def _ensure_split(self):
        """
        Step mode needs a train split.
        If user hasn't trained yet, we still create a train/test split.
        """
        if self.X is None or self.y is None:
            raise ValueError("No dataset loaded. Generate XOR or load CSV first.")
        if self.X_train is None:
            test_ratio = float(self.test_ratio_entry.get().strip())
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_ratio=test_ratio, seed=0
            )
            self.y_train_oh = one_hot(self.y_train, self.num_classes)
            self.y_test_oh = one_hot(self.y_test, self.num_classes)

    def _update_backprop_panel(self, bp_info):
        self.latest_bp_info = bp_info
        self.bp_act_notes.configure(text=f"Hidden activation: {bp_info['activation']}\nNotes: {bp_info['notes']}")

        for row in self.bp_tree.get_children():
            self.bp_tree.delete(row)

        for layer_info in bp_info["layers"]:
            self.bp_tree.insert("", "end", values=(
                layer_info["layer"],
                str(layer_info["W_shape"]),
                str(layer_info["b_shape"]),
                str(layer_info["Z_shape"]),
                str(layer_info["A_prev_shape"]),
                str(layer_info["A_shape"]),
                f"{layer_info['dZ_norm']:.4f}",
                f"{layer_info['dW_norm']:.4f}",
                f"{layer_info['db_norm']:.4f}",
            ))

    def explain_activation(self):
        name = self.act_var.get()
        messagebox.showinfo(f"{name} activation", ACTIVATIONS[name]["notes"])

    def backprop_help_popup(self):
        win = tk.Toplevel(self)
        win.title("Backprop Explanation (Student-Friendly)")
        win.geometry("850x560")
        text = tk.Text(win, wrap="word")
        text.pack(fill=tk.BOTH, expand=True)

        long_expl = """
Backpropagation: the chain rule, made systematic
------------------------------------------------

Forward pass:
  For each hidden layer l:
    Z[l] = A[l] @ W[l] + b[l]
    A[l+1] = activation(Z[l])

  Output layer:
    logits = A[last] @ W[last] + b[last]
    probs  = softmax(logits)

Loss:
  L = cross_entropy(probs, y)

Backward pass:
  The key "signal" is dZ[l] = dL/dZ[l] (gradient of loss wrt pre-activation).

  For the output layer with softmax + cross-entropy:
    dZ_last = (probs - y) / N

  Then for each layer:
    dW = A_prev^T @ dZ
    db = sum(dZ)
    dA_prev = dZ @ W^T
    dZ_prev = dA_prev * activation'(Z_prev)

Update:
  W = W - lr * dW
  b = b - lr * db

Why show gradient norms (||dW|| etc.)?
- Tiny norms -> vanishing gradients -> slow learning.
- Huge norms -> exploding gradients -> unstable learning.

Step Mode:
- Sample batch
- Forward step
- Backprop step (fills Visual Backprop panel)
- Update params
Repeat to see how loss and predictions change.
"""
        text.insert("1.0", long_expl)
        text.configure(state="disabled")

    # ---------------- plots ----------------

    def _refresh_plots(self):
        self.ax_loss.clear()
        self.ax_vis.clear()

        self.ax_loss.set_title("Training Loss (logged points)")
        self.ax_loss.set_xlabel("Log index")
        self.ax_loss.set_ylabel("Cross-Entropy")
        if self.loss_history:
            self.ax_loss.plot(self.loss_history)

        self.ax_vis.set_title("2D Data / Decision Boundary (2 features only)")
        self.ax_vis.set_xlabel("x1")
        self.ax_vis.set_ylabel("x2")

        if self.X is not None and self.X.shape[1] == 2 and self.y is not None:
            for c in range(self.num_classes):
                mask = (self.y == c)
                self.ax_vis.scatter(self.X[mask, 0], self.X[mask, 1], label=f"class {c}", s=18)

            if self.model is not None:
                self._plot_decision_boundary()

            self.ax_vis.legend(loc="best", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_decision_boundary(self):
        X = self.X
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        steps = 230
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                             np.linspace(y_min, y_max, steps))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.model.predict(grid).reshape(xx.shape)
        self.ax_vis.contourf(xx, yy, preds, alpha=0.22)

    # ---------------- dataset ----------------

    def generate_xor(self):
        rng = np.random.default_rng(0)
        N = 500
        X = rng.uniform(-1, 1, size=(N, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

        self.X = X
        self.y = y
        self.num_classes = int(np.max(y)) + 1

        self._reset_everything_model_related()
        self.log(f"Generated XOR dataset: X={self.X.shape}, classes={self.num_classes}")
        self._refresh_plots()

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            data = np.genfromtxt(path, delimiter=",", dtype=float)
            if data.ndim == 1:
                raise ValueError("CSV must have multiple rows.")

            X = data[:, :-1]
            y_raw = data[:, -1].astype(int)

            if X.shape[1] < 1:
                raise ValueError("Need at least 1 feature column.")

            unique = np.unique(y_raw)
            remap = {val: i for i, val in enumerate(unique)}
            y = np.array([remap[v] for v in y_raw], dtype=int)

            self.X = X
            self.y = y
            self.num_classes = len(unique)

            self._reset_everything_model_related()
            self.log(f"Loaded CSV: {path}")
            self.log(f"X={self.X.shape}, classes={self.num_classes}")
            self._refresh_plots()

        except Exception as e:
            messagebox.showerror("CSV Load Error", str(e))

    def _reset_everything_model_related(self):
        # model and splits
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_train_oh = self.y_test_oh = None

        # history
        self.loss_history.clear()
        self.train_acc_history.clear()
        self.test_acc_history.clear()

        # backprop info
        self.latest_bp_info = None
        self.bp_act_notes.configure(text="Activation notes appear after Build/Train/Step Backprop.")

        # step mode state
        self.step_reset_state(log_it=False)

    # ---------------- model + normal training ----------------

    def build_model(self):
        if self.X is None:
            messagebox.showwarning("No data", "Generate XOR or load a CSV first.")
            return
        try:
            hidden = self._parse_hidden_layers()
            act = self.act_var.get()

            self.model = MLP(
                input_dim=self.X.shape[1],
                hidden_layers=hidden,
                output_dim=self.num_classes,
                activation_name=act,
                seed=0
            )

            # reset splits and histories (new model = new training run)
            self.X_train = self.X_test = None
            self.y_train = self.y_test = None
            self.y_train_oh = self.y_test_oh = None

            self.loss_history.clear()
            self.train_acc_history.clear()
            self.test_acc_history.clear()

            # reset step state too (new parameters)
            self.step_reset_state(log_it=False)

            self.log(f"Built model: input={self.X.shape[1]}, hidden={hidden}, output={self.num_classes}, act={act}")
            self.bp_act_notes.configure(text=f"Hidden activation: {act}\nNotes: {ACTIVATIONS[act]['notes']}")
            self._refresh_plots()
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Build Model Error", str(e))

    def train_normal(self):
        if self.model is None:
            messagebox.showwarning("No model", "Click 'Build Model' first.")
            return
        if self.X is None or self.y is None:
            messagebox.showwarning("No data", "Load/generate data first.")
            return

        try:
            lr = self._get_lr()
            epochs = int(self.epochs_entry.get().strip())
            batch = int(self.batch_entry.get().strip())
            test_ratio = float(self.test_ratio_entry.get().strip())
            bp_every = int(self.bp_every_entry.get().strip())
            if bp_every < 1:
                bp_every = 1

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_ratio=test_ratio, seed=0
            )
            self.y_train_oh = one_hot(self.y_train, self.num_classes)
            self.y_test_oh = one_hot(self.y_test, self.num_classes)

            self.log(f"Train={len(self.X_train)} samples, Test={len(self.X_test)} samples")
            self.log("Training (Normal Mode)...")

            rng = np.random.default_rng(0)
            N = len(self.X_train)
            log_every = max(1, epochs // 50)

            for epoch in range(1, epochs + 1):
                idx = np.arange(N)
                rng.shuffle(idx)
                Xs = self.X_train[idx]
                Ys = self.y_train_oh[idx]

                for start in range(0, N, batch):
                    end = start + batch
                    Xb = Xs[start:end]
                    Yb = Ys[start:end]

                    logits, cache = self.model.forward(Xb)
                    dW, db, bp_info = self.model.backward(Yb, cache)
                    self.model.step(dW, db, lr)

                    if self.show_bp_var.get() and (epoch % bp_every == 0) and (start == 0):
                        self._update_backprop_panel(bp_info)
                        self.update_idletasks()

                if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                    train_logits = self.model.predict_logits(self.X_train)
                    test_logits = self.model.predict_logits(self.X_test)

                    train_loss = cross_entropy(softmax(train_logits), self.y_train_oh)
                    train_acc = accuracy_from_logits(train_logits, self.y_train)
                    test_acc = accuracy_from_logits(test_logits, self.y_test)

                    self.loss_history.append(train_loss)
                    self.train_acc_history.append(train_acc)
                    self.test_acc_history.append(test_acc)

                    self.log(f"Epoch {epoch:>5}/{epochs} | loss={train_loss:.4f} | "
                             f"train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")

                    self._refresh_plots()
                    self.update_idletasks()

            self.log("Done training.")
            self._refresh_plots()

            # Step mode still works after training; it will use the current parameters.
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def evaluate(self):
        if self.model is None:
            messagebox.showwarning("No model", "Build a model first.")
            return
        if self.X is None or self.y is None:
            messagebox.showwarning("No data", "Load/generate data first.")
            return

        try:
            self._ensure_split()
            train_acc = float(np.mean(self.model.predict(self.X_train) == self.y_train))
            test_acc = float(np.mean(self.model.predict(self.X_test) == self.y_test))
            self.log(f"Evaluation: train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")
            self._refresh_plots()
        except Exception as e:
            messagebox.showerror("Evaluate Error", str(e))

    # ============================================================
    # 5) Step Mode (Single Batch) methods
    # ============================================================

    def _refresh_step_buttons(self):
        """
        Enable/disable step buttons based on what is currently available.
        """
        has_model = self.model is not None
        has_batch = self.step_batch_X is not None
        has_forward = self.step_cache is not None and self.step_logits is not None
        has_grads = self.step_dW is not None and self.step_db is not None

        self.btn_sample.configure(state=("normal" if has_model else "disabled"))
        self.btn_forward.configure(state=("normal" if (has_model and has_batch) else "disabled"))
        self.btn_backprop.configure(state=("normal" if (has_model and has_forward) else "disabled"))
        self.btn_update.configure(state=("normal" if (has_model and has_grads) else "disabled"))
        self.btn_reset.configure(state=("normal" if has_model else "disabled"))

    def step_reset_state(self, log_it=True):
        """
        Reset only the step-mode pipeline (does NOT reset the whole model).
        """
        self.step_batch_X = None
        self.step_batch_y_int = None
        self.step_batch_y_oh = None

        self.step_logits = None
        self.step_probs = None
        self.step_loss = None
        self.step_cache = None

        self.step_dW = None
        self.step_db = None
        self.step_bp_info = None

        self.step_update_count = 0

        self.step_status_var.set("No step batch sampled yet.")
        self.step_pred_text.delete("1.0", tk.END)
        if log_it:
            self.log("Step Mode: reset state.")
        self._refresh_step_buttons()

    def step_sample_batch(self):
        """
        Choose a mini-batch from training data.
        If no train/test split exists yet, create it.
        """
        if self.model is None:
            messagebox.showwarning("No model", "Build a model first.")
            return

        try:
            self._ensure_split()
            batch_size = int(self.batch_entry.get().strip())
            if not (1 <= batch_size <= len(self.X_train)):
                raise ValueError(f"Batch size must be between 1 and {len(self.X_train)}.")

            rng = np.random.default_rng(0 if self.step_shuffle_var.get() else 123)
            idx = np.arange(len(self.X_train))
            if self.step_shuffle_var.get():
                rng.shuffle(idx)
            idx = idx[:batch_size]

            self.step_batch_X = self.X_train[idx]
            self.step_batch_y_int = self.y_train[idx]
            self.step_batch_y_oh = self.y_train_oh[idx]

            # Clear downstream step outputs (new batch means redo forward/backward)
            self.step_logits = None
            self.step_probs = None
            self.step_loss = None
            self.step_cache = None
            self.step_dW = None
            self.step_db = None
            self.step_bp_info = None
            self.step_update_count = 0

            self.log(f"Step Mode: sampled new batch of size {batch_size}.")
            self._update_step_snapshot("Batch sampled. Next: Forward Step.")
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Step Sample Error", str(e))

    def step_forward(self):
        """
        Forward pass on the current step batch.
        Stores logits/probs/loss/cache for the Backprop Step.
        """
        if self.model is None or self.step_batch_X is None:
            messagebox.showwarning("Missing", "Need a model and a sampled batch first.")
            return

        try:
            logits, cache = self.model.forward(self.step_batch_X)
            probs = softmax(logits)
            loss = cross_entropy(probs, self.step_batch_y_oh)
            acc = accuracy_from_logits(logits, self.step_batch_y_int)

            self.step_logits = logits
            self.step_probs = probs
            self.step_loss = float(loss)
            self.step_cache = cache

            # Clear grads (must run backprop again after a new forward)
            self.step_dW = None
            self.step_db = None
            self.step_bp_info = None

            self.log(f"Step Mode: forward done | batch loss={loss:.4f} | batch acc={acc*100:.1f}%")
            self._update_step_snapshot("Forward done. Next: Backprop Step.", extra_loss=loss, extra_acc=acc)
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Step Forward Error", str(e))

    def step_backprop(self):
        """
        Backprop on the current step batch using cached forward results.
        Updates Visual Backprop panel.
        """
        if self.model is None or self.step_cache is None or self.step_batch_y_oh is None:
            messagebox.showwarning("Missing", "Run Forward Step first.")
            return

        try:
            dW, db, bp_info = self.model.backward(self.step_batch_y_oh, self.step_cache)
            self.step_dW = dW
            self.step_db = db
            self.step_bp_info = bp_info

            self._update_backprop_panel(bp_info)

            # also compute batch acc to show
            acc = accuracy_from_logits(self.step_logits, self.step_batch_y_int)
            self.log("Step Mode: backprop done (see Visual Backprop tab for shapes + gradient norms).")
            self._update_step_snapshot("Backprop done. Next: Update Params.", extra_loss=self.step_loss, extra_acc=acc)
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Step Backprop Error", str(e))

    def step_update_params(self):
        """
        Apply one gradient descent update using the stored gradients.
        Then clear forward/backprop cache because params changed.
        """
        if self.model is None or self.step_dW is None or self.step_db is None:
            messagebox.showwarning("Missing", "Run Backprop Step first.")
            return

        try:
            lr = self._get_lr()
            self.model.step(self.step_dW, self.step_db, lr)
            self.step_update_count += 1

            # After update, cached forward/backward results are "stale" (old parameters)
            self.step_logits = None
            self.step_probs = None
            self.step_loss = None
            self.step_cache = None
            self.step_dW = None
            self.step_db = None
            self.step_bp_info = None

            self.log(f"Step Mode: updated parameters with lr={lr}. Updates so far: {self.step_update_count}")
            self._update_step_snapshot("Params updated. Next: Forward Step (same batch) or Sample New Batch.")
            self._refresh_plots()          # decision boundary can change
            self._refresh_step_buttons()

        except Exception as e:
            messagebox.showerror("Step Update Error", str(e))

    def _update_step_snapshot(self, headline, extra_loss=None, extra_acc=None):
        """
        Write a readable snapshot of the batch and predictions to the step tab.
        """
        if self.step_batch_X is None:
            self.step_status_var.set("No step batch sampled yet.")
            return

        bs = len(self.step_batch_X)
        msg = f"{headline}\n\nBatch size: {bs}\nParam updates applied in Step Mode: {self.step_update_count}\n"
        if extra_loss is not None:
            msg += f"Current batch loss (most recent forward): {float(extra_loss):.6f}\n"
        if extra_acc is not None:
            msg += f"Current batch accuracy (most recent forward): {float(extra_acc)*100:.2f}%\n"

        self.step_status_var.set(msg)

        # Print first few rows of batch with probs/preds if available
        self.step_pred_text.delete("1.0", tk.END)

        max_rows = min(10, bs)
        self.step_pred_text.insert(tk.END, "Row | x[...] | y_true | pred | probs\n")
        self.step_pred_text.insert(tk.END, "-" * 90 + "\n")

        if self.step_logits is not None and self.step_probs is not None:
            preds = np.argmax(self.step_logits, axis=1)
            for i in range(max_rows):
                x = self.step_batch_X[i]
                y_true = int(self.step_batch_y_int[i])
                pred = int(preds[i])
                probs = self.step_probs[i]
                probs_str = "[" + ", ".join(f"{p:.3f}" for p in probs) + "]"
                self.step_pred_text.insert(tk.END, f"{i:>3} | {np.array2string(x, precision=3)} | {y_true} | {pred} | {probs_str}\n")
        else:
            for i in range(max_rows):
                x = self.step_batch_X[i]
                y_true = int(self.step_batch_y_int[i])
                self.step_pred_text.insert(tk.END, f"{i:>3} | {np.array2string(x, precision=3)} | {y_true} | (run Forward Step) | ...\n")

    # ---------------- manual predict (left panel feature) ----------------

    def manual_predict(self):
        if self.model is None:
            messagebox.showwarning("No model", "Build and train a model first.")
            return

        try:
            raw = self.manual_entry.get().strip()
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            x = np.array([float(p) for p in parts], dtype=float).reshape(1, -1)

            if self.X is not None and x.shape[1] != self.X.shape[1]:
                raise ValueError(f"Expected {self.X.shape[1]} features, got {x.shape[1]}.")

            logits = self.model.predict_logits(x)
            probs = softmax(logits)[0]
            pred = int(np.argmax(probs))
            prob_str = ", ".join([f"class {i}: {probs[i]:.3f}" for i in range(len(probs))])
            self.log(f"Manual input -> predicted class = {pred} | probs: {prob_str}")

        except Exception as e:
            messagebox.showerror("Predict Error", str(e))


# ============================================================
# 6) Main
# ============================================================

if __name__ == "__main__":
    app = NeuralNetGUI()
    app.mainloop()
