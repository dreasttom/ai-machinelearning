"""
Neural Network From Scratch (NO TensorFlow / PyTorch)
====================================================

This single-file Python script builds a tiny “MLP” (Multi-Layer Perceptron)
using only NumPy for math, and Tkinter for a student-friendly GUI.

What it can do (in the GUI):
- Generate a classic XOR dataset (great for learning non-linear separation)
- (Optional) Load a simple CSV dataset (last column is the label)
- Choose network size (hidden layers), learning rate, epochs, batch size
- Train the network and watch loss + accuracy change
- Plot training loss
- If the data is 2D, plot the decision boundary

Requirements:
- Python 3.9+
- numpy
- matplotlib (commonly installed in many Python environments)

Install if needed:
    pip install numpy matplotlib

Run:
    python nn_gui_from_scratch.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Matplotlib for plots, embedded inside Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ============================================================
# 1) Utility Functions (data handling, activations, metrics)
# ============================================================

def one_hot(y_int, num_classes):
    """
    Convert integer labels (e.g., [0,1,2]) into one-hot vectors.
    Example: y=[2], num_classes=4 -> [0,0,1,0]
    """
    y_int = y_int.astype(int).ravel()
    out = np.zeros((y_int.size, num_classes), dtype=float)
    out[np.arange(y_int.size), y_int] = 1.0
    return out

def train_test_split(X, Y, test_ratio=0.25, seed=0):
    """
    Shuffle and split into train/test sets.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

def accuracy_from_logits(logits, y_true_int):
    """
    logits: (N, C)
    y_true_int: (N,) integer class labels
    """
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true_int))


# ---- Activation functions and their derivatives ----

def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    # derivative is 1 where z > 0 else 0
    return (z > 0).astype(float)

def sigmoid(z):
    # numerically stable sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(a):
    # If a = sigmoid(z), derivative wrt z is a*(1-a)
    return a * (1.0 - a)

def softmax(z):
    """
    Stable softmax:
    softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(probs, y_onehot):
    """
    Cross-entropy loss for multi-class classification.
    probs: (N, C) after softmax
    y_onehot: (N, C)
    """
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


# ============================================================
# 2) MLP Neural Network (NumPy only)
# ============================================================

class MLP:
    """
    A simple fully-connected neural network for classification:

        X -> Dense -> ReLU -> Dense -> ReLU -> ... -> Dense -> Softmax

    Notes for students:
    - “Dense” layer means y = XW + b (matrix multiply + bias)
    - ReLU adds non-linearity, enabling learning of complex patterns
    - Softmax converts final scores into class probabilities
    - We train by gradient descent using backpropagation
    """

    def __init__(self, input_dim, hidden_layers, output_dim, seed=0):
        """
        input_dim: number of input features
        hidden_layers: list like [8, 8] meaning two hidden layers with 8 units each
        output_dim: number of classes
        """
        self.rng = np.random.default_rng(seed)

        # Build the layer sizes list, e.g. [input_dim, 8, 8, output_dim]
        layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]

        # Parameters: for each layer l, have weight W[l] and bias b[l]
        self.W = []
        self.b = []

        # He initialization-ish for ReLU layers (good default)
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # Weight matrix: shape (fan_in, fan_out)
            # Small random values help break symmetry.
            w = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            bias = np.zeros((1, fan_out), dtype=float)

            self.W.append(w)
            self.b.append(bias)

    def forward(self, X):
        """
        Forward pass: compute outputs layer-by-layer.

        Returns:
            logits: (N, C) raw scores before softmax
            cache: values needed for backprop (activations + pre-activations)
        """
        A = X  # activation of "current layer"
        cache = {
            "A": [X],   # store activations (A0 = X)
            "Z": []     # store pre-activations (Z = XW + b)
        }

        # For all layers except the last: Dense + ReLU
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]  # pre-activation
            A = relu(Z)                    # activation
            cache["Z"].append(Z)
            cache["A"].append(A)

        # Last layer: Dense only (no ReLU)
        Z_last = A @ self.W[-1] + self.b[-1]
        cache["Z"].append(Z_last)
        # We won't store softmax output as an "activation" in the cache,
        # because for backprop we typically use logits + softmax derivative trick.
        return Z_last, cache

    def backward(self, X, y_onehot, cache):
        """
        Backpropagation: compute gradients for all weights and biases.

        We'll use the classic simplification:
            softmax + cross-entropy gradient wrt logits is (probs - y) / N

        Returns:
            dW: list of gradients for weights
            db: list of gradients for biases
        """
        N = X.shape[0]
        logits = cache["Z"][-1]
        probs = softmax(logits)

        # Gradient at output (logits)
        dZ = (probs - y_onehot) / N   # shape (N, C)

        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # Last layer gradients:
        A_prev = cache["A"][-1]        # activation from previous layer
        dW[-1] = A_prev.T @ dZ         # (H, N) @ (N, C) -> (H, C)
        db[-1] = np.sum(dZ, axis=0, keepdims=True)

        # Propagate backward through hidden layers:
        dA_prev = dZ @ self.W[-1].T    # (N, C) @ (C, H) -> (N, H)

        # Iterate hidden layers in reverse order
        for layer in reversed(range(len(self.W) - 1)):
            Z = cache["Z"][layer]      # pre-activation for this layer
            dZ = dA_prev * relu_grad(Z)

            A_prev = cache["A"][layer]  # activation from layer before this one
            dW[layer] = A_prev.T @ dZ
            db[layer] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA_prev = dZ @ self.W[layer].T

        return dW, db

    def step(self, dW, db, lr):
        """
        Gradient descent update:
            W = W - lr * dW
            b = b - lr * db
        """
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict_logits(self, X):
        logits, _ = self.forward(X)
        return logits

    def predict(self, X):
        logits = self.predict_logits(X)
        return np.argmax(logits, axis=1)


# ============================================================
# 3) Tkinter GUI Application
# ============================================================

class NeuralNetGUI(tk.Tk):
    """
    A GUI wrapper so students can interactively:
    - Build a network
    - Train it
    - See loss and accuracy
    - Visualize results
    """

    def __init__(self):
        super().__init__()
        self.title("Neural Network From Scratch (NumPy) - Student GUI")
        self.geometry("1100x700")

        # Data storage
        self.X = None
        self.y_int = None
        self.num_classes = None

        # Train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_oh = None
        self.y_test_oh = None

        # Model
        self.model = None

        # Training history
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

        self._build_ui()

    def _build_ui(self):
        # ---------- Layout frames ----------
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---------- Controls (LEFT) ----------
        ttk.Label(left, text="Dataset", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Button(left, text="Generate XOR (2D)", command=self.generate_xor).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Load CSV (last col = label)", command=self.load_csv).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Network & Training Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

        # Hidden layers entry: e.g. "8,8"
        self.hidden_entry = ttk.Entry(left)
        self.hidden_entry.insert(0, "8,8")
        ttk.Label(left, text="Hidden layers (comma-separated):").pack(anchor="w")
        self.hidden_entry.pack(fill=tk.X, pady=2)

        self.lr_entry = ttk.Entry(left)
        self.lr_entry.insert(0, "0.05")
        ttk.Label(left, text="Learning rate (lr):").pack(anchor="w")
        self.lr_entry.pack(fill=tk.X, pady=2)

        self.epochs_entry = ttk.Entry(left)
        self.epochs_entry.insert(0, "200")
        ttk.Label(left, text="Epochs:").pack(anchor="w")
        self.epochs_entry.pack(fill=tk.X, pady=2)

        self.batch_entry = ttk.Entry(left)
        self.batch_entry.insert(0, "32")
        ttk.Label(left, text="Batch size:").pack(anchor="w")
        self.batch_entry.pack(fill=tk.X, pady=2)

        self.test_ratio_entry = ttk.Entry(left)
        self.test_ratio_entry.insert(0, "0.25")
        ttk.Label(left, text="Test split ratio (0-0.9):").pack(anchor="w")
        self.test_ratio_entry.pack(fill=tk.X, pady=2)

        ttk.Button(left, text="Build Model", command=self.build_model).pack(fill=tk.X, pady=(10, 2))
        ttk.Button(left, text="Train", command=self.train).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Evaluate", command=self.evaluate).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Manual Predict (comma-separated features)", font=("Arial", 11, "bold")).pack(anchor="w")
        self.manual_entry = ttk.Entry(left)
        self.manual_entry.insert(0, "0,0")
        self.manual_entry.pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Predict", command=self.manual_predict).pack(fill=tk.X, pady=2)

        # Status box
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Status / Logs", font=("Arial", 12, "bold")).pack(anchor="w")
        self.log_text = tk.Text(left, height=18, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ---------- Plots (RIGHT) ----------
        ttk.Label(right, text="Plots", font=("Arial", 12, "bold")).pack(anchor="w")

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_loss = self.fig.add_subplot(211)    # top plot: loss
        self.ax_vis = self.fig.add_subplot(212)     # bottom plot: data/decision boundary

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._refresh_plots()

    # ---------------- GUI helper methods ----------------

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _parse_hidden_layers(self):
        """
        Parse something like "8,8,4" into [8,8,4].
        If empty, return [] (no hidden layers).
        """
        txt = self.hidden_entry.get().strip()
        if txt == "":
            return []
        parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError("Hidden layers must be integers separated by commas (e.g., 8,8).")

    def _refresh_plots(self):
        # Clear axes
        self.ax_loss.clear()
        self.ax_vis.clear()

        # Loss plot
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Cross-Entropy Loss")
        if self.loss_history:
            self.ax_loss.plot(self.loss_history)

        # Visualization plot (data + boundary)
        self.ax_vis.set_title("Data / Decision Boundary (2D only)")
        self.ax_vis.set_xlabel("x1")
        self.ax_vis.set_ylabel("x2")

        # Plot points if 2D data exists
        if self.X is not None and self.X.shape[1] == 2 and self.y_int is not None:
            for c in range(self.num_classes):
                mask = (self.y_int == c)
                self.ax_vis.scatter(self.X[mask, 0], self.X[mask, 1], label=f"class {c}", s=20)

            # Plot decision boundary if model exists
            if self.model is not None:
                self._plot_decision_boundary()

            self.ax_vis.legend(loc="best", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_decision_boundary(self):
        """
        For 2D data only: create a grid and color by predicted class.
        This is a standard visualization for classification.
        """
        X = self.X
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # Grid resolution: higher is smoother but slower
        steps = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                             np.linspace(y_min, y_max, steps))
        grid = np.c_[xx.ravel(), yy.ravel()]

        preds = self.model.predict(grid)
        zz = preds.reshape(xx.shape)

        # Use contourf to color regions
        self.ax_vis.contourf(xx, yy, zz, alpha=0.25)

    # ---------------- Dataset methods ----------------

    def generate_xor(self):
        """
        Create a classic XOR dataset:
        - Two inputs (x1, x2)
        - Two classes (0 or 1)
        XOR pattern is NOT linearly separable, so it demonstrates why neural nets help.
        """
        rng = np.random.default_rng(0)
        N = 400

        X = rng.uniform(-1, 1, size=(N, 2))
        # XOR: label is 1 if signs differ, else 0
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

        self.X = X
        self.y_int = y
        self.num_classes = int(np.max(y)) + 1

        self.model = None
        self.loss_history.clear()
        self.train_acc_history.clear()
        self.test_acc_history.clear()

        self.log("Generated XOR dataset: X shape = {}, classes = {}".format(self.X.shape, self.num_classes))
        self._refresh_plots()

    def load_csv(self):
        """
        Load a CSV where:
        - All columns except the last are features (X)
        - Last column is the label (integer classes)

        For learning simplicity, this expects numeric values and integer labels.
        """
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
            y = data[:, -1].astype(int)

            if X.shape[1] < 1:
                raise ValueError("Need at least 1 feature column.")

            # Ensure labels start at 0 (common convention for one-hot)
            unique = np.unique(y)
            # If labels are like [1,2] we remap to [0,1]
            remap = {val: i for i, val in enumerate(unique)}
            y_mapped = np.array([remap[val] for val in y], dtype=int)

            self.X = X
            self.y_int = y_mapped
            self.num_classes = len(unique)

            self.model = None
            self.loss_history.clear()
            self.train_acc_history.clear()
            self.test_acc_history.clear()

            self.log(f"Loaded CSV: {path}")
            self.log(f"X shape = {self.X.shape}, num_classes = {self.num_classes}")
            self._refresh_plots()

        except Exception as e:
            messagebox.showerror("CSV Load Error", str(e))

    # ---------------- Model / Training methods ----------------

    def build_model(self):
        """
        Build a new MLP based on the GUI settings and current dataset.
        """
        if self.X is None or self.y_int is None:
            messagebox.showwarning("No data", "Please generate XOR or load a CSV first.")
            return

        try:
            hidden = self._parse_hidden_layers()
            input_dim = self.X.shape[1]
            output_dim = self.num_classes

            self.model = MLP(input_dim=input_dim, hidden_layers=hidden, output_dim=output_dim, seed=0)

            self.loss_history.clear()
            self.train_acc_history.clear()
            self.test_acc_history.clear()

            self.log(f"Built model: input_dim={input_dim}, hidden={hidden}, output_dim={output_dim}")
            self._refresh_plots()
        except Exception as e:
            messagebox.showerror("Build Model Error", str(e))

    def train(self):
        """
        Train the model using mini-batch gradient descent.
        """
        if self.model is None:
            messagebox.showwarning("No model", "Click 'Build Model' first.")
            return
        if self.X is None or self.y_int is None:
            messagebox.showwarning("No data", "Please load/generate data first.")
            return

        try:
            lr = float(self.lr_entry.get().strip())
            epochs = int(self.epochs_entry.get().strip())
            batch_size = int(self.batch_entry.get().strip())
            test_ratio = float(self.test_ratio_entry.get().strip())

            if not (0.0 < lr <= 10.0):
                raise ValueError("Learning rate should be > 0 and not huge (try 0.001 to 0.5).")
            if not (1 <= epochs <= 50000):
                raise ValueError("Epochs should be between 1 and 50000 for this demo.")
            if not (1 <= batch_size <= len(self.X)):
                raise ValueError("Batch size must be between 1 and N.")
            if not (0.0 <= test_ratio <= 0.9):
                raise ValueError("Test ratio should be between 0.0 and 0.9.")

            # Split data
            Y_oh = one_hot(self.y_int, self.num_classes)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y_int, test_ratio=test_ratio, seed=0
            )
            self.y_train_oh = one_hot(self.y_train, self.num_classes)
            self.y_test_oh = one_hot(self.y_test, self.num_classes)

            self.log(f"Train split: {len(self.X_train)} samples, Test split: {len(self.X_test)} samples")
            self.log("Training...")

            rng = np.random.default_rng(0)
            N = len(self.X_train)

            for epoch in range(1, epochs + 1):
                # Shuffle training data each epoch
                idx = np.arange(N)
                rng.shuffle(idx)
                Xs = self.X_train[idx]
                Ys = self.y_train_oh[idx]

                # Mini-batches
                for start in range(0, N, batch_size):
                    end = start + batch_size
                    Xb = Xs[start:end]
                    Yb = Ys[start:end]

                    logits, cache = self.model.forward(Xb)
                    probs = softmax(logits)
                    loss = cross_entropy(probs, Yb)

                    dW, db = self.model.backward(Xb, Yb, cache)
                    self.model.step(dW, db, lr)

                # Track metrics occasionally (every few epochs for speed)
                if epoch == 1 or epoch % max(1, epochs // 50) == 0 or epoch == epochs:
                    train_logits = self.model.predict_logits(self.X_train)
                    test_logits = self.model.predict_logits(self.X_test)

                    train_probs = softmax(train_logits)
                    train_loss = cross_entropy(train_probs, self.y_train_oh)

                    train_acc = accuracy_from_logits(train_logits, self.y_train)
                    test_acc = accuracy_from_logits(test_logits, self.y_test)

                    self.loss_history.append(train_loss)
                    self.train_acc_history.append(train_acc)
                    self.test_acc_history.append(test_acc)

                    self.log(f"Epoch {epoch:>5}/{epochs} | loss={train_loss:.4f} | "
                             f"train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")

                    # Update plots and allow GUI to refresh
                    self._refresh_plots()
                    self.update_idletasks()

            self.log("Done training.")
            self._refresh_plots()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def evaluate(self):
        """
        Evaluate accuracy on current train/test split (or split now if not done).
        """
        if self.model is None:
            messagebox.showwarning("No model", "Click 'Build Model' first.")
            return
        if self.X is None or self.y_int is None:
            messagebox.showwarning("No data", "Please load/generate data first.")
            return

        # If no split has been created yet (e.g., user never trained), create it quickly
        if self.X_train is None:
            try:
                test_ratio = float(self.test_ratio_entry.get().strip())
            except:
                test_ratio = 0.25
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y_int, test_ratio=test_ratio, seed=0
            )

        train_acc = float(np.mean(self.model.predict(self.X_train) == self.y_train))
        test_acc = float(np.mean(self.model.predict(self.X_test) == self.y_test))

        self.log(f"Evaluation: train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")
        self._refresh_plots()

    def manual_predict(self):
        """
        Let the student type a single input sample into the GUI,
        like: "0.2, -0.5, 1.0"
        and see the predicted class.
        """
        if self.model is None:
            messagebox.showwarning("No model", "Build and train a model first.")
            return

        try:
            raw = self.manual_entry.get().strip()
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            x = np.array([float(p) for p in parts], dtype=float).reshape(1, -1)

            if self.X is not None and x.shape[1] != self.X.shape[1]:
                raise ValueError(f"Expected {self.X.shape[1]} features, got {x.shape[1]}.")

            logits = self.model.predict_logits(x)
            probs = softmax(logits)[0]
            pred = int(np.argmax(probs))

            # Show all class probabilities (helpful for understanding softmax)
            prob_str = ", ".join([f"class {i}: {probs[i]:.3f}" for i in range(len(probs))])
            self.log(f"Manual input -> predicted class = {pred} | probs: {prob_str}")

        except Exception as e:
            messagebox.showerror("Predict Error", str(e))


# ============================================================
# 4) Main entry point
# ============================================================

if __name__ == "__main__":
    app = NeuralNetGUI()
    app.mainloop()
