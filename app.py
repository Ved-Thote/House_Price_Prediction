"""
=============================================================================
app.py — House Price Prediction System  (GUI Application)
=============================================================================
Purpose : Tkinter-based desktop GUI that loads the pre-trained model and
          lets the user predict house prices by filling out a simple form.

Requires : trained_model.pkl to exist in the models/ directory.
           Run train_model.py first if it does not exist.

Author  : College ML Project
Python  : 3.8+
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import pickle
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np

# ── Local Modules ─────────────────────────────────────────────────────────────
# train_model.py must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import ModelSerializer, run_training_pipeline

# =============================================================================
# PATH CONSTANTS
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")


# =============================================================================
# COLOUR PALETTE & FONTS  (easy to theme later)
# =============================================================================

COLOURS = {
    "bg"          : "#F0F4F8",   # window background
    "header_bg"   : "#1E3A5F",   # dark-blue header
    "header_fg"   : "#FFFFFF",
    "frame_bg"    : "#FFFFFF",   # card/panel background
    "accent"      : "#2E86AB",   # buttons & highlights
    "accent_hover": "#1B6CA8",
    "label_fg"    : "#2D3436",
    "subtle_fg"   : "#636E72",
    "result_bg"   : "#EBF5FB",
    "result_fg"   : "#1E3A5F",
    "success_fg"  : "#27AE60",
    "error_fg"    : "#E74C3C",
    "border"      : "#D5DBDB",
}

FONTS = {
    "header"  : ("Segoe UI", 18, "bold"),
    "subhead" : ("Segoe UI", 11, "bold"),
    "label"   : ("Segoe UI", 10),
    "entry"   : ("Segoe UI", 10),
    "button"  : ("Segoe UI", 11, "bold"),
    "result"  : ("Segoe UI", 14, "bold"),
    "small"   : ("Segoe UI", 9),
}


# =============================================================================
# INPUT VALIDATION HELPERS
# =============================================================================

def _validate_positive_int(value: str, field: str, low: int = 0, high: int = 9999) -> int:
    """Parse *value* as an integer in [low, high] or raise ValueError."""
    try:
        v = int(value)
    except ValueError:
        raise ValueError(f"'{field}' must be a whole number.")
    if not (low <= v <= high):
        raise ValueError(f"'{field}' must be between {low} and {high}.")
    return v


def _validate_positive_float(value: str, field: str, low: float = 1.0) -> float:
    """Parse *value* as a positive float or raise ValueError."""
    try:
        v = float(value)
    except ValueError:
        raise ValueError(f"'{field}' must be a number.")
    if v < low:
        raise ValueError(f"'{field}' must be at least {low}.")
    return v


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class HousePricePredictorApp:
    """
    Tkinter GUI for house price prediction.

    Layout
    ──────
    ┌──────────────────────────────────────────┐
    │  HEADER (title + subtitle)               │
    ├──────────────────────────────────────────┤
    │  LEFT PANEL         │  RIGHT PANEL       │
    │  (numeric inputs)   │  (dropdown inputs) │
    ├──────────────────────────────────────────┤
    │  RESULT PANEL (predicted price)          │
    ├──────────────────────────────────────────┤
    │  FOOTER (metrics)                        │
    └──────────────────────────────────────────┘
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_root()

        # Load model artefacts (train automatically if missing)
        self._payload = self._load_or_train_model()

        # Build the UI
        self._build_ui()

    # ── initialisation ───────────────────────────────────────────────────────

    def _configure_root(self):
        self.root.title("🏠  House Price Prediction System")
        self.root.geometry("860x680")
        self.root.resizable(True, True)
        self.root.configure(bg=COLOURS["bg"])
        # Centre window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  - 860) // 2
        y = (self.root.winfo_screenheight() - 680) // 2
        self.root.geometry(f"860x680+{x}+{y}")

    def _load_or_train_model(self) -> dict:
        """Try to load saved model; if missing or incompatible, run the training pipeline."""
        try:
            return ModelSerializer.load(MODEL_PATH)
        except (FileNotFoundError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
            messagebox.showinfo(
                "Model Load Failed",
                "The saved model could not be loaded or is incompatible.\nTraining now — this may take a moment …",
            )
            return run_training_pipeline()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_form()
        self._build_result_panel()
        self._build_footer()

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLOURS["header_bg"], pady=14)
        header.pack(fill="x")

        tk.Label(
            header, text="🏠  House Price Prediction System",
            font=FONTS["header"],
            bg=COLOURS["header_bg"], fg=COLOURS["header_fg"],
        ).pack()

        tk.Label(
            header,
            text="Powered by Linear Regression  •  Enter property details to get an instant estimate",
            font=FONTS["small"],
            bg=COLOURS["header_bg"], fg="#AED6F1",
        ).pack(pady=(2, 0))

    def _build_form(self):
        """Two-column form inside a white card."""
        outer = tk.Frame(self.root, bg=COLOURS["bg"], padx=20, pady=12)
        outer.pack(fill="both", expand=True)

        card = tk.Frame(outer, bg=COLOURS["frame_bg"],
                        highlightbackground=COLOURS["border"],
                        highlightthickness=1, bd=0)
        card.pack(fill="both", expand=True)

        # Section heading
        tk.Label(
            card, text="Property Details",
            font=FONTS["subhead"],
            bg=COLOURS["frame_bg"], fg=COLOURS["accent"],
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(14, 4))

        ttk.Separator(card, orient="horizontal").grid(
            row=1, column=0, columnspan=4, sticky="ew", padx=20, pady=(0, 10)
        )

        # ── Numeric fields (left two columns) ────────────────────────────────
        numeric_fields = [
            ("Area (sq ft)",      "area",      "e.g. 1500"),
            ("Bedrooms",          "bedrooms",  "1–10"),
            ("Bathrooms",         "bathrooms", "1–10"),
            ("Floors",            "floors",    "1–5"),
            ("Parking Spaces",    "parking",   "0–5"),
            ("House Age (years)", "house_age", "0–100"),
        ]

        self._entries: dict[str, tk.StringVar] = {}

        for idx, (label, key, placeholder) in enumerate(numeric_fields):
            row = idx + 2
            tk.Label(
                card, text=label, font=FONTS["label"],
                bg=COLOURS["frame_bg"], fg=COLOURS["label_fg"], anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(20, 6), pady=5)

            var = tk.StringVar()
            self._entries[key] = var

            entry = ttk.Entry(card, textvariable=var, font=FONTS["entry"], width=18)
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 20), pady=5)

            # Placeholder-style hint label
            tk.Label(
                card, text=placeholder, font=FONTS["small"],
                bg=COLOURS["frame_bg"], fg=COLOURS["subtle_fg"],
            ).grid(row=row, column=1, sticky="e", padx=(0, 24))

        # ── Dropdown fields (right two columns) ──────────────────────────────
        location_opts   = self._payload["location_classes"]
        furnishing_opts = self._payload["furnishing_classes"]

        dropdown_fields = [
            ("Location",         "location",   location_opts),
            ("Furnishing Status","furnishing",  furnishing_opts),
        ]

        self._combos: dict[str, tk.StringVar] = {}

        for idx, (label, key, options) in enumerate(dropdown_fields):
            row = idx + 2
            tk.Label(
                card, text=label, font=FONTS["label"],
                bg=COLOURS["frame_bg"], fg=COLOURS["label_fg"], anchor="w",
            ).grid(row=row, column=2, sticky="w", padx=(20, 6), pady=5)

            var = tk.StringVar(value=options[0])
            self._combos[key] = var

            combo = ttk.Combobox(
                card, textvariable=var,
                values=options, state="readonly",
                font=FONTS["entry"], width=18,
            )
            combo.grid(row=row, column=3, sticky="ew", padx=(0, 20), pady=5)

        # Column weights so the form stretches nicely
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)

        # ── Predict button ────────────────────────────────────────────────────
        btn_frame = tk.Frame(card, bg=COLOURS["frame_bg"])
        btn_frame.grid(row=8, column=0, columnspan=4, pady=(10, 16))

        self._predict_btn = tk.Button(
            btn_frame,
            text="  🔍  Predict Price  ",
            font=FONTS["button"],
            bg=COLOURS["accent"], fg="white",
            activebackground=COLOURS["accent_hover"], activeforeground="white",
            relief="flat", cursor="hand2", padx=18, pady=8,
            command=self._on_predict,
        )
        self._predict_btn.pack(side="left", padx=8)

        clear_btn = tk.Button(
            btn_frame,
            text="  ✖  Clear  ",
            font=FONTS["button"],
            bg="#95A5A6", fg="white",
            activebackground="#7F8C8D", activeforeground="white",
            relief="flat", cursor="hand2", padx=14, pady=8,
            command=self._clear_form,
        )
        clear_btn.pack(side="left", padx=8)

    def _build_result_panel(self):
        """Green-tinted panel that shows the predicted price."""
        self._result_var  = tk.StringVar(value="Enter property details and click 'Predict Price'")
        self._result_colour = COLOURS["subtle_fg"]

        panel = tk.Frame(self.root, bg=COLOURS["result_bg"],
                         highlightbackground=COLOURS["border"],
                         highlightthickness=1, pady=14)
        panel.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(
            panel, text="Estimated Price",
            font=FONTS["subhead"],
            bg=COLOURS["result_bg"], fg=COLOURS["accent"],
        ).pack()

        self._result_label = tk.Label(
            panel, textvariable=self._result_var,
            font=FONTS["result"],
            bg=COLOURS["result_bg"], fg=self._result_colour,
            wraplength=700,
        )
        self._result_label.pack(pady=(4, 0))

    def _build_footer(self):
        """Small strip showing the model metrics."""
        metrics = self._payload.get("metrics", {})
        r2  = metrics.get("R² Score", 0)
        mae = metrics.get("MAE", 0)
        mse = metrics.get("MSE", 0)
        rmse = metrics.get("RMSE", 0)

        footer = tk.Frame(self.root, bg=COLOURS["header_bg"], pady=6)
        footer.pack(fill="x", side="bottom")

        info = (
            f"Model: Linear Regression   "
            f"R²: {r2:.4f}   "
            f"MAE: ₹{mae:,.0f}   "
            f"RMSE: ₹{rmse:,.0f}"
        )
        tk.Label(
            footer, text=info,
            font=FONTS["small"],
            bg=COLOURS["header_bg"], fg="#AED6F1",
        ).pack()

    # ── event handlers ───────────────────────────────────────────────────────

    def _on_predict(self):
        """Validate inputs, run prediction, display result."""
        try:
            inputs = self._collect_and_validate()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        try:
            price = self._predict(inputs)
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))
            return

        # Format and display price
        price_millions = price / 1_000_000
        display = (
            f"₹ {price:,.0f}"
            f"   ({price_millions:.2f} Million)"
        )
        self._result_var.set(display)
        self._result_label.configure(fg=COLOURS["success_fg"])

    def _collect_and_validate(self) -> dict:
        """Read form values, validate types/ranges, and return a clean dict."""
        e = self._entries
        c = self._combos

        return {
            "Area"       : _validate_positive_float(e["area"].get(),      "Area",            low=100.0),
            "Bedrooms"   : _validate_positive_int  (e["bedrooms"].get(),  "Bedrooms",        low=1, high=10),
            "Bathrooms"  : _validate_positive_int  (e["bathrooms"].get(), "Bathrooms",       low=1, high=10),
            "Floors"     : _validate_positive_int  (e["floors"].get(),    "Floors",          low=1, high=5),
            "Parking"    : _validate_positive_int  (e["parking"].get(),   "Parking Spaces",  low=0, high=10),
            "HouseAge"   : _validate_positive_int  (e["house_age"].get(), "House Age",       low=0, high=100),
            "Location"   : c["location"].get(),
            "Furnishing" : c["furnishing"].get(),
        }

    def _predict(self, inputs: dict) -> float:
        """Transform inputs and return the predicted price (float)."""
        preprocessor = self._payload["preprocessor"]
        model        = self._payload["model"]

        X_scaled = preprocessor.transform_single(inputs)
        price    = model.predict(X_scaled)[0]
        return max(price, 0)   # clip to zero in case of negative extrapolation

    def _clear_form(self):
        """Reset all entry fields and the result label."""
        for var in self._entries.values():
            var.set("")
        self._result_var.set("Enter property details and click 'Predict Price'")
        self._result_label.configure(fg=COLOURS["subtle_fg"])


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    root = tk.Tk()

    # Apply ttk theme (clam looks clean cross-platform)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TEntry",  padding=4, relief="flat", borderwidth=1)
    style.configure("TCombobox", padding=4)

    app = HousePricePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
