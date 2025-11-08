# -*- coding: utf-8 -*-
# Frame09_EL.py — Controller cho màn Expected Loss (MVC)
"""
Controller / pipeline cho Frame09 (Expected Loss).
Đặt file này tại Function/Frame09_EL.py (như cấu trúc project của bạn).
"""

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, Menu
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from QMess.Qmess_calling import Qmess
# View (UI layout)
from Frame.Frame09.ui_Frame09 import Frame09

from matplotlib.font_manager import FontProperties

# === Load Crimson Pro font manually ===
FONT_DIR = os.path.join(os.path.dirname(__file__), "../Font/Crimson_Pro/static")
if os.path.exists(FONT_DIR):
    for font_file in os.listdir(FONT_DIR):
        if font_file.lower().endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, font_file))
    matplotlib.rcParams["font.family"] = "Crimson Pro"
    print(f"[INFO] Loaded custom font: Crimson Pro ({len(os.listdir(FONT_DIR))} files)")
else:
    print(f"[WARN] Font directory not found: {FONT_DIR}")

# -*- coding: utf-8 -*-
# Frame09_EL.py — Controller cho màn Expected Loss (MVC)
"""
Controller / pipeline cho Frame09 (Expected Loss).
Đặt file này tại Function/Frame09_EL.py (như cấu trúc project của bạn).
"""

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, Menu
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

# View (UI layout)
from Frame.Frame09.ui_Frame09 import Frame09

from matplotlib.font_manager import FontProperties

# === Load Crimson Pro font manually ===
FONT_DIR = os.path.join(os.path.dirname(__file__), "../Font/Crimson_Pro/static")
if os.path.exists(FONT_DIR):
    for font_file in os.listdir(FONT_DIR):
        if font_file.lower().endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, font_file))
    matplotlib.rcParams["font.family"] = "Crimson Pro"
    print(f"[INFO] Loaded custom font: Crimson Pro ({len(os.listdir(FONT_DIR))} files)")
else:
    print(f"[WARN] Font directory not found: {FONT_DIR}")

# ===================== GLOBAL CONFIG =====================
# NOTE: we do NOT force 'Agg' here because we embed matplotlib in Tkinter via FigureCanvasTkAgg.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

sns.set_style("whitegrid")
custom_palette = ['#644E94', '#9B86C2', '#E3D4E0']
bg_color = "#F9F7FB"
text_color = "#2E2E2E"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.normpath(os.path.join(BASE_DIR, "../Dataset/Output/"))
FILE_CLUSTER = os.path.join(PATH, "df_cluster_full.csv")
FILE_CHURN = os.path.join(PATH, "churn_predictions_preview.csv")
FILE_RAW = os.path.join(PATH, "df_raw_dashboard.csv")

# Safe load raw (may not exist during dev)
try:
    df_raw = pd.read_csv(FILE_RAW)
except Exception:
    df_raw = pd.DataFrame()
    print(f"[WARN] Could not load df_raw from {FILE_RAW} — proceeding with empty df_raw.")

# ===================== DATA / MODEL FUNCTIONS =====================
def load_and_preprocess():
    """Đọc df_cluster + df_churn, xử lý missing và merge.
    Trả về: df (merged), df_cluster (original standardized)
    """
    if not os.path.exists(FILE_CLUSTER) or not os.path.exists(FILE_CHURN):
        raise FileNotFoundError(f"Missing files: {FILE_CLUSTER=} or {FILE_CHURN=}")

    df_cluster = pd.read_csv(FILE_CLUSTER)
    df_churn = pd.read_csv(FILE_CHURN)
    print(f"[INFO] Cluster: {df_cluster.shape} | Churn: {df_churn.shape}")

    # Fill missing numbers with median, cats with mode
    for col in df_cluster.select_dtypes(include=[np.number]).columns:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())
    for col in df_cluster.select_dtypes(exclude=[np.number]).columns:
        if df_cluster[col].isnull().any():
            mode_val = df_cluster[col].mode()
            df_cluster[col] = df_cluster[col].fillna(mode_val.iloc[0] if not mode_val.empty else "")

    # standardize cluster column name
    cluster_col = [c for c in df_cluster.columns if 'cluster' in c.lower()]
    if cluster_col:
        df_cluster = df_cluster.rename(columns={cluster_col[0]: 'cluster'})
    else:
        # try fallback
        if 'cluster' not in df_cluster.columns:
            df_cluster['cluster'] = np.nan

    # Merge on CustomerID / Customer_ID
    left_key = "Customer_ID" if "Customer_ID" in df_churn.columns else "CustomerID"
    right_key = "CustomerID" if "CustomerID" in df_cluster.columns else None
    if right_key is None:
        # if cluster lacks CustomerID, try index align (risky)
        df = pd.concat([df_churn.reset_index(drop=True), df_cluster.reset_index(drop=True)], axis=1)
    else:
        df = pd.merge(df_churn, df_cluster, left_on=left_key, right_on=right_key, how="left")

    # resolve possible cluster_x / cluster_y
    if 'cluster_x' in df.columns and 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'}).drop(columns=['cluster_x'])
    elif 'cluster_y' in df.columns:
        df = df.rename(columns={'cluster_y': 'cluster'})
    elif 'cluster_x' in df.columns:
        df = df.rename(columns={'cluster_x': 'cluster'})

    # Ensure CustomerID present
    if 'CustomerID' not in df.columns and left_key in df.columns:
        df = df.rename(columns={left_key: 'CustomerID'})

    df = df.sort_values("CustomerID").reset_index(drop=True)
    df_cluster = df_cluster.sort_values("CustomerID") if "CustomerID" in df_cluster.columns else df_cluster
    df_cluster = df_cluster.reset_index(drop=True)
    return df, df_cluster


def train_churn_model(df, df_cluster):
    """Train small set of classifiers via KFold; attach proba_churn_model back to df.
    Returns: df (with proba_churn_model), X_train (DataFrame used to fit scaler later)
    """
    required_y = "pred_churn"
    if required_y not in df.columns:
        raise KeyError(f"Target column '{required_y}' not found in merged df.")

    leak_vars = ["proba_churn", "proba_churn_model", "ExpectedLoss_pred"]
    leak_vars = [v for v in leak_vars if v in df_cluster.columns or v in df.columns]
    print(f"[INFO] Removing potential leakage vars: {leak_vars}")

    X_no_leak = df_cluster.drop(columns=[c for c in leak_vars if c in df_cluster.columns], errors='ignore')
    feature_cols = [c for c in X_no_leak.columns if c not in ["CustomerID", "cluster"]]
    if not feature_cols:
        raise ValueError("No feature columns found in df_cluster after removing ids/leakage.")
    X = X_no_leak[feature_cols]
    y = df[required_y]

    # fill na in X
    for c in X.columns:
        if X[c].dtype.kind in 'biufc':
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna("")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2,
                                                        random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=SEED, stratify=y_temp)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Logistic": LogisticRegression(max_iter=1000, n_jobs=1, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=1, use_label_encoder=False)
    }

    results = []
    for name, model in models.items():
        auc_scores = []
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_va)[:, 1]
                auc_scores.append(roc_auc_score(y_va, y_pred_proba))
            except Exception as ex:
                print(f"[WARN] model {name} failed during fold fit/predict: {ex}")
        if auc_scores:
            results.append({"Model": name, "Mean_AUC": np.mean(auc_scores)})
        else:
            results.append({"Model": name, "Mean_AUC": np.nan})

    df_result = pd.DataFrame(results)
    print("\nKFold AUC Results:")
    print(df_result)

    # choose best available
    if df_result["Mean_AUC"].isnull().all():
        # fallback: use RandomForest
        best_model_name = "RandomForest"
        print("[WARN] No valid AUC scores — fallback to RandomForest.")
    else:
        best_model_name = df_result.loc[df_result["Mean_AUC"].idxmax(), "Model"]
    print(f"Best churn model: {best_model_name}")

    best_model = models.get(best_model_name, RandomForestClassifier(n_estimators=200, random_state=SEED))
    best_model.fit(X_train, y_train)
    df["proba_churn_model"] = best_model.predict_proba(X)[:, 1]

    # (Optional) show ROC on test if available
    try:
        fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{best_model_name} (AUC={roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.3f})')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Churn Model")
        plt.legend()
        plt.tight_layout()
        plt.close()  # close to avoid displaying in non-UI contexts
    except Exception:
        pass

    return df, X_train


def compute_expected_loss(df, X_train):
    """Scale 'Order Value' based on training index and compute ExpectedLoss_score."""
    if "Order Value" not in df.columns:
        raise KeyError("Column 'Order Value' not found in df for scaling Expected Loss.")
    if "proba_churn_model" not in df.columns:
        raise KeyError("Column 'proba_churn_model' not found — run churn model first.")

    scaler = MinMaxScaler()
    # safe indexing: ensure indices exist
    train_idx = X_train.index.intersection(df.index)
    if train_idx.empty:
        # fallback fit on whole
        scaler.fit(df[["Order Value"]].fillna(0))
    else:
        scaler.fit(df.loc[train_idx, ["Order Value"]].fillna(0))
    df["OrderValue_scaled"] = scaler.transform(df[["Order Value"]].fillna(0))
    df["ExpectedLoss_score"] = df["proba_churn_model"] * df["OrderValue_scaled"]
    print("[INFO] ExpectedLoss_score calculated successfully.")
    return df


def dual_expected_loss(df, df_cluster, PATH=PATH):
    """Train two regressors (full vs no-order) và xuất bảng hiển thị cho UI (Cluster 1–3)."""
    print("[STEP] Building Dual Expected Loss models...")

    drop_leak = ["ExpectedLoss_score", "OrderValue_scaled", "proba_churn_model"]
    df_model = df.drop(columns=[c for c in drop_leak if c in df.columns], errors='ignore')

    target_col = "ExpectedLoss_score"
    if target_col not in df.columns:
        raise KeyError(f"{target_col} missing. Compute ExpectedLoss_score before dual_expected_loss().")

    y_full = df[target_col]
    cols_full = [c for c in df_model.columns if c not in ["CustomerID", "Customer_ID", "cluster"]]
    cols_noorder = [c for c in cols_full if not any(x in c.lower() for x in ["order", "price", "value"])]

    def encode_df(df_in):
        df_out = df_in.copy()
        for c in df_out.columns:
            if df_out[c].dtype == "object" or df_out[c].dtype.name == "category":
                le = LabelEncoder()
                df_out[c] = le.fit_transform(df_out[c].astype(str))
            else:
                df_out[c] = df_out[c].fillna(df_out[c].median() if df_out[c].dtype.kind in 'biufc' else 0)
        return df_out

    X_full = encode_df(df_model[cols_full])
    X_noorder = encode_df(df_model[cols_noorder])

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noorder, y_full, test_size=0.2, random_state=SEED)

    rf_full = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_noorder = RandomForestRegressor(random_state=SEED, n_jobs=1)
    rf_full.fit(Xf_train, yf_train)
    rf_noorder.fit(Xn_train, yn_train)

    # === Xuất file dual map & cluster summary ===
    df_dual_map = pd.DataFrame({
        "ExpectedLoss_full_pred": rf_full.predict(Xf_test),
        "ExpectedLoss_noOrder_pred": rf_noorder.predict(Xn_test)
    }, index=Xf_test.index)

    if "cluster" in df_cluster.columns:
        df_dual_map["cluster"] = df_cluster.loc[df_dual_map.index, "cluster"].values
    else:
        df_dual_map["cluster"] = np.nan

    # ⚙️ dịch cluster lên 1 để thành 1–3 thay vì 0–2
    df_dual_map["cluster"] = df_dual_map["cluster"] + 1

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_dual_map[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]]),
        columns=["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"],
        index=df_dual_map.index
    )
    df_scaled["cluster"] = df_dual_map["cluster"]
    df_scaled["risk_segment"] = np.select(
        [
            (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6) & (df_scaled["ExpectedLoss_full_pred"] >= 0.6),
            (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6),
            (df_scaled["ExpectedLoss_full_pred"] >= 0.6)
        ],
        ["High Behavior + High Value", "High Behavior Only", "High Value Only"],
        default="Low Risk"
    )

    os.makedirs(PATH, exist_ok=True)
    df_scaled.to_csv(os.path.join(PATH, "expected_loss_dual_map.csv"), index=False)
    df_scaled.groupby("cluster")[["ExpectedLoss_full_pred", "ExpectedLoss_noOrder_pred"]].mean().to_csv(
        os.path.join(PATH, "expected_loss_cluster_summary.csv")
    )

    # === Chuẩn bị bảng hiển thị cho UI ===
    df_pred = df.copy()
    df_pred["ExpectedLoss_real"] = df_pred.get("Order Value", 0) * df_pred.get("proba_churn_model", 0)

    df_display = df_pred.rename(columns={
        "CustomerID": "Customer ID",
        "cluster": "Cluster",
        "Order Value": "Order Value (1–3)"
    })
    df_display["Cluster"] = df_display["Cluster"] + 1  # ⚙️ shift +1 ở đây nữa

    df_display["Churn Probability (%)"] = (df_display.get("proba_churn_model", 0) * 100).round(1)
    df_display["Expected Loss Score (%)"] = (df_display.get("ExpectedLoss_score", 0) * 100).round(1)
    df_display["Expected Loss (Value 1–3)"] = df_display.get("ExpectedLoss_real", 0).round(2)

    display_cols = [
        "Customer ID", "Cluster", "Order Value (1–3)",
        "Churn Probability (%)", "Expected Loss Score (%)",
        "Expected Loss (Value 1–3)"
    ]
    df_display = df_display[[c for c in display_cols if c in df_display.columns]]
    df_display.to_csv(os.path.join(PATH, "expected_loss_top50_display.csv"), index=False)

    print(f"[INFO] Saved readable version for UI → expected_loss_top50_display.csv")
    print("[DONE] Dual Expected Loss analysis complete.")

# ===================== UI HELPERS =====================
def load_expected_loss_table(PATH=PATH):
    """Đọc bảng expected_loss_top50_display.csv để hiển thị trong UI (Table)."""
    file_path = os.path.join(PATH, "expected_loss_top50_display.csv")
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded expected_loss_top50_display.csv ({df.shape})")
    return df


# ===================== CONTROLLER CLASS (inherits View) =====================
class Frame09_EL(Frame09):
    """Frame09 (UI + Expected Loss handling) — Controller: xử lý và gắn callback vào View."""

    def __init__(self, parent, controller=None):
        super().__init__(parent, controller)
        self.controller = controller
        self.df_display = pd.DataFrame()
        self.selected_cluster = None

        # override view callbacks to controller methods
        self.on_search_customer = self._on_search_customer
        self.on_cluster_dropdown = self._on_cluster_selected

        # --- Re-bind UI events to the overridden controller callbacks ---
        try:
            # Unbind previous binding and bind to current (calls the attribute at runtime)
            self.entry_id.unbind("<Return>")
            # bind via a small lambda so it will call the *current* self.on_search_customer
            self.entry_id.bind("<Return>", lambda e: self.on_search_customer(e))
        except Exception as e:
            print("[WARN] Re-binding entry_id Return failed:", e)


        # attach run pipeline action if you add a button later (kept for completeness)
        self.on_run_pipeline = self._on_run_pipeline

        # --- Load full expected loss dataset for searching (if available) ---
        full_file = os.path.join(PATH, "expected_loss_by_customer_display.csv")
        if os.path.exists(full_file):
            try:
                self.df_full = pd.read_csv(full_file, encoding="utf-8-sig")
                self.df_full.columns = [c.replace("â€“", "-").replace("–", "-").strip() for c in self.df_full.columns]
                print(f"[INFO] Loaded full expected loss dataset for searching: {self.df_full.shape}")
                print("[DEBUG] Columns in df_full:", list(self.df_full.columns))
            except Exception as e:
                print(f"[WARN] Failed to load full expected loss file: {e}")
                self.df_full = pd.DataFrame()
        else:
            self.df_full = pd.DataFrame()

        # --- Setup table + data ---
        self._setup_table()
        self.load_table_data()  # bây giờ update_cards() sẽ dùng df_full đã có
        try:
            self._on_show_dualmap()
            self._on_show_clusterbar()
        except Exception as e:
            print("[INFO] Charts not drawn on init:", e)

        if os.path.exists(full_file):
            try:
                # đọc với encoding chuẩn để tránh lỗi "â€“"
                self.df_full = pd.read_csv(full_file, encoding="utf-8-sig")

                # chuẩn hóa tên cột: sửa mọi ký tự "â€“" hoặc "–" thành "-"
                self.df_full.columns = [c.replace("â€“", "-").replace("–", "-").strip() for c in self.df_full.columns]

                print(f"[INFO] Loaded full expected loss dataset for searching: {self.df_full.shape}")
                print("[DEBUG] Columns in df_full:", list(self.df_full.columns))
            except Exception as e:
                print(f"[WARN] Failed to load full expected loss file: {e}")
                self.df_full = pd.DataFrame()
        else:
            self.df_full = pd.DataFrame()

    # ----------------- Table setup (custom pastel scroll + style) -----------------
    def _setup_table(self):
        HEADER_BG = "#B79AC8"
        ROW_EVEN = "#FFFFFF"
        ROW_ODD = "#F7F4F7"
        TEXT = "#2E2E2E"

        self.frame_table.config(bg="#F9F7FB", bd=0, highlightthickness=0)

        columns = [
            "Customer ID", "Cluster", "Order Value (1–3)",
            "Churn Probability (%)", "Expected Loss Score (%)",
            "Expected Loss (Value 1–3)"
        ]

        style = ttk.Style()
        style.theme_use("default")

        # --- Remove outer border ---
        style.configure(
            "Custom.Treeview",
            background=ROW_EVEN,
            fieldbackground=ROW_EVEN,
            foreground=TEXT,
            bordercolor=ROW_EVEN,
            relief="flat"
        )
        style.layout("Custom.Treeview", [
            ('Treeview.treearea', {'sticky': 'nswe'})
        ])

        self.tree = ttk.Treeview(
            self.frame_table,
            columns=columns,
            show="headings",
            height=12,
            style="Custom.Treeview",
            selectmode="browse"
        )
        style.configure("Custom.Treeview", borderwidth=0, relief="flat")

        # --- Columns ---
        col_widths = {
            "Customer ID": 130,
            "Cluster": 80,
            "Order Value (1–3)": 150,
            "Churn Probability (%)": 170,
            "Expected Loss Score (%)": 180,
            "Expected Loss (Value 1–3)": 200
        }
        for col in columns:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=col_widths.get(col, 150),
                             anchor="center", stretch=True)

        # --- Scrollbar pastel mảnh, bo tròn mềm ---
        scroll_canvas = tk.Canvas(
            self.frame_table,
            width=6,
            bg="#F7F4F7",
            highlightthickness=0,
            bd=0
        )
        scroll_canvas.pack(side="right", fill="y", padx=(2, 6), pady=6)

        scroll_thumb = scroll_canvas.create_rectangle(
            1, 0, 5, 40,
            outline="",
            fill="#C9C4CE",
            width=0
        )

        self._scroll_drag_start = None
        self._scroll_start_pos = None

        def update_thumb(first, last):
            """Cập nhật thumb khi Treeview cuộn"""
            height = scroll_canvas.winfo_height()
            first, last = float(first), float(last)
            thumb_len = max(30, (last - first) * height)
            y1 = first * height
            y2 = y1 + thumb_len
            scroll_canvas.coords(scroll_thumb, 1, y1 + 3, 5, y2 - 3)

        def scroll(*args):
            """Cuộn treeview và cập nhật thumb"""
            self.tree.yview(*args)
            update_thumb(*self.tree.yview())

        def on_mousewheel(event):
            self.tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
            update_thumb(*self.tree.yview())
            return "break"

        # === Kéo thumb thực tế theo pixel ===
        def on_thumb_press(event):
            self._scroll_drag_start = event.y
            self._scroll_start_pos = self.tree.yview()[0]

        def on_thumb_drag(event):
            if self._scroll_drag_start is None:
                return
            dy = event.y - self._scroll_drag_start
            height = scroll_canvas.winfo_height()
            first, last = self.tree.yview()
            visible = last - first
            thumb_len = max(30, visible * height)
            scrollable_height = height - thumb_len
            delta_fraction = dy / scrollable_height * (1 - visible)
            new_first = max(0, min(1 - visible, self._scroll_start_pos + delta_fraction))
            self.tree.yview_moveto(new_first)
            update_thumb(*self.tree.yview())

        def on_thumb_release(event):
            self._scroll_drag_start = None
            self._scroll_start_pos = None

        # --- Bind ---
        scroll_canvas.tag_bind(scroll_thumb, "<ButtonPress-1>", on_thumb_press)
        scroll_canvas.tag_bind(scroll_thumb, "<B1-Motion>", on_thumb_drag)
        scroll_canvas.tag_bind(scroll_thumb, "<ButtonRelease-1>", on_thumb_release)
        self.tree.configure(yscrollcommand=update_thumb)
        self.tree.bind("<MouseWheel>", on_mousewheel)

        # --- Pack Treeview ---
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)
        self.frame_table.pack_propagate(False)

        # Header style
        style.configure(
            "Treeview.Heading",
            background=HEADER_BG,
            foreground="white",
            font=("Crimson Pro Bold", 13),
            relief="flat",
            padding=(4, 6)
        )

        # Body style
        style.configure(
            "Treeview",
            background=ROW_EVEN,
            foreground=TEXT,
            rowheight=30,
            fieldbackground=ROW_EVEN,
            font=("Crimson Pro Bold", 11)
        )

        style.map("Treeview", background=[("selected", "#E8DAEF")])

        # Xen kẽ màu dòng
        self.tree.tag_configure("evenrow", background=ROW_EVEN)
        self.tree.tag_configure("oddrow", background=ROW_ODD)

    def load_table_data(self):
        df = load_expected_loss_table(PATH)

        # --- Chuẩn hóa Cluster về 1–3 thay vì 0–2 ---
        if not df.empty:
            col_cluster = next((c for c in df.columns if "cluster" in c.lower()), None)
            if col_cluster:
                try:
                    df[col_cluster] = pd.to_numeric(df[col_cluster], errors="coerce").fillna(0).astype(int) + 1
                except Exception:
                    pass

        if df.empty:
            self.df_display = pd.DataFrame()
            # update cards to zeros
            self.update_cards()
            return

        self.df_display = df
        self._populate_table(df)
        self.update_cards()

    def _populate_table(self, df):
        # clear and insert
        if not hasattr(self, "tree"):
            return
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, (_, row) in enumerate(df.iterrows()):
            tag = "evenrow" if idx % 2 == 0 else "oddrow"
            vals = [row.get(col, "") for col in df.columns]
            # if tree columns differ, fallback to row.values
            try:
                self.tree.insert("", "end", values=list(row.values), tags=(tag,))
            except Exception:
                self.tree.insert("", "end", values=vals, tags=(tag,))

    # ----------------- Callbacks -----------------
    def _on_search_customer(self, event=None):
        """Tìm kiếm khách hàng theo ID — chỉ trong bảng top50 (expected_loss_top50_display.csv)."""
        cid = self.entry_id.get().strip()
        if not cid:
            return

        # Luôn tìm trong df_display (top50)
        search_df = getattr(self, "df_display", pd.DataFrame())
        if search_df is None or search_df.empty:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu để tìm kiếm.")
            return

        # Xác định cột Customer ID
        col_customer = next((c for c in search_df.columns if "customer" in c.lower()), None)
        if not col_customer:
            messagebox.showinfo("Thông báo", "Không tìm thấy cột Customer trong dữ liệu.")
            return

        # Chuyển cột sang string, strip khoảng trắng
        df_temp = search_df.copy()
        df_temp[col_customer] = df_temp[col_customer].astype(str).str.strip()

        # Tìm chính xác trước
        df_filtered = df_temp[df_temp[col_customer].str.lower() == cid.lower()]

        # Nếu không thấy, thử tìm chứa chuỗi con (case-insensitive)
        if df_filtered.empty:
            df_filtered = df_temp[df_temp[col_customer].str.contains(cid, case=False, regex=False, na=False)]

        if df_filtered.empty:
            Qmess.popup_23(parent=self,
                        title="Warning",
                        subtitle=f"Do not found customer with ID '{cid}' in top 50 Expected Loss.")
            return

        # Chuẩn hóa tên cột khớp TreeView
        tree_cols = list(self.tree["columns"])
        rename_map = {"CustomerID": "Customer ID", "Customer_ID": "Customer ID", "cluster": "Cluster"}
        df_show = df_filtered.rename(columns={k: v for k, v in rename_map.items() if k in df_filtered.columns})
        df_show = df_show[[c for c in tree_cols if c in df_show.columns]]

        # Hiển thị kết quả
        self._populate_table(df_show)
        self.update_cards()

        print(f"[SEARCH] Found {len(df_show)} record(s) for {cid} in top50 file.")

    def _on_cluster_selected(self, cluster_name):
        """Lọc bảng theo cluster — hỗ trợ thêm 'All Cluster'."""
        # xử lý None hoặc All Cluster
        if cluster_name is None or str(cluster_name).lower() in ["all", "all cluster", "tất cả", "all clusters"]:
            self.entry_cluster.config(text="All Cluster")
            self._populate_table(self.df_display)
            self.update_cards()
            print("[FILTER] Showing all clusters.")
            return

        # Chuẩn hóa cluster label
        import re
        cluster_name = str(cluster_name).strip()
        m = re.search(r"(\d+)", cluster_name)
        cluster_val = int(m.group(1)) if m else 1

        self.entry_cluster.config(text=f"Cluster {cluster_val}")

        # Nếu chưa có dữ liệu → thoát
        if self.df_display.empty:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu để lọc.")
            return

        # Tìm cột cluster trong df_display
        col_cluster = next((c for c in self.df_display.columns if "cluster" in c.lower()), None)
        if not col_cluster:
            messagebox.showwarning("Lỗi", "Không tìm thấy cột Cluster trong dữ liệu.")
            return

        # Chuyển về str để tránh lỗi type (float/int)
        df_temp = self.df_display.copy()
        df_temp[col_cluster] = df_temp[col_cluster].astype(str).str.strip()

        # Lọc theo cluster được chọn
        df_filtered = df_temp[df_temp[col_cluster] == str(cluster_val)]

        if df_filtered.empty:
            messagebox.showinfo("Info", f"Không có khách hàng trong Cluster {cluster_val}.")
            return

        # Cập nhật bảng và thống kê
        self._populate_table(df_filtered)
        self.update_cards()
        print(f"[FILTER] Showing Cluster {cluster_val} ({len(df_filtered)} rows)")

        # === FIXED: chuẩn hóa cluster value ===
        import re
        m = re.search(r"(\d+)", cluster_name)
        cluster_val = int(m.group(1)) if m else 1
        self.entry_cluster.config(text=f"Cluster {cluster_val}")

        # Nếu chưa có dữ liệu → thoát
        if self.df_display.empty:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu để lọc.")
            return

        # Tìm cột cluster trong df_display
        col_cluster = next((c for c in self.df_display.columns if "cluster" in c.lower()), None)
        if not col_cluster:
            messagebox.showwarning("Lỗi", "Không tìm thấy cột Cluster trong dữ liệu.")
            return

        # Chuyển về str để tránh lỗi type (float/int)
        df_temp = self.df_display.copy()
        df_temp[col_cluster] = df_temp[col_cluster].astype(str).str.strip()

        # Lọc theo cluster được chọn
        df_filtered = df_temp[df_temp[col_cluster] == str(cluster_val)]

        if df_filtered.empty:
            messagebox.showinfo("Info", f"Không có khách hàng trong Cluster {cluster_val}.")
            return

        # Cập nhật bảng và thống kê
        self._populate_table(df_filtered)
        self.update_cards()
        print(f"[FILTER] Showing Cluster {cluster_val} ({len(df_filtered)} rows)")

    def _on_run_pipeline(self):
        """Chạy pipeline đầy đủ (load -> train -> compute -> dual) và reload UI files."""
        try:
            messagebox.showinfo("Processing", "Đang chạy Expected Loss pipeline...")
            df, df_cluster = load_and_preprocess()
            df, X_train = train_churn_model(df, df_cluster)
            df = compute_expected_loss(df, X_train)
            dual_expected_loss(df, df_cluster)
            messagebox.showinfo("Done", "Đã hoàn thành phân tích Expected Loss.")
            # reload tables and charts
            self.load_table_data()
            try:
                self._on_show_dualmap()
                self._on_show_clusterbar()
            except Exception as e:
                print("[WARN] drawing charts after pipeline:", e)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print("[ERROR] _on_run_pipeline:", e)

    # ----------------- Plot embed helpers -----------------
    def _on_show_dualmap(self):
        dual_file = os.path.join(PATH, "expected_loss_dual_map.csv")
        if not os.path.exists(dual_file):
            return
        df_scaled = pd.read_csv(dual_file)
        if "risk_segment" not in df_scaled.columns:
            df_scaled["risk_segment"] = np.select(
                [
                    (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6) & (df_scaled["ExpectedLoss_full_pred"] >= 0.6),
                    (df_scaled["ExpectedLoss_noOrder_pred"] >= 0.6),
                    (df_scaled["ExpectedLoss_full_pred"] >= 0.6)
                ],
                ["High Behavior + High Value", "High Behavior Only", "High Value Only"],
                default="Low Risk"
            )

        for w in self.frame_dualmap.winfo_children():
            w.destroy()

        palette_map = {
            "High Behavior + High Value": "#4B2E83",
            "High Behavior Only": "#7A5BA8",
            "High Value Only": "#B39CD0",
            "Low Risk": "#E3D4E0"
        }

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=df_scaled,
            x="ExpectedLoss_noOrder_pred",
            y="ExpectedLoss_full_pred",
            hue="risk_segment",
            palette=palette_map,
            s=60, alpha=0.9, edgecolor="white", linewidth=0.5, ax=ax
        )
        ax.axvline(0.6, color='#C9BFE5', linestyle='--', lw=1)
        ax.axhline(0.6, color='#C9BFE5', linestyle='--', lw=1)

        # áp dụng font & màu chữ
        text_color = "#7A467A"
        font_family = "Crimson Pro"

        ax.set_xlabel("Behavioral Expected Loss", fontfamily=font_family,
                      fontweight="normal", color=text_color)
        ax.set_ylabel("Financial Expected Loss", fontfamily=font_family,
                      fontweight="normal", color=text_color)

        # đổi màu tick label
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontweight("normal")
            label.set_color(text_color)

        font_prop = FontProperties(family="Crimson Pro", size=9, weight="normal")
        title_font_prop = FontProperties(family="Crimson Pro", size=10, weight="normal")

        legend = ax.legend(
            title="Risk Segment",
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.02),
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
            framealpha=0.8,
            markerscale=0.8,
            prop=font_prop,  # ép cỡ & font cho nội dung
            title_fontproperties=title_font_prop  # ép cỡ & font cho tiêu đề
        )

        # Màu chữ legend
        for text in legend.get_texts():
            text.set_color("#7A467A")
        legend.get_title().set_color("#7A467A")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_dualmap)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _on_show_clusterbar(self):
        heat_file = os.path.join(PATH, "expected_loss_cluster_summary.csv")
        if not os.path.exists(heat_file):
            return
        df = pd.read_csv(heat_file)
        if "cluster" not in df.columns:
            df = df.reset_index().rename(columns={"index": "cluster"})
        df = df.sort_values("ExpectedLoss_full_pred", ascending=True)

        for w in self.frame_bar.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        clusters = np.arange(len(df))
        bar_width = 0.4
        ax.barh(clusters - bar_width / 2, df["ExpectedLoss_full_pred"], height=bar_width,
                color="#644E94", label="Financial Risk")
        ax.barh(clusters + bar_width / 2, df["ExpectedLoss_noOrder_pred"], height=bar_width,
                color="#B39CD0", label="Behavioral Risk")

        text_color = "#7A467A"
        font_family = "Crimson Pro"

        ax.set_yticks(clusters)
        ax.set_yticklabels([f"Cluster {int(i) + 1}" for i in range(len(df))],
                           fontfamily=font_family, fontweight="normal", color=text_color)

        ax.set_xlabel("Mean Expected Loss (scaled)",
                      fontfamily=font_family, fontweight="normal", color=text_color)

        for label in ax.get_xticklabels():
            label.set_fontfamily(font_family)
            label.set_fontweight("normal")
            label.set_color(text_color)

        legend = ax.legend(fontsize=8, loc="lower right",
                           prop={'family': font_family, 'weight': 'normal'})
        plt.setp(legend.get_texts(), color=text_color)

        ax.grid(axis="x", linestyle="--", alpha=0.4)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_bar)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ----------------- Cards update -----------------
    def update_cards(self):
        """Cập nhật KPI — Total EL từ full dataset, High-Risk theo cluster hiển thị."""
        try:
            # --- 1️⃣ High Risk Count (tính từ bảng đang hiển thị) ---
            df_display = getattr(self, "df_display", pd.DataFrame())
            if not df_display.empty and "Churn Probability (%)" in df_display.columns:
                pd_series = pd.to_numeric(df_display["Churn Probability (%)"], errors="coerce").fillna(0)
                high_risk_count = (pd_series > 50).sum()
            else:
                high_risk_count = 0

            # --- 2️⃣ Total Expected Loss (tính từ file full) ---
            df_full = getattr(self, "df_full", pd.DataFrame())
            total_el = 0.0
            if not df_full.empty:
                # tìm cột Expected Loss, dù tên có thể khác nhau (vì encoding)
                col_el = next(
                    (c for c in df_full.columns if "expected" in c.lower() and "value" in c.lower()),
                    None
                )
                if col_el:
                    el_values = pd.to_numeric(
                        df_full[col_el].astype(str).str.replace(",", "."),
                        errors="coerce"
                    ).fillna(0)

                    # Nếu có giá trị âm → scale về 1–3 để cộng dương
                    if el_values.min() < 0:
                        el_values = (el_values - el_values.min()) / (el_values.max() - el_values.min()) * 2 + 1

                    total_el = el_values.sum()
                else:
                    print("[WARN] Không tìm thấy cột 'Expected Loss (Value 1–3)' trong df_full.")

            # --- 3️⃣ Cập nhật UI ---
            self.canvas.itemconfig(self.text_highrisk_value, text=str(int(high_risk_count)))
            self.canvas.itemconfig(self.text_totalel_value, text=f"{total_el:,.2f}")

            print(f"[INFO] update_cards → HighRisk={high_risk_count}, TotalEL={total_el:.2f}")

        except Exception as e:
            print("[ERROR] update_cards:", e)
            try:
                self.canvas.itemconfig(self.text_highrisk_value, text="0")
                self.canvas.itemconfig(self.text_totalel_value, text="0")
            except Exception:
                pass


# Safe load raw (may not exist during dev)
try:
    df_raw = pd.read_csv(FILE_RAW)
except Exception:
    df_raw = pd.DataFrame()
    print(f"[WARN] Could not load df_raw from {FILE_RAW} — proceeding with empty df_raw.")