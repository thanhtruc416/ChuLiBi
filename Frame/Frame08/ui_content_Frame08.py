# -*- coding: utf-8 -*-
# ui_content_Frame08.py

# =========================================================
# ÉP BACKEND TKAGG TRƯỚC KHI IMPORT CÁC BACKEND KHÁC
# =========================================================
import matplotlib
if matplotlib.get_backend().lower() != "tkagg":
    matplotlib.use("TkAgg")
print("[DEBUG] Matplotlib backend (after set):", matplotlib.get_backend())

from pathlib import Path
import sys
import tkinter as tk
from tkinter import Canvas, PhotoImage, Frame, Label

# Matplotlib cho embed chart (sau khi đã set TkAgg)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ==== import từ module thật (có fallback) ====
HAS_FUNCTIONS = True
try:
    from Function.Frame08_churn import (
        get_churn_data,
        _plot_churn_rate_by_segment,
        _plot_reasons_pie,
        _plot_feature_importance,
        _plot_shap_summary,
    )
except Exception as e:
    print(f"Warning: Could not import Function.Frame08_churn: {e}")
    HAS_FUNCTIONS = False
    # Fallback “dummy”
    def get_churn_data(): return None
    def _plot_churn_rate_by_segment(ax, df_core, df_original=None):
        ax.clear(); ax.text(0.5,0.5,"No data",ha="center",va="center")
    def _plot_reasons_pie(ax, feature_importance_df):
        ax.clear(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return [],[],[]
    def _plot_feature_importance(ax, feature_importance_df, top_n=10):
        ax.clear(); ax.text(0.5,0.5,"No data",ha="center",va="center")
    def _plot_shap_summary(fig, bundle, X):
        ax = fig.add_subplot(111); ax.text(0.5,0.5,"No data",ha="center",va="center")

# =========================================================
# Asset resolver
# =========================================================
OUTPUT_PATH = Path(__file__).parent
_ASSET_DIRS = [
    OUTPUT_PATH / "assets_Frame08_2",
    OUTPUT_PATH / "assets_Frame08",
    OUTPUT_PATH / "assets_frame0",
    OUTPUT_PATH / "assets",
    OUTPUT_PATH,
]

def relative_to_assets(path: str) -> Path:
    name = Path(path).name
    for d in _ASSET_DIRS:
        p = d / name
        if p.exists():
            return p
    low = name.lower()
    for d in _ASSET_DIRS:
        try:
            for f in d.iterdir():
                if f.name.lower() == low:
                    return f
        except Exception:
            pass
    return _ASSET_DIRS[0] / name

# =========================================================
# Thử import get_churn_data lần nữa theo PROJECT_ROOT (an toàn)
# =========================================================
try:
    PROJECT_ROOT = OUTPUT_PATH.parent.parent  # .../ChuLiBi
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Function.Frame08_churn import get_churn_data as _gcd2  # noqa
    get_churn_data = _gcd2
    HAS_FUNCTIONS = True
except Exception as e:
    print(f"Warning: Could not import Function.Frame08_churn (2nd try): {e}")

# Fallback data nếu không có Function
import numpy as np
import pandas as pd

def _fallback_data_dict() -> dict:
    N = 80
    df = pd.DataFrame({
        "Customer_ID": [f"CUS{i:03d}" for i in range(1, N+1)],
        "cluster": [i % 4 for i in range(1, N+1)],
        "proba_churn": np.random.uniform(0.05, 0.95, N),
    })
    df["pred_churn_label"] = (df["proba_churn"] >= 0.5).astype(int)
    df["cluster_label"] = "Cluster " + df["cluster"].astype(str)
    df_pred = df[["Customer_ID", "cluster", "proba_churn", "pred_churn_label", "cluster_label"]].copy()
    df_pred["proba_churn_pct"] = (df_pred["proba_churn"] * 100).round(2)

    return {
        "df": df.copy(),
        "df_core": df.copy(),
        "df_result": df_pred,
        "eval_metrics": pd.DataFrame([{"Model":"Baseline","AUC":0.80,"F1":0.70,"Precision":0.68,"Recall":0.72,"Accuracy":0.77}]),
        "feature_importance": pd.DataFrame({
            "Feature": [f"feat_{i}" for i in range(1, 11)],
            "Importance": np.linspace(0.23, 0.05, 10)
        }),
        "avg_churn": float(df["pred_churn_label"].mean()),
        "num_clusters": int(df["cluster"].nunique()),
        "best_model": "Logistic Regression",
        "bundle": None,
        "X": None,
    }

def _normalize_bundle(obj) -> dict:
    if obj is None:
        return _fallback_data_dict()

    if isinstance(obj, dict):
        d = obj.copy()
        if "df_result" not in d and "df_pred" in d:
            d["df_result"] = d["df_pred"]
        if "avg_churn" not in d and "metrics" in d and isinstance(d["metrics"], dict):
            try:
                d["avg_churn"] = float(d["metrics"].get("churn_rate", 0.0))
            except Exception:
                d["avg_churn"] = 0.0
        if "num_clusters" not in d and "df" in d:
            try:
                d["num_clusters"] = int(d["df"]["cluster"].nunique())
            except Exception:
                d["num_clusters"] = 4
        d.setdefault("best_model", "Logistic Regression")
        d.setdefault("bundle", None)
        d.setdefault("X", None)
        return d

    # dataclass fallback
    d = {}
    try:
        d["df"] = obj.df
        d["df_core"] = obj.df
        d["df_result"] = getattr(obj, "df_pred", obj.df)
        metrics = getattr(obj, "metrics", {}) or {}
        d["avg_churn"] = float(metrics.get("churn_rate", 0.0))
        d["num_clusters"] = int(obj.df["cluster"].nunique()) if "cluster" in obj.df.columns else 4
        d["best_model"] = "Logistic Regression"
        d["feature_importance"] = pd.DataFrame({
            "Feature": [f"feat_{i}" for i in range(1, 9)],
            "Importance": np.linspace(0.22, 0.06, 8)
        })
        d["bundle"] = None
        d["X"] = None
        return d
    except Exception:
        return _fallback_data_dict()

# Bảng dự đoán (builder)
HAS_TABLE_BUILDER = True
try:
    from .ui_content_Frame08_table import build_prediction_table, build_cluster_filter_dropdown
except Exception:
    try:
        from ui_content_Frame08_table import build_prediction_table, build_cluster_filter_dropdown
    except Exception as e:
        print(f"Warning: Could not import table builder: {e}")
        HAS_TABLE_BUILDER = False

if not HAS_TABLE_BUILDER:
    def build_prediction_table(parent, width, height, df_result=None):
        c = Canvas(parent, bg="#FFFFFF", width=width, height=height, bd=0, highlightthickness=0)
        c.create_text(10, 10, anchor="nw", text="(Table builder fallback)", font=("Crimson Pro", 12))
        return c
    def build_cluster_filter_dropdown(parent, df_result=None, on_filter_change=None):
        import tkinter.ttk as ttk
        var = tk.StringVar(value="All Clusters")
        cmb = ttk.Combobox(parent, values=["All Clusters","Cluster 0","Cluster 1","Cluster 2","Cluster 3"],
                           state="readonly", textvariable=var, width=20)
        if on_filter_change:
            cmb.bind("<<ComboboxSelected>>", lambda e: on_filter_change(var.get()))
        return cmb

# =========================================================
# Global cache
# =========================================================
_churn_data = None
def get_cached_data() -> dict:
    global _churn_data
    if _churn_data is not None:
        return _churn_data
    try:
        if HAS_FUNCTIONS:
            bundle = get_churn_data()
            _churn_data = _normalize_bundle(bundle)
        else:
            _churn_data = _fallback_data_dict()
        print("✓ Churn data prepared")
    except Exception as e:
        print("✗ get_churn_data failed, use fallback:", e)
        _churn_data = _fallback_data_dict()
    return _churn_data

# =========================================================
# Main builder
# =========================================================
def build_content(parent: tk.Widget, width: int, height: int) -> Canvas:
    canvas = Canvas(parent, bg="#D4C5D2", width=width, height=height,
                    bd=0, highlightthickness=0, relief="ridge")
    canvas.pack(fill="both", expand=True)

    _img_refs = []
    def _img(name: str):
        try:
            im = PhotoImage(file=relative_to_assets(name))
            _img_refs.append(im)
            return im
        except Exception as e:
            print(f"Could not load image {name}: {e}")
            return None
    canvas._img_refs = _img_refs

    data = get_cached_data()

    # ===== TOP KPIs =====
    img_avg = _img("image_AvgChurn.png")
    if img_avg: canvas.create_image(179.0, 101.0, image=img_avg)
    canvas.create_text(120, 120.0, anchor="nw", text="Avg Churn",
                       fill="#000000", font=("Young Serif", 18))
    avg_churn_value = f"{data.get('avg_churn', 0.0) * 100:.0f}%"
    canvas.create_text(135, 30, anchor="nw", text=avg_churn_value,
                       fill="#706093", font=("Kodchasan Regular", 40))

    img_clusters = _img("image_Clusters.png")
    if img_clusters: canvas.create_image(525.0, 101.0, image=img_clusters)
    canvas.create_text(475.0, 120.0, anchor="nw", text="Clusters",
                       fill="#000000", font=("Young Serif", 18))
    num_clusters = str(int(data.get("num_clusters", 4)))
    canvas.create_text(500.0, 30, anchor="nw", text=num_clusters,
                       fill="#706093", font=("Kodchasan Regular", 45))

    # ----- BEST MODEL (centered on the card) -----
    cx, cy = 880.0, 104.0
    img_best = _img("image_LogisticRegression.png")
    if img_best:
        canvas.create_image(cx, cy, image=img_best)
    canvas.create_text(cx, cy - 58, anchor="n", text="Best Model",
                       fill="#000000", font=("Young Serif", 18))

    import tkinter.font as tkfont
    import pandas as pd

    def _pretty_name(name: str) -> str:
        name = (name or "").strip().replace("\n", " ")
        mapping = {
            "LogisticRegression": "Logistic Regression",
            "RandomForestClassifier": "Random Forest",
            "ExtraTreesClassifier": "Extra Trees",
            "GradientBoostingClassifier": "Gradient Boosting",
            "XGBClassifier": "XGBoost",
            "LGBMClassifier": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "SVC": "SVM",
            "KNeighborsClassifier": "KNN",
            "DecisionTreeClassifier": "Decision Tree",
            "GaussianNB": "Naive Bayes",
            "Pipeline": "",
            "GridSearchCV": "",
            "RandomizedSearchCV": "",
        }
        return mapping.get(name, name)

    def _resolve_model_name(data: dict) -> str:
        txt = str(data.get("best_model", "") or "").strip()
        if txt and txt.lower() not in ("best", "pipeline"):
            return _pretty_name(txt)

        ev = data.get("eval_metrics")
        if isinstance(ev, pd.DataFrame) and not ev.empty and "Model" in ev.columns:
            first = str(ev["Model"].iloc[0]).strip()
            if first and first.lower() not in ("best", "pipeline"):
                return _pretty_name(first)

        bundle = data.get("bundle") or {}
        model = bundle.get("model") if isinstance(bundle, dict) else None
        est = model
        for attr in ("best_estimator_", "estimator"):
            if hasattr(est, attr):
                est = getattr(est, attr)
        try:
            from sklearn.pipeline import Pipeline
            if est is not None and isinstance(est, Pipeline) and est.steps:
                est = est.steps[-1][1]
        except Exception:
            pass
        name = getattr(est, "__class__", type("X", (), {})).__name__ if est is not None else ""
        name = _pretty_name(name)
        return name or "Logistic Regression"

    best_model_text = _resolve_model_name(data)
    fm = tkfont.Font(family="Young Serif", size=34)
    max_w = 240
    while fm.measure(best_model_text) > max_w and fm['size'] > 18:
        fm['size'] -= 1
    canvas.create_text(cx, cy - 10, anchor="n", text=best_model_text,
                       fill="#706093", font=fm)

    # ===== CHART 1: Churn rate by segment =====
    img_rate_bg = _img("image_ChurnRate.png")
    if img_rate_bg: canvas.create_image(354.0, 380.0, image=img_rate_bg)
    canvas.create_text(59.0, 213.0, anchor="nw",
                       text="Churn Rate by Customer Segment",
                       fill="#000000", font=("Young Serif", 20))
    chart_frame_1 = Frame(canvas, bg="#FFFFFF", width=420, height=300)
    canvas.create_window(350, 420, window=chart_frame_1, anchor="center")

    if data:
        try:
            axfig = Figure(figsize=(4.2, 3.0), dpi=100, facecolor='#FFFFFF')
            ax = axfig.add_subplot(111)

            # --- gọi hàm plot ---
            import inspect
            try:
                n_params = len(inspect.signature(_plot_churn_rate_by_segment).parameters)
            except Exception:
                n_params = 3
            if n_params >= 3:
                _plot_churn_rate_by_segment(ax, data['df_core'], data['df'])
            else:
                _plot_churn_rate_by_segment(ax, data['df_core'])

            # --- KHÔNG cần title (đã có phía trên) ---
            # ax.set_title("Churn Rate by Customer Segment", fontsize=12, color="#4A3F6A", pad=6)

            # --- căn layout để không bị cắt ---
            try:
                axfig.tight_layout(pad=0.8)
                axfig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.86)
            except Exception:
                pass

            # --- EMBED chart vào Frame ---
            fc = FigureCanvasTkAgg(axfig, master=chart_frame_1)
            fc.draw()
            fc.get_tk_widget().pack(fill="both", expand=True)

            # --- KHÔNG dùng hover nữa (bỏ toàn bộ mplcursors) ---

            print("✓ Churn rate chart created")
        except Exception as e:
            print("✗ Chart1 failed:", e)

    # ===== CHART 2: Reasons / Feature importance (Pie) =====
    img_reason_bg = _img("image_ReasonsChart.png")
    if img_reason_bg: canvas.create_image(880.0, 380.0, image=img_reason_bg)
    canvas.create_text(740.0, 216.0, anchor="nw", text="Reasons Chart",
                       fill="#000000", font=("Young Serif", 20))
    chart_frame_2 = Frame(canvas, bg="#FFFFFF", width=420, height=300)
    canvas.create_window(880, 410, window=chart_frame_2, anchor="center")

    try:
        fig2 = Figure(figsize=(3.0, 2.8), dpi=100, facecolor="#FFFFFF")
        ax2 = fig2.add_subplot(111)

        wedges, labels, vals = _plot_reasons_pie(ax2, data["feature_importance"])
        labels = list(labels)
        vals = [float(v) for v in vals]
        print(f"[DEBUG] pie -> wedges={len(wedges)}, labels={len(labels)}, vals={len(vals)}")

        fig2.tight_layout(pad=1.0, rect=[0.04, 0.04, 0.96, 0.96])
        fc2 = FigureCanvasTkAgg(fig2, master=chart_frame_2)
        fc2.draw()
        w2 = fc2.get_tk_widget()
        w2.pack(fill="both", expand=True)

        if not hasattr(chart_frame_2, "_mpl_refs"):
            chart_frame_2._mpl_refs = []
        chart_frame_2._mpl_refs.extend([fc2, fig2, ax2, wedges, labels, vals])

        for i, w in enumerate(wedges):
            try:
                w.set_picker(True)
                w.set_gid(i)
            except Exception:
                pass

        try:
            import mplcursors
            print("[DEBUG] mplcursors available for pie")
            cursor = mplcursors.cursor(wedges, hover=True)
            chart_frame_2._mpl_refs.append(cursor)

            @cursor.connect("add")
            def _on_add(sel):
                try:
                    idx = wedges.index(sel.artist)
                except Exception:
                    idx = getattr(sel, "index", 0) or 0
                if idx < 0 or idx >= len(labels):
                    return
                import math
                def _wedge_centroid(wedge, frac=0.75):
                    theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
                    cx, cy = wedge.center
                    r = getattr(wedge, "r", 1.0) * frac
                    return (cx + r * math.cos(theta), cy + r * math.sin(theta))
                x, y = _wedge_centroid(wedges[idx], 0.75)
                x_px, y_px = ax2.transData.transform((x, y))
                fig_w, fig_h = fig2.canvas.get_width_height()
                M = 80
                dx, dy = 10, 10
                if x_px > fig_w - M: dx = -60
                elif x_px < M:       dx = 10
                if y_px > fig_h - M: dy = -30
                elif y_px < M:       dy = 20
                ha = "left" if dx >= 0 else "right"
                va = "bottom" if dy >= 0 else "top"
                sel.annotation.set_text(f"{labels[idx]}: {vals[idx] * 100:.1f}%")
                sel.annotation.xy = (x, y)
                sel.annotation.set_position((dx, dy))
                sel.annotation.set_ha(ha); sel.annotation.set_va(va)
                sel.annotation.get_bbox_patch().set(fc="white", ec="#644E94", alpha=0.95)
                sel.annotation.set_fontsize(9)
        except Exception as e:
            print("[DEBUG] mplcursors not available:", e)

        import math
        annot = ax2.annotate(
            "", xy=(0, 0), xycoords='data',
            xytext=(0, 0), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="white", ec="#644E94", alpha=0.95),
            fontsize=9, visible=False
        )
        chart_frame_2._mpl_refs.append(annot)

        def _wedge_centroid(wedge, frac=0.75):
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            cx, cy = wedge.center
            r = getattr(wedge, "r", 1.0) * frac
            return (cx + r * math.cos(theta), cy + r * math.sin(theta))

        def _compute_smart_offset(x_data, y_data):
            x_px, y_px = ax2.transData.transform((x_data, y_data))
            fig_w, fig_h = fig2.canvas.get_width_height()
            M = 80
            dx, dy = 10, 10
            if x_px > fig_w - M: dx = -60
            elif x_px < M:       dx = 10
            if y_px > fig_h - M: dy = -30
            elif y_px < M:       dy = 20
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            return dx, dy, ha, va

        def _safe_show(i):
            if i is None or i < 0 or i >= len(labels):
                return
            x, y = _wedge_centroid(wedges[i], 0.75)
            dx, dy, ha, va = _compute_smart_offset(x, y)
            annot.xy = (x, y)
            annot.set_position((dx, dy))
            annot.set_ha(ha); annot.set_va(va)
            annot.set_text(f"{labels[i]}: {vals[i] * 100:.1f}%")
            if not annot.get_visible():
                annot.set_visible(True)
            fig2.canvas.draw_idle()

        def _hide():
            if annot.get_visible():
                annot.set_visible(False)
                fig2.canvas.draw_idle()

        def _on_pick(event):
            idx = wedges.index(event.artist) if event.artist in wedges else None
            _safe_show(idx)

        def _on_move(event):
            if event.inaxes is not ax2:
                _hide(); return
            idx = None
            for k, w in enumerate(wedges):
                if w.contains(event)[0]:
                    idx = k; break
            if idx is None: _hide()
            else: _safe_show(idx)

        cid1 = fig2.canvas.mpl_connect("pick_event", _on_pick)
        cid2 = fig2.canvas.mpl_connect("motion_notify_event", _on_move)
        chart_frame_2._mpl_refs.extend([cid1, cid2])

        try:
            w2.focus_set()
            w2.bind("<Enter>", lambda e: w2.focus_set())
        except Exception:
            pass

        print("✓ Reasons chart wired for hover")
    except Exception as e:
        print("✗ Chart2 failed:", e)

    # ===== EVAL TABLE =====
    img_table_bg = _img("image_Table.png")
    if img_table_bg:
        canvas.create_image(538.0, 764.0, image=img_table_bg)

    title_id = canvas.create_text(
        69.0, 630.0, anchor="nw",
        text="Model Evaluation Metrics Table",
        fill="#000000", font=("Young Serif", 20)
    )

    TABLE_W_EVAL = 940
    TABLE_H_EVAL = 230
    title_bottom = canvas.bbox(title_id)[3]
    EVAL_Y = title_bottom + 24

    table_frame = Frame(canvas, bg="#FFFFFF")
    canvas.create_window(
        538, EVAL_Y,
        window=table_frame,
        anchor="n",
        width=TABLE_W_EVAL,
        height=TABLE_H_EVAL
    )

    eval_df = data.get("eval_metrics")
    if isinstance(eval_df, pd.DataFrame) and not eval_df.empty:
        try:
            import tkinter.font as tkfont
            header_bg = "#644E94"; header_fg = "#FFFFFF"
            row_bg_1 = "#F5F5F5"; row_bg_2 = "#FFFFFF"; text_color = "#2E2E2E"

            table_container = Frame(table_frame, bg="#FFFFFF")
            table_container.pack(fill="both", expand=True, padx=14, pady=10)

            header_font = tkfont.Font(family="Young Serif", size=12, weight="bold")
            cell_font = tkfont.Font(family="Crimson Pro", size=11)

            if {"AUC_mean", "F1_mean"}.issubset(set(eval_df.columns)):
                columns = ["Model", "AUC_mean", "F1_mean", "AUC_std", "F1_std", "Brier_mean"]
            else:
                columns = ["Model", "AUC", "F1", "Precision", "Recall", "Accuracy"]

            for c in range(len(columns)):
                table_container.grid_columnconfigure(c, weight=1, uniform="metrics")

            header_row = Frame(table_container, bg=header_bg)
            header_row.grid(row=0, column=0, columnspan=len(columns), sticky="ew", pady=(0, 2))
            for c in range(len(columns)):
                header_row.grid_columnconfigure(c, weight=1, uniform="metrics")
            for j, title in enumerate(columns):
                Label(header_row, text=title, bg=header_bg, fg=header_fg,
                      font=header_font, width=15 if j == 0 else 12,
                      anchor="center", padx=8, pady=8).grid(row=0, column=j, sticky="ew", padx=1)

            for i, (_, r) in enumerate(eval_df.iterrows()):
                row_bg = row_bg_1 if i % 2 == 0 else row_bg_2
                for j, col in enumerate(columns):
                    val = r.get(col, "")
                    if col != "Model":
                        try:
                            val = f"{float(val):.3f}"
                        except:
                            val = str(val)
                    fg = "#644E94" if i == 0 else text_color
                    font = tkfont.Font(family="Crimson Pro", size=11, weight="bold") if i == 0 else cell_font
                    Label(table_container, text=str(val), bg=row_bg, fg=fg, font=font,
                          width=15 if j == 0 else 12, anchor="center",
                          padx=8, pady=6).grid(row=i + 1, column=j, sticky="ew", padx=1, pady=1)
            print("✓ Model evaluation table created")
        except Exception as e:
            print("✗ Eval table failed:", e)
            Label(table_frame, text="No model evaluation data available",
                  bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)
    else:
        Label(table_frame, text="No model evaluation data available",
              bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)

    # ===== DETAIL ANALYSIS =====
    canvas.create_text(48.0, 960.0, anchor="nw", text="Detail Analysis",
                       fill="#706093", font=("Young Serif", 32))

    # ===== FEATURE IMPORTANCE =====
    FI_OFFSET_Y = 0
    FI_W, FI_H = 340, 100
    FI_FIG_W_H = (2.8, 1.9)

    # nền card
    fi_bg_id = None
    _img_feat_bg = _img("image_FeatureImportant.png")
    if _img_feat_bg:
        fi_bg_id = canvas.create_image(280.0, 1275.0 + FI_OFFSET_Y, image=_img_feat_bg)

    # khung nhúng chart
    fi_frame = Frame(canvas, bg='white', width=FI_W, height=FI_H)
    # --- thêm tiêu đề bên trong ô trắng ---
    Label(
        fi_frame,
        text="Feature Importance",
        bg="white",
        fg="#000000",
        font=("Young Serif", 20)
    ).pack(side="top", anchor="n", pady=(6, 0))

    fi_win = canvas.create_window(280, 1280 + FI_OFFSET_Y, window=fi_frame, anchor="center")
    try:
        fi_frame.configure(height=FI_H)
        x, y = canvas.coords(fi_win)
        canvas.coords(fi_win, x, y - 6)  # nhích lên 6px
    except Exception:
        pass

    # vẽ biểu đồ
    fig3 = Figure(figsize=(4.2,3), dpi=100, facecolor="#FFFFFF")
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Feature Importance", fontsize=13, fontweight="bold", color="#4A3F6A", pad=10)
    _plot_feature_importance(ax3, data["feature_importance"], top_n=10)
    ax3.set_xlabel("")  # ẩn nhãn trục X

    # lưu tên feature rồi ẩn trục y
    feature_names = [t.get_text() for t in ax3.get_yticklabels()]
    ax3.set_ylabel("")
    ax3.set_yticklabels([])
    ax3.set_yticks([])
    ax3.tick_params(axis='y', length=0)
    ax3.spines['left'].set_visible(False)
    ax3.margins(y=0.18)
    ax3.set_ylim(ax3.get_ylim())
    fig3.tight_layout(pad=0.25)

    import matplotlib as mpl
    PALETTE = ['#644E94', '#BB95BB', '#A08CB1', '#AF96B9', '#BEA1C1', '#C7AAC6', '#E3D4E0', '#FAE4F2']
    bars = [p for p in ax3.patches if isinstance(p, mpl.patches.Rectangle) and p.get_width() > 0]
    bars = sorted(bars, key=lambda b: b.get_y())
    for i, bar in enumerate(bars):
        c = PALETTE[i % len(PALETTE)]
        bar.set_facecolor(c)
        bar.set_edgecolor(c)

    # --- HOVER: chỉ trên các thanh bar ---
    try:
        import mplcursors
        names_by_bar = {}
        for i, b in enumerate(bars):
            if i < len(feature_names) and feature_names[i]:
                names_by_bar[id(b)] = feature_names[i]
            else:
                names_by_bar[id(b)] = f"Feature {i + 1}"

        cursor_fi = mplcursors.cursor(bars, hover=True)

        @cursor_fi.connect("add")
        def _on_add_fi(sel):
            b = sel.artist
            try:
                val = float(getattr(b, "get_width", lambda: 0)())
            except Exception:
                val = 0.0
            name = names_by_bar.get(id(b), "Feature")
            txt = f"{name}: {val * 100:.1f}%" if 0.0 <= val <= 1.0 else f"{name}: {val:.3f}"
            sel.annotation.set_text(txt)
            sel.annotation.get_bbox_patch().set(fc="white", ec="#644E94", alpha=0.95)
            sel.annotation.set_fontsize(9)
    except Exception as e:
        print("[DEBUG] mplcursors for FI failed:", e)

    fc3 = FigureCanvasTkAgg(fig3, master=fi_frame)
    fc3.draw()
    fc3.get_tk_widget().pack(fill="both", expand=True)

    # ===== SHAP =====

    SHAP_W, SHAP_H = 460, 340
    DPI = 100

    shap_bg_id = None
    _img_shap_bg = _img("image_SHAP.png")
    if _img_shap_bg:
        shap_bg_id = canvas.create_image(807.0, 1275.0, image=_img_shap_bg)

    shap_frame = tk.Frame(canvas, bg='white', width=SHAP_W, height=SHAP_H)
    # --- thêm tiêu đề bên trong ô trắng ---
    Label(
        shap_frame,
        text="SHAP Summary Plot",
        bg="white",
        fg="#000000",
        font=("Young Serif", 20)
    ).pack(side="top", anchor="n", pady=(6, 0))

    shap_win = canvas.create_window(807, 1280, window=shap_frame, anchor="center")
    try:
        shap_frame.configure(height=SHAP_H)
        x, y = canvas.coords(shap_win)
        canvas.coords(shap_win, x, y - 6)
    except Exception:
        pass
    try:
        shap_frame.pack_propagate(False)
        shap_frame.grid_propagate(False)
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        import shap
        from matplotlib.collections import PathCollection
        from matplotlib.colors import ListedColormap

        SHAP_PALETTE = ['#644E94', '#BB95BB', '#A08CB1', '#AF96B9',
                        '#BEA1C1', '#C7AAC6', '#E3D4E0', '#FAE4F2']
        SHAP_CMAP = ListedColormap(SHAP_PALETTE, name="chuLiBi_shap")

        def _force_cmap_on_figure(fig, cmap):
            for ax in fig.axes:
                for coll in list(ax.collections) + list(ax.images):
                    try:
                        coll.set_cmap(cmap)
                    except Exception:
                        pass

        try:
            if hasattr(shap, "plots") and hasattr(shap.plots, "colors"):
                shap.plots.colors.red_blue = SHAP_CMAP
                shap.plots.colors.blue_red = SHAP_CMAP
                if hasattr(shap.plots.colors, "red_blue_transparent"):
                    shap.plots.colors.red_blue_transparent = SHAP_CMAP
        except Exception:
            pass

        bundle = data.get("bundle")
        X = data.get("X")

        fig_shap = None
        used_beeswarm = False

        try:
            fig_try = Figure(figsize=(SHAP_W / float(DPI), SHAP_H / float(DPI)), dpi=DPI, facecolor='white')
            _plot_shap_summary(fig_try, bundle, X)
            if fig_try.axes and (fig_try.axes[0].has_data() or fig_try.axes[0].collections or fig_try.axes[0].lines):
                fig_shap = fig_try
                try:
                    _force_cmap_on_figure(fig_shap, SHAP_CMAP)
                except Exception as _e:
                    print("[DEBUG] skip force-cmap (A):", _e)
        except Exception as e:
            print("[DEBUG] cannot use _plot_shap_summary():", e)

        if fig_shap is None and isinstance(bundle, dict) and X is not None and (
            ("shap_values" in bundle) or ("shap" in bundle)
        ):
            plt.close('all')
            shap_values = bundle.get("shap_values", bundle.get("shap"))
            try:
                shap.summary_plot(shap_values, X, show=False, plot_type="dot",
                                  max_display=12, color=SHAP_CMAP)
            except Exception:
                shap.plots.beeswarm(shap_values, max_display=12, show=False, color=SHAP_CMAP)
            fig_shap = plt.gcf()
            used_beeswarm = True

            try:
                _force_cmap_on_figure(fig_shap, SHAP_CMAP)
            except Exception as _e:
                print("[DEBUG] skip force-cmap (B):", _e)

            ax = fig_shap.axes[0]
            feature_names = [t.get_text() for t in ax.get_yticklabels()]
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.tick_params(axis='y', length=0)
            try:
                ax.spines['left'].set_visible(False)
            except Exception:
                pass

            try:
                import mplcursors
                cols = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets()) > 0]
                useful_cols = cols[-len(feature_names):] if len(cols) >= len(feature_names) and feature_names else cols
                col2feat = {}
                for i, c in enumerate(useful_cols):
                    name = feature_names[i] if i < len(feature_names) else f"Feature {i + 1}"
                    col2feat[id(c)] = name

                cursor = mplcursors.cursor(useful_cols, hover=True)

                @cursor.connect("add")
                def _on_add(sel):
                    name = col2feat.get(id(sel.artist), "Feature")
                    try:
                        x_val = float(sel.target[0])
                    except Exception:
                        x_val = None
                    txt = f"{name}: SHAP={x_val:.3f}" if x_val is not None else name
                    sel.annotation.set_text(txt)
                    sel.annotation.get_bbox_patch().set(fc="white", ec="#644E94", alpha=0.95)
                    sel.annotation.set_fontsize(9)
            except Exception as e:
                print("[DEBUG] mplcursors for SHAP failed:", e)

        if fig_shap is None:
            fig_shap = Figure(figsize=(SHAP_W / float(DPI), SHAP_H / float(DPI)), dpi=DPI, facecolor='white')
            ax_fb = fig_shap.add_subplot(111)
            ax_fb.text(0.5, 0.5, "No SHAP data available", ha="center", va="center",
                       fontsize=11, color="#7A5FA5")
            ax_fb.set_axis_off()

        try:
            fig_shap.set_size_inches(SHAP_W / float(DPI), SHAP_H / float(DPI), forward=True)
        except Exception:
            pass

        # --- Title + margin chuẩn cho SHAP ---
        SHAP_TITLE = "SHAP Summary Plot"
        try:
            fig_shap.suptitle(SHAP_TITLE, fontsize=12, y=0.98)
        except Exception:
            pass
        try:
            top_margin = 0.88
            bottom_margin = 0.22 if used_beeswarm else 0.10
            fig_shap.subplots_adjust(left=0.06, right=0.98, top=top_margin, bottom=bottom_margin)
        except Exception:
            pass
        try:
            for ax in fig_shap.axes:
                ax.set_xlabel("")  # ẩn nhãn trục X
        except Exception:
            pass

        fc4 = FigureCanvasTkAgg(fig_shap, master=shap_frame)
        fc4.draw()
        w4 = fc4.get_tk_widget()
        w4.configure(width=SHAP_W, height=SHAP_H)
        w4.pack(fill="both", expand=False)

        if not hasattr(shap_frame, "_mpl_refs"):
            shap_frame._mpl_refs = []
        shap_frame._mpl_refs.extend([fc4, fig_shap])
    except Exception as e:
        print("✗ SHAP section failed:", e)
        tk.Label(shap_frame, text="Install package 'shap' để xem biểu đồ SHAP",
                 bg='white', fg='#7A5FA5', font=("Crimson Pro", 12)).pack(expand=True)

    # ======= ĐẨY BẢNG XUỐNG — chống dính khung trắng =======
    canvas.update_idletasks()

    def _bottom_of(win_id, fallback_y, fallback_h):
        bb = canvas.bbox(win_id)
        if bb:
            return bb[3]
        return fallback_y + fallback_h / 2

    fi_bottom   = _bottom_of(fi_win,   1280 + FI_OFFSET_Y, FI_H)
    shap_bottom = _bottom_of(shap_win, 1280,               SHAP_H)
    detail_bottom = max(fi_bottom, shap_bottom)

    CARD_GAP_AFTER = 180  # tăng/giảm nếu muốn
    TITLE_Y = detail_bottom + (CARD_GAP_AFTER // 2)
    TABLE_Y = detail_bottom + CARD_GAP_AFTER

    # ===== PREDICTION TABLE =====
    image_BGPredict = _img("image_Table.png")
    if image_BGPredict:
        canvas.create_image(538.0, TABLE_Y + 70,  image=image_BGPredict)
        canvas.create_image(538.0, TABLE_Y + 270, image=image_BGPredict)

    canvas.create_text(69.0, TITLE_Y, anchor="nw",
                       text="Churn Prediction Results",
                       fill="#000000", font=("Young Serif", 20))

    TABLE_X, TABLE_W, TABLE_H = 69, 950, 380

    table_holder = Frame(canvas, bg="#FFFFFF", highlightthickness=0, bd=0, relief="flat")
    canvas.create_window(TABLE_X, TABLE_Y, window=table_holder, anchor="nw", width=TABLE_W, height=TABLE_H)

    df_result = data.get("df_result")

    # === Filter (đặt sát bên trên bảng) ===
    table_section = Frame(table_holder, bg="#FFFFFF")
    # giảm padding trên để dính sát hơn
    table_section.pack(fill="x", side="top", anchor="n", padx=6, pady=(2, 0))

    filter_container = Frame(table_holder, bg="#FFFFFF", highlightthickness=0)
    filter_container.pack(fill="x", side="top", anchor="n", padx=(10, 10), pady=(6, 0))


    # callback lọc
    def on_filter_change(cluster_selection):
        # chỉ xóa phần bảng, giữ filter_container
        for w in table_holder.winfo_children():
            if w is filter_container:
                continue
            w.destroy()

        if cluster_selection == "All Clusters":
            filtered = df_result
        else:
            try:
                cluster_num = int(cluster_selection.split()[-1])
                filtered = df_result[df_result["cluster"].astype(float).astype(int) == cluster_num]
            except Exception:
                filtered = df_result

        table_canvas = build_prediction_table(table_holder, TABLE_W, TABLE_H, filtered)
        table_canvas.pack(fill="both", expand=True)

    dropdown = build_cluster_filter_dropdown(filter_container, df_result, on_filter_change)
    dropdown.pack(side="right", padx=(0, 12), pady=4)

    if isinstance(df_result, pd.DataFrame) and not df_result.empty:
        try:
            table_canvas = build_prediction_table(table_holder, TABLE_W, TABLE_H, df_result)
            table_canvas.pack(fill="both", expand=True)
            print(f"✓ Prediction table created with {len(df_result)} records)")
        except Exception as e:
            print("✗ Prediction table failed:", e)
            Label(table_holder, text="Prediction table unavailable",
                  bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)
    else:
        Label(table_holder, text="Prediction table unavailable",
              bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)

    # cập nhật scrollregion để cuộn đến tận table
    def _sync_scrollregion(_=None):
        canvas.update_idletasks()
        bbox = canvas.bbox("all") or (0, 0, width, height)
        x0, y0, x1, y1 = bbox
        y1 = max(y1, TABLE_Y + TABLE_H + 150)
        canvas.configure(scrollregion=(x0, y0, x1, y1))

    canvas.bind("<Configure>", _sync_scrollregion)
    canvas.after(100, _sync_scrollregion)

    return canvas

# =========================================================
# Preview runner
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Frame08 - Churn (Standalone ui_content)")
    root.geometry("1200x1440")
    root.configure(bg="#D4C5D2")

    outer = Frame(root, bg="#D4C5D2")
    outer.pack(fill="both", expand=True)

    canvas = build_content(outer, width=1100, height=850)
    root.mainloop()
