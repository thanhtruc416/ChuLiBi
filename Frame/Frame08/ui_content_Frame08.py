# -*- coding: utf-8 -*-
# ui_content_Frame08.py

# =========================================================
# ÉP BACKEND TKAGG TRƯỚC KHI IMPORT CÁC BACKEND KHÁC
# =========================================================
import matplotlib
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import ListedColormap

if matplotlib.get_backend().lower() != "tkagg":
    matplotlib.use("TkAgg")
print("[DEBUG] Matplotlib backend (after set):", matplotlib.get_backend())

# --- Đặt font mặc định toàn cục cho Matplotlib ---
from matplotlib import rcParams


rcParams['font.family'] = 'Crimson Pro'
print("[INFO] Default Matplotlib font:", rcParams['font.family'])


from pathlib import Path
import sys
import tkinter as tk
from tkinter import Canvas, PhotoImage, Frame, Label

# Matplotlib cho embed chart (sau khi đã set TkAgg)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# --- Đăng ký font Crimson Pro thủ công (đúng thư mục Font/Crimson_Pro/static) ---
import matplotlib.font_manager as fm
from matplotlib import rcParams
from pathlib import Path

# Xác định thư mục font (theo cấu trúc của bạn)
FONT_DIR = Path(__file__).resolve().parents[2] / "Font" / "Crimson_Pro" / "static"

loaded_fonts = 0
if FONT_DIR.exists():
    for f in FONT_DIR.glob("CrimsonPro-*.ttf"):
        try:
            fm.fontManager.addfont(str(f))
            loaded_fonts += 1
        except Exception as e:
            print(f"[WARN] Cannot load font {f.name}: {e}")

if loaded_fonts > 0:
    rcParams['font.family'] = 'Crimson Pro'
    print(f"[INFO] Loaded {loaded_fonts} Crimson Pro fonts for Matplotlib")
else:
    print("[WARN] No Crimson Pro font files found — using default font")

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
        cmb = ttk.Combobox(parent, values=["All Clusters","Cluster 0","Cluster 1","Cluster 2","Cluster 3"],state="readonly", textvariable=var, width=20)
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
    canvas = Canvas(parent, bg="#E2E2E2", width=width, height=height,
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
    canvas.create_text(110, 120.0, anchor="nw", text="Avg Churn",
                       fill="#000000", font=("Crimson Pro Bold", 25))
    avg_churn_value = f"{data.get('avg_churn', 0.0) * 100:.0f}%"
    canvas.create_text(135, 40, anchor="nw", text=avg_churn_value,
                       fill="#706093", font=("Kodchasan Regular", 40))

    img_clusters = _img("image_Clusters.png")
    if img_clusters: canvas.create_image(525.0, 101.0, image=img_clusters)
    canvas.create_text(465.0, 120.0, anchor="nw", text="Clusters",
                       fill="#000000", font=("Crimson Pro Bold", 25))
    num_clusters = str(int(data.get("num_clusters", 4)))
    canvas.create_text(500.0, 40, anchor="nw", text=num_clusters,
                       fill="#706093", font=("Kodchasan Regular", 40))

    # ----- BEST MODEL (centered on the card) -----
    cx, cy = 880.0, 100.0
    img_best = _img("image_LogisticRegression.png")
    if img_best:
        canvas.create_image(cx, cy, image=img_best)
    canvas.create_text(cx, cy - 50, anchor="n", text="Best Model",
                       fill="#000000", font=("Crimson Pro Bold", 25))

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
    if img_rate_bg:
        canvas.create_image(354.0, 380.0, image=img_rate_bg)

    canvas.create_text(
        65.0, 213.0, anchor="nw",
        text="Churn Rate by Customer Segment",
        fill="#706093", font=("Crimson Pro Bold", 25)
    )

    chart_frame_1 = Frame(canvas, bg="#FFFFFF", width=600, height=310)
    canvas.create_window(330, 415, window=chart_frame_1, anchor="center")

    if data:
        try:
            axfig = Figure(figsize=(5.7, 3.05), dpi=100, facecolor='#FFFFFF')
            ax = axfig.add_subplot(111)

            import inspect
            try:
                n_params = len(inspect.signature(_plot_churn_rate_by_segment).parameters)
            except Exception:
                n_params = 3

            if n_params >= 3:
                _plot_churn_rate_by_segment(ax, data['df_core'], data['df'])
            else:
                _plot_churn_rate_by_segment(ax, data['df_core'])

            # ===== STYLE TRỤC (giống Feature Importance) =====
            axis_color = "#7A467A"  # màu bạn yêu cầu

            # đổi màu ticks + nhãn trục
            ax.tick_params(axis='x', colors=axis_color)
            ax.tick_params(axis='y', colors=axis_color)

            # đổi màu label trục, nếu có
            if ax.get_xlabel():
                ax.xaxis.label.set_color(axis_color)
            if ax.get_ylabel():
                ax.yaxis.label.set_color(axis_color)

            # đổi màu đường trục (spines)
            for spine in ['bottom', 'left']:
                if spine in ax.spines:
                    ax.spines[spine].set_color(axis_color)

            # nhẹ nhàng đổi màu grid (nếu có grid trong hàm vẽ)
            ax.grid(True, axis='y', alpha=0, color=axis_color)

            # ===== Căn layout không bị cắt =====
            try:
                axfig.tight_layout(pad=0.8)
                axfig.subplots_adjust(left=0.2, right=0.98, bottom=0.22, top=0.90)
            except Exception:
                pass

            # ===== Embed vào Tkinter =====
            fc = FigureCanvasTkAgg(axfig, master=chart_frame_1)
            fc.draw()
            fc.get_tk_widget().pack(fill="both", expand=True)

            print("Churn rate chart created")
        except Exception as e:
            print("Chart1 failed:", e)

    # ===== CHART 2: Reasons Chart =====
    img_reason_bg = _img("image_ReasonsChart.png")
    if img_reason_bg: canvas.create_image(880.0, 380.0, image=img_reason_bg)
    canvas.create_text(785.0, 213.0, anchor="nw", text="Reasons Chart", fill="#706093", font=("Crimson Pro Bold", 25))
    chart_frame_2 = Frame(canvas, bg="#FFFFFF", width=430, height=280)
    canvas.create_window(880, 405, window=chart_frame_2, anchor="center")

    try:
        fig2 = Figure(figsize=(3.13, 2.85), dpi=100, facecolor="#FFFFFF")
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
                sel.annotation.set_ha(ha);
                sel.annotation.set_va(va)
                sel.annotation.get_bbox_patch().set(fc="white", ec="#644E94", alpha=0.95)
                sel.annotation.set_fontsize(11)

                # Quan trọng: không cho annotation bị cắt bởi khung
                sel.annotation.set_clip_on(False)
                sel.annotation.get_bbox_patch().set_clip_on(False)


        except Exception as e:
            print("[DEBUG] mplcursors not available:", e)

        import math
        annot = ax2.annotate(
            "", xy=(0, 0), xycoords='data',
            xytext=(0, 0), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="white", ec="#644E94", alpha=0.95),
            fontsize=12, visible=False
        )
        chart_frame_2._mpl_refs.append(annot)

        def _wedge_centroid(wedge, frac=0.75):
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            cx, cy = wedge.center
            r = getattr(wedge, "r", 1.0) * frac
            return (cx + r * math.cos(theta), cy + r * math.sin(theta))

        def _compute_smart_offset(x_data, y_data):
            # chuyển vị trí wedge sang pixel
            x_px, y_px = ax2.transData.transform((x_data, y_data))
            fig_w, fig_h = fig2.canvas.get_width_height()

            # offset mặc định: dịch 10px sang phải, 10px xuống
            dx, dy = 10, 10

            # ước lượng kích thước tooltip (text + viền)
            TOOLTIP_W = 120  # đủ để chứa 'pca_deal_sensitive: 14%'
            TOOLTIP_H = 40

            # 1) Nếu tooltip vượt mép phải của Figure → đẩy ngược vào trong vừa đủ
            if x_px + dx + TOOLTIP_W > fig_w:
                dx = (fig_w - TOOLTIP_W) - x_px - 10  # giữ cách mép phải 10px

            # 2) Nếu tooltip vượt mép trái sau khi đẩy → kéo lại bên trong
            if x_px + dx < 10:  # cách mép trái 10px
                dx = 10 - x_px

            # 3) Xử lý mép dưới
            if y_px + dy + TOOLTIP_H > fig_h:
                dy = -(TOOLTIP_H + 10)

            # 4) Xử lý mép trên
            if y_px + dy < 10:
                dy = 10

            # canh lề text
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

            # Quan trọng: không bị cắt bởi Figure/Axes
            annot.set_clip_on(False)
            annot.get_bbox_patch().set_clip_on(False)

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

        print("Reasons chart wired for hover")
    except Exception as e:
        print("Chart2 failed:", e)

    # ===== EVAL TABLE =====
    img_table_bg = _img("image_Table.png")
    if img_table_bg:
        canvas.create_image(538.0, 764.0, image=img_table_bg)

    title_id = canvas.create_text(
        69.0, 615.0, anchor="nw", text="Model Evaluation Metrics Table", fill="#706093", font=("Crimson Pro Bold", 25))

    TABLE_W_EVAL = 980
    TABLE_H_EVAL = 250
    title_bottom = canvas.bbox(title_id)[3]
    EVAL_Y = title_bottom + 10

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
            header_bg = "#B79AC8"; header_fg = "#FFFFFF"
            row_bg_1 = "#F7F4F7"; row_bg_2 = "#FFFFFF"; text_color = "#2E2E2E"

            table_container = Frame(table_frame, bg="#FFFFFF")
            table_container.pack(fill="both", expand=True, padx=14, pady=10)

            header_font = tkfont.Font(family="Crimson Pro Bold", size=20, weight="bold")
            cell_font = tkfont.Font(family="Crimson Pro", size=17)

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
                      anchor="center", padx=16, pady=8).grid(row=0, column=j, sticky="ew", padx=1)

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
                    font = tkfont.Font(family="Crimson Pro", size=17, weight="bold") if i == 0 else cell_font
                    Label(table_container, text=str(val), bg=row_bg, fg=fg, font=font,
                          width=15 if j == 0 else 12, anchor="center",
                          padx=16, pady=6).grid(row=i + 1, column=j, sticky="ew", padx=1, pady=1)
            print("Model evaluation table created")
        except Exception as e:
            print("Eval table failed:", e)
            Label(table_frame, text="No model evaluation data available",
                  bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)
    else:
        Label(table_frame, text="No model evaluation data available",
              bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 12)).pack(expand=True)

    # ===== DETAIL ANALYSIS =====
    canvas.create_text(48.0, 940.0, anchor="nw", text="Detail Analysis", fill="#000000", font=("Young Serif", 32))

    # ===== FEATURE IMPORTANCE =====
    FI_OFFSET_Y = 0
    FI_W, FI_H = 340, 100

    # nền card
    fi_bg_id = None
    _img_feat_bg = _img("image_FeatureImportant.png")
    if _img_feat_bg:
        fi_bg_id = canvas.create_image(280.0, 1275.0 + FI_OFFSET_Y, image=_img_feat_bg)

    # khung nhúng chart
    fi_frame = Frame(canvas, bg='white', width=FI_W, height=FI_H)
    Label(fi_frame, text="Feature Importance", bg="white", fg="#706093",
          font=("Crimson Pro Bold", 28), anchor="w", padx=10).pack(fill="x", pady=(6, 0))

    fi_win = canvas.create_window(270, 1270 + FI_OFFSET_Y, window=fi_frame, anchor="center")

    # ===== VẼ BIỂU ĐỒ (Giữ màu PALETTE + vị trí giống ban đầu) =====
    fig3 = Figure(figsize=(4.2, 4), dpi=100, facecolor="#FFFFFF")
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Feature Importance", fontsize=13, fontweight="bold", color="#4A3F6A", pad=10)

    # lấy top N (đảo để bar từ trên xuống giống giao diện cũ)
    fi_df = data["feature_importance"].head(10).iloc[::-1].reset_index(drop=True)
    vals = fi_df["Importance"].values
    names = fi_df["Feature"].values

    # palette (giữ như bạn định)
    PALETTE = ['#644E94', '#BB95BB', '#A08CB1', '#AF96B9', '#BEA1C1', '#C7AAC6', '#E3D4E0', '#FAE4F2']

    # vẽ barh với height gần kín để trông giống original
    bar_height = 0.85
    y_pos = list(range(len(vals)))
    bars = ax3.barh(y_pos, vals, height=bar_height)

    # gán màu theo PALETTE (vòng lại nếu ít/more)
    for i, b in enumerate(bars):
        c = PALETTE[i % len(PALETTE)]
        b.set_facecolor(c)
        b.set_edgecolor(c)

    # Giữ vị trí và tỉ lệ như bản gốc
    ax3.set_yticks([])  # nếu bạn muốn ẩn tick names như trước
    ax3.set_yticklabels([])
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(axis='x', colors='#7A467A')
    ax3.spines['bottom'].set_color('#7A467A')
    ax3.xaxis.label.set_color('#7A467A')  # nếu cần luôn cho đồng bộ

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ép các thiết lập layout giống trước
    ax3.margins(y=0.02)
    ax3.set_xlim(0, max(vals) * 1.02)
    ax3.set_position([0.12, 0.15, 0.82, 0.75])
    fig3.tight_layout(pad=0.1)
    ax3.set_ylim(-0.5, len(bars) - 0.5)
    fig3.subplots_adjust(top=0.97, bottom=0.07)

    # mapping bar -> name
    names_by_bar = {id(b): names[i] for i, b in enumerate(bars)}

    # nhúng vào Tkinter
    fc3 = FigureCanvasTkAgg(fig3, master=fi_frame)
    canvas3 = fc3.get_tk_widget()
    canvas3.pack(fill="both", expand=True)
    fc3.draw()

    # ===== TOOLTIP: đặt ở đầu thanh, offset giống ban đầu =====
    annot_fi = ax3.annotate(
        "", xy=(0, 0), xytext=(10, -5), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="#644E94", alpha=0.95),
        fontsize=11, visible=False
    )

    def _show_tooltip(bar):
        val = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        name = names_by_bar.get(id(bar), "Feature")

        # đặt text đầu tiên
        text = f"{name}: {val * 100:.1f}%" if 0 <= val <= 1 else f"{name}: {val:.3f}"
        annot_fi.set_text(text)
        annot_fi.xy = (val, y)

        # --- kiểm tra tràn màn hình (bên phải hoặc trái) ---
        fig_w, fig_h = fig3.canvas.get_width_height()
        x_disp, y_disp = ax3.transData.transform((val, y))  # vị trí theo pixel sau transform

        # vị trí dự kiến tooltip + offset pixel
        tooltip_x = x_disp + 10
        tooltip_y = y_disp - 5

        # nếu sát mép phải quá → kéo qua trái
        if tooltip_x + 120 > fig_w:  # 120px = ước lượng chiều rộng tooltip
            annot_fi.set_position((-120, -5))  # dịch sang trái
        # nếu qua trái quá → đẩy nhẹ vào trong
        elif tooltip_x < 10:
            annot_fi.set_position((10, -5))
        else:
            annot_fi.set_position((10, -5))

        annot_fi.set_visible(True)

    def _hide_tooltip():
        if annot_fi.get_visible():
            annot_fi.set_visible(False)

    def _hover_event(event):
        if event.inaxes != ax3:
            _hide_tooltip()
            fig3.canvas.draw_idle()
            return

        for b in bars:
            contains, _ = b.contains(event)
            if contains:
                _show_tooltip(b)
                fig3.canvas.draw_idle()
                return

        _hide_tooltip()
        fig3.canvas.draw_idle()

    fig3.canvas.mpl_connect("motion_notify_event", _hover_event)

    # ===== SHAP =====
    SHAP_W, SHAP_H = 440, 418
    DPI = 100

    # --- nền ảnh ---
    shap_bg_id = None
    _img_shap_bg = _img("image_SHAP.png")
    if _img_shap_bg:
        shap_bg_id = canvas.create_image(807.0, 1275.0, image=_img_shap_bg)

    # --- ô trắng chứa chart ---
    shap_frame = tk.Frame(canvas, bg='white', width=SHAP_W, height=SHAP_H)
    shap_win = canvas.create_window(807, 1315, window=shap_frame, anchor="center")

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

    # --- chữ SHAP (tạo sau cùng để không bị che) ---
    shap_title_id = canvas.create_text(
        620.0, 1052.0, anchor="nw", text="SHAP Summary Plot",
        fill="#706093", font=("Crimson Pro Bold", -34))
    canvas.tag_raise(shap_title_id)

    # --- xử lý và hiển thị biểu đồ SHAP ---
    try:
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

        # --- thử gọi hàm custom ---
        try:
            fig_try = Figure(figsize=(SHAP_W / float(DPI), SHAP_H / float(DPI)), dpi=DPI, facecolor='white')
            _plot_shap_summary(fig_try, bundle, X)
            if fig_try.axes and (fig_try.axes[0].has_data() or fig_try.axes[0].collections or fig_try.axes[0].lines):
                fig_shap = fig_try
                _force_cmap_on_figure(fig_shap, SHAP_CMAP)
        except Exception as e:
            print("[DEBUG] cannot use _plot_shap_summary():", e)

        # --- fallback: shap.summary_plot ---
        if fig_shap is None and isinstance(bundle, dict) and X is not None and (
                ("shap_values" in bundle) or ("shap" in bundle)
        ):
            plt.close('all')
            shap_values = bundle.get("shap_values", bundle.get("shap"))
            try:
                shap.summary_plot(
                    shap_values, X,
                    show=False,
                    plot_type="dot",
                    max_display=12,
                    color=SHAP_CMAP,
                    plot_size=(4.8, 5.2)
                )
            except Exception:
                shap.plots.beeswarm(shap_values, max_display=12, show=False, color=SHAP_CMAP)

            fig_shap = plt.gcf()
            used_beeswarm = True
            _force_cmap_on_figure(fig_shap, SHAP_CMAP)

            # --- ẩn nhãn mặc định (cả trục chính + colorbar) ---
            try:
                for ax in fig_shap.axes:
                    if hasattr(ax, "get_xlabel") and "SHAP value" in ax.get_xlabel():
                        ax.set_xlabel("")
                    if hasattr(ax, "xaxis") and hasattr(ax.xaxis, "label"):
                        ax.xaxis.label.set_visible(False)
                # ẩn mọi text “SHAP value” xuất hiện trong figure
                for t in fig_shap.texts[:]:
                    if "SHAP value" in t.get_text():
                        t.set_visible(False)
            except Exception:
                pass

            # --- ẩn trục trái + định dạng ---
            try:
                ax = fig_shap.axes[0]
                feature_names = [t.get_text() for t in ax.get_yticklabels()]
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.tick_params(axis='y', length=0)
                if 'left' in ax.spines:
                    ax.spines['left'].set_visible(False)
            except Exception:
                pass

        # --- nếu không có dữ liệu SHAP ---
        if fig_shap is None:
            fig_shap = Figure(figsize=(SHAP_W / float(DPI), SHAP_H / float(DPI)), dpi=DPI, facecolor='white')
            ax_fb = fig_shap.add_subplot(111)
            ax_fb.text(0.5, 0.5, "No SHAP data available", ha="center", va="center",
                       fontsize=11, color="#7A5FA5")
            ax_fb.set_axis_off()

        # --- căn lề và chỉnh style trục ---
        try:
            fig_shap.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
            for ax in fig_shap.axes:
                ax.tick_params(axis='x', colors='#7A467A', labelsize=13)
                ax.tick_params(axis='y', colors='#7A467A', labelsize=13)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname("Crimson Pro")
                    label.set_color("#7A467A")
                for spine in ax.spines.values():
                    spine.set_color("#7A467A")
                    spine.set_linewidth(1)
        except Exception as e:
            print("[DEBUG] SHAP axis style fail:", e)

        # --- hiển thị lên tkinter ---
        fc4 = FigureCanvasTkAgg(fig_shap, master=shap_frame)
        w4 = fc4.get_tk_widget()
        w4.configure(width=SHAP_W, height=SHAP_H)
        w4.pack(fill="both", expand=False)

        # --- kích hoạt tooltip SHAP (final working version) ---
        try:
            import mplcursors

            w4.focus_set()  # BẮT BUỘC: đảm bảo canvas nhận focus
            ax = fig_shap.axes[0]
            cols = [c for c in ax.collections if hasattr(c, "get_offsets") and len(c.get_offsets()) > 0]

            print(f"[DEBUG] Found {len(cols)} PathCollections for SHAP tooltip")

            if cols:
                cursor = mplcursors.cursor(cols, hover=True)

                @cursor.connect("add")
                def _on_add(sel):
                    try:
                        x_val = float(sel.target[0])
                        y_val = float(sel.target[1])
                    except Exception:
                        x_val, y_val = None, None

                    # Lấy tên feature nếu có
                    try:
                        idx = sel.index
                        feat = X.columns[idx % len(X.columns)] if hasattr(X, "columns") else f"Feature {idx + 1}"
                    except Exception:
                        feat = "Feature"

                    txt = f"{feat}\nSHAP={x_val:.3f}" if x_val is not None else feat
                    sel.annotation.set_text(txt)
                    sel.annotation.get_bbox_patch().set(fc="white", ec="#644E94", alpha=0.9)
                    sel.annotation.set_fontsize(11)
                    sel.annotation.get_bbox_patch().set_boxstyle("round,pad=0.3")
                    sel.annotation.set_visible(True)

                # Giữ reference (ngăn bị GC)
                fig_shap._cursor_ref = cursor
                shap_frame._cursor_ref = cursor

                # --- ép update thủ công ---
                def _motion(event):
                    try:
                        cursor._update()
                    except Exception:
                        pass

                fc4.mpl_connect("motion_notify_event", _motion)

                # --- đảm bảo event loop Tkinter chạy ---
                def _refresh_hover():
                    try:
                        fc4.draw_idle()
                    except Exception:
                        pass
                    shap_frame.after(300, _refresh_hover)

                _refresh_hover()

                print("[INFO] Tooltip SHAP activated and listening for motion events.")
            else:
                print("[WARN] Không tìm thấy điểm dữ liệu SHAP để bật tooltip.")

            # kiểm tra event mouse
            def _enter(e):
                print("[DEBUG] mouse entered SHAP canvas")

            fc4.mpl_connect("figure_enter_event", _enter)

        except Exception as e:
            print("[DEBUG] Tooltip SHAP init failed:", e)

        fc4.draw()

        # --- giữ tham chiếu matplotlib ---
        if not hasattr(shap_frame, "_mpl_refs"):
            shap_frame._mpl_refs = []
        shap_frame._mpl_refs.extend([fc4, fig_shap])

    except Exception as e:
        print("SHAP section failed:", e)
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

    CARD_GAP_AFTER = 150  # tăng/giảm nếu muốn
    TITLE_Y = detail_bottom + (CARD_GAP_AFTER // 2)
    TABLE_Y = detail_bottom + CARD_GAP_AFTER

    # ===== PREDICTION TABLE =====
    image_BGPredict = _img("image_Table.png")
    if image_BGPredict:
        canvas.create_image(538.0, TABLE_Y + 70,  image=image_BGPredict)
        canvas.create_image(538.0, TABLE_Y + 270, image=image_BGPredict)

    canvas.create_text(69.0, TITLE_Y, anchor="nw",
                       text="Churn Prediction Results",
                       fill="#706093", font=("Crimson Pro Bold", 30))

    TABLE_X, TABLE_W, TABLE_H = 69, 950, 400

    table_holder = Frame(canvas, bg="#FFFFFF", highlightthickness=0, bd=0, relief="flat")
    canvas.create_window(TABLE_X, TABLE_Y, window=table_holder, anchor="nw", width=TABLE_W, height=TABLE_H)

    df_result = data.get("df_result")

    # === Filter (đặt sát bên trên bảng) ===
    table_section = Frame(table_holder, bg="#FFFFFF")
    # giảm padding trên để dính sát hơn
    table_section.pack(fill="x", side="top", anchor="n", padx=6, pady=(0, 0))

    filter_container = Frame(table_holder, bg="#FFFFFF", highlightthickness=0)
    filter_container.pack(fill="x", side="top", anchor="n", padx=(10, 10), pady=(0,0))

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
                cluster_num = int(cluster_selection.split()[-1]) - 1
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
            print(f"Prediction table created with {len(df_result)} records)")
        except Exception as e:
            print("Prediction table failed:", e)
            Label(table_holder, text="Prediction table unavailable",
                  bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 14)).pack(expand=True)
    else:
        Label(table_holder, text="Prediction table unavailable",
              bg="#FFFFFF", fg="#999999", font=("Crimson Pro", 14)).pack(expand=True)

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
