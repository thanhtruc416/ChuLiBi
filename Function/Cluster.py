# Function/segmentation.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ======== Cấu hình nhóm PCA & biến dùng để phân cụm ========
LIKERT_GROUPS = {
    "pca_convenience": ["Ease and convenient_encoded", "Self Cooking_encoded", "Health Concern_encoded"],
    "pca_service_issue": ["Poor Hygiene_encoded", "Bad past experience_encoded", "Late Delivery_encoded"],
    "pca_deal_sensitive": ["More Offers and Discount_encoded", "Influence of rating_encoded"],
}

CLUSTER_FEATURES = [
    "Age","Family size","Restaurant Rating","Delivery Rating",
    "No. of orders placed","Delivery Time","Order Value",
    "pca_convenience","pca_service_issue","pca_deal_sensitive",
]

# ======== Data prep ========
def load_scaled_dataset(csv_path: Path, id_col: str = "CustomerID") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    ids = df[id_col] if id_col in df.columns else pd.Series(range(len(df)))
    df = df.drop(columns=[id_col], errors="ignore")
    return df, ids

def attach_group_pca(df_scaled: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    df = df_scaled.copy()
    for new_col, cols in LIKERT_GROUPS.items():
        miss = [c for c in cols if c not in df.columns]
        if miss:
            continue
        comp = PCA(n_components=1, random_state=random_state).fit_transform(df[cols])
        df[new_col] = comp
    return df

def select_X(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    return df[cols].values

# ======== KMeans ========
def kmeans_labels(X: np.ndarray, k: int, random_state: int = 42, n_init: int = 10) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    return km.fit_predict(X)

def evaluate_k_range(X: np.ndarray, k_min: int = 2, k_max: int = 11) -> tuple[list[int], list[float], list[float]]:
    Ks, inertias, sils = [], [], []
    for k in range(k_min, k_max):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        Ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lab))
    return Ks, inertias, sils

def counts_by_cluster(labels: Iterable[int], k: Optional[int] = None) -> list[int]:
    s = pd.Series(labels).value_counts().sort_index()
    if k is None:
        k = int(s.index.max()) + 1
    return [int(s.get(i, 0)) for i in range(k)]

# --- Elbow & Silhouette (gọn, không đè tiêu đề UI) ---
def figure_elbow_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 11) -> Figure:
    Ks, inertias, sils = evaluate_k_range(X, k_min, k_max)

    # figure nhỏ hơn + bật constrained layout
    fig = Figure(figsize=(3.8, 2.0), dpi=120)
    fig.set_constrained_layout(True)
    ax  = fig.add_subplot(111)
    ax2 = ax.twinx()

    # 2 đường
    l1, = ax.plot(Ks, inertias, "o-", linewidth=1.6, markersize=3.5, color="#644E94", label="Inertia (WCSS)")
    l2, = ax2.plot(Ks, sils,     "s-", linewidth=1.6, markersize=3.5, color="#BB95BB", label="Silhouette")

    # khung & lưới
    for s in ax.spines.values():  s.set_color("#7A5FA5")
    for s in ax2.spines.values(): s.set_color("#7A5FA5")
    ax.grid(True, alpha=0.18, color="#7A5FA5", linewidth=0.6)

    # tick & nhãn nhỏ lại
    ax.set_xticks(Ks)                           # hiển thị đúng các k
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax2.tick_params(axis="y",    labelsize=7, pad=1)
    ax.set_xlabel("k", fontsize=8, labelpad=2)
    ax.set_ylabel("Inertia",     fontsize=8, color="#644E94")
    ax2.set_ylabel("Silhouette", fontsize=8, color="#BB95BB")

    # legend trong biểu đồ, có khung
    leg = ax.legend([l1, l2], ["Inertia (WCSS)", "Silhouette"],
                    loc="upper right", bbox_to_anchor=(0.98, 0.98),
                    fontsize=7, frameon=True, fancybox=True, framealpha=0.9,
                    borderpad=0.3, labelspacing=0.25, handlelength=1.4)
    leg.get_frame().set_edgecolor("#7A5FA5")
    leg.get_frame().set_linewidth(0.7)
    leg.get_frame().set_facecolor("white")

    # ép viền để KHÔNG cắt số khi render rất nhỏ
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.22, top=0.95)

    return fig


# --- Pie phân bố cụm (màu & chữ hài hoà) ---
def figure_cluster_distribution(labels: Iterable[int], scale: float = 1.2) -> Figure:
    s = pd.Series(labels).value_counts().sort_index()

    # Kích thước gốc ~400x215 px → nhân theo scale
    base_w, base_h = 3.35, 1.85
    fig = Figure(figsize=(base_w * scale, base_h * scale), dpi=120)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)

    colors = ["#FAE4F2", "#C6ABC5", "#644E94", "#ffcc99"][:len(s)]

    # Font theo scale
    label_fs   = max(8, int(9 * scale))   # chữ trong lát
    legend_fs  = max(8, int(8 * scale))   # chữ legend
    edge_lw    = 0.6 * (0.9 + 0.25 * (scale - 1))  # viền lát tăng nhẹ theo scale

    wedges, _texts, autotexts = ax.pie(
        s.values,
        labels=None,
        autopct=lambda pct: "{:.1f}%\n({:.0f})".format(pct, pct/100.0*s.sum()),
        startangle=140,
        colors=colors,
        pctdistance=0.62,  # giữ gọn trong lát
        textprops={"fontsize": label_fs, "fontweight": "bold", "color": "black"},
        wedgeprops={"linewidth": edge_lw, "edgecolor": "white"},
        radius=1.0
    )
    ax.axis("equal")

    # Legend dưới bánh, không tràn
    ax.legend(
        [f"Cluster {i+1}" for i in s.index],
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=len(s), fontsize=legend_fs, frameon=False
    )

    # Mép figure — có scale vẫn an toàn
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)
    return fig


# --- PCA scatter (viền, lưới thanh mảnh) ---
def figure_pca_scatter(X: np.ndarray, labels: Iterable[int]) -> Figure:
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    # figure gọn + để tự chỉnh lề thủ công (không dùng constrained_layout)
    fig = Figure(figsize=(5.0, 2.8), dpi=120)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)

    palette = ["#644E94", "#C6ABC5", "#9282AA", "#FAE4F2"][:len(np.unique(labels))]
    sns.scatterplot(
        x=X2[:, 0], y=X2[:, 1], hue=list(labels),
        palette=palette, s=26, edgecolor="#3a2a68", linewidth=0.35, ax=ax
    )

    # style trục
    for spine in ax.spines.values():
        spine.set_color("#7A5FA5")
    ax.grid(True, alpha=0.18, color="#7A5FA5", linewidth=0.7)

    # 👉 Nhãn trục + khoảng cách để không bị cắt
    ax.set_xlabel("PC1", fontsize=9, labelpad=6)
    ax.set_ylabel("PC2", fontsize=9, labelpad=8)

    # legend gọn, tránh đè chữ trục
    ax.legend(
        title="Cluster", fontsize=8, title_fontsize=9,
        loc="upper right", bbox_to_anchor=(0.98, 0.98),
        frameon=False
    )

    # 👉 Nới lề để không cắt nhãn/tick (đặc biệt là "PC2")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.96)
    return fig


