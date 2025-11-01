# Function/Frame07_Cluster.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================= PATH =================
# File này ở: Project/Function/Frame07_Cluster.py
project_root = Path(__file__).resolve().parents[1]
DATA_DIR = project_root / "Dataset" / "Output"
DATA_DIR = Path(os.getenv("DATA_DIR", str(DATA_DIR)))  # cho phép override

NEEDED = ["df_scaled_model.csv", "df_raw_dashboard.csv"]
missing = [f for f in NEEDED if not (DATA_DIR / f).exists()]
if missing:
    raise FileNotFoundError(
        f"Không tìm thấy {missing} trong {DATA_DIR}. "
        f"Hãy kiểm tra đường dẫn hoặc đặt biến môi trường DATA_DIR."
    )

# ========== CẤU HÌNH ==========
LIKERT_GROUPS = {
    "pca_convenience": ["Ease and convenient_encoded", "Self Cooking_encoded", "Health Concern_encoded"],
    "pca_service_issue": ["Poor Hygiene_encoded", "Bad past experience_encoded", "Late Delivery_encoded"],
    "pca_deal_sensitive": ["More Offers and Discount_encoded", "Influence of rating_encoded"],
}

CLUSTER_FEATURES = [
    "Age", "Family size", "Restaurant Rating", "Delivery Rating",
    "No. of orders placed", "Delivery Time", "Order Value",
    "pca_convenience", "pca_service_issue", "pca_deal_sensitive",
]

# ========== HELPERS ==========
def _ensure_X_array(X):
    """Bóc (X, use_cols) nếu có và ép về ndarray float"""
    if isinstance(X, (tuple, list)) and len(X) >= 1 and hasattr(X[0], "shape"):
        X = X[0]
    return np.asarray(X, dtype=float)

def load_scaled_dataset(csv_path: Path, id_col: str = "CustomerID") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    ids = df[id_col] if id_col in df.columns else pd.Series(range(len(df)))
    df = df.drop(columns=[id_col], errors="ignore")
    return df, ids

def attach_group_pca(df_scaled: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    df = df_scaled.copy()
    for new_col, cols in LIKERT_GROUPS.items():
        miss = [c for c in cols if c not in df.columns]
        if miss:
            # thiếu cột thì bỏ qua để không vỡ pipeline
            continue
        comp = PCA(n_components=1, random_state=random_state).fit_transform(df[cols])
        df[new_col] = comp
    return df

def select_X(df: pd.DataFrame, features: Optional[List[str]] = None) -> np.ndarray:
    """Trả về ndarray 2D dùng cho KMeans (UI không cần sửa)."""
    cols_src = features if features is not None else CLUSTER_FEATURES
    use_cols = [c for c in cols_src if c in df.columns]
    if len(use_cols) < 2:
        raise RuntimeError(f"Quá ít biến để phân cụm: {len(use_cols)}")
    return df[use_cols].to_numpy(dtype=float, copy=False)

# ========== KMEANS / EVAL ==========
def kmeans_labels(X, k: int, random_state: int = 42, n_init: int = 20) -> np.ndarray:
    X = _ensure_X_array(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    return km.fit_predict(X)

def evaluate_k_range(X, k_min: int = 2, k_max: int = 11) -> Tuple[List[int], List[float], List[float]]:
    X = _ensure_X_array(X)
    Ks, inertias, sils = [], [], []
    for k in range(k_min, k_max):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        lab = km.fit_predict(X)
        Ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lab))
    return Ks, inertias, sils

def counts_by_cluster(labels: Iterable[int], k: Optional[int] = None) -> List[int]:
    s = pd.Series(labels).value_counts().sort_index()
    if k is None:
        k = int(s.index.max()) + 1
    return [int(s.get(i, 0)) for i in range(k)]

# ========== FIGURES ==========
def figure_elbow_silhouette(X, k_min: int = 2, k_max: int = 11) -> Figure:
    X = _ensure_X_array(X)
    Ks, inertias, sils = evaluate_k_range(X, k_min, k_max)

    fig = Figure(figsize=(3.8, 2.0), dpi=120)
    # fig.set_constrained_layout(True)   # <-- xoá dòng này
    ...
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.22, top=0.95)  # giữ dòng này
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1, = ax.plot(Ks, inertias, "o-", linewidth=1.6, markersize=3.5, color="#644E94", label="Inertia (WCSS)")
    l2, = ax2.plot(Ks, sils,     "s-", linewidth=1.6, markersize=3.5, color="#BB95BB", label="Silhouette")

    for s in ax.spines.values():  s.set_color("#7A5FA5")
    for s in ax2.spines.values(): s.set_color("#7A5FA5")
    ax.grid(True, alpha=0.18, color="#7A5FA5", linewidth=0.6)

    ax.set_xticks(Ks)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax2.tick_params(axis="y", labelsize=7, pad=1)
    ax.set_xlabel("k", fontsize=8, labelpad=2)
    ax.set_ylabel("Inertia", fontsize=8, color="#644E94")
    ax2.set_ylabel("Silhouette", fontsize=8, color="#BB95BB")

    leg = ax.legend([l1, l2], ["Inertia (WCSS)", "Silhouette"],
                    loc="upper right", bbox_to_anchor=(0.98, 0.98),
                    fontsize=7, frameon=True, fancybox=True, framealpha=0.9,
                    borderpad=0.3, labelspacing=0.25, handlelength=1.4)
    leg.get_frame().set_edgecolor("#7A5FA5")
    leg.get_frame().set_linewidth(0.7)
    leg.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.22, top=0.95)
    return fig

def figure_cluster_distribution(labels: Iterable[int], scale: float = 1.2) -> Figure:
    s = pd.Series(labels).value_counts().sort_index()

    base_w, base_h = 3.35, 1.85
    fig = Figure(figsize=(base_w * scale, base_h * scale), dpi=120)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0)

    colors = ["#FAE4F2", "#C6ABC5", "#644E94", "#ffcc99"][:len(s)]

    label_fs   = max(8, int(9 * scale))
    legend_fs  = max(8, int(8 * scale))
    edge_lw    = 0.6 * (0.9 + 0.25 * (scale - 1))

    wedges, _texts, autotexts = ax.pie(
        s.values,
        labels=None,
        autopct=lambda pct: "{:.1f}%\n({:.0f})".format(pct, pct/100.0*s.sum()),
        startangle=140,
        colors=colors,
        pctdistance=0.62,
        textprops={"fontsize": label_fs, "fontweight": "bold", "color": "black"},
        wedgeprops={"linewidth": edge_lw, "edgecolor": "white"},
        radius=1.0
    )
    ax.axis("equal")

    ax.legend(
        [f"Cluster {i+1}" for i in s.index],
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=len(s), fontsize=legend_fs, frameon=False
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)
    return fig

def figure_pca_scatter(X, labels: Iterable[int]) -> Figure:
    X = _ensure_X_array(X)
    labels = np.asarray(list(labels))
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    uniq = np.sort(np.unique(labels))
    idx_map = {lab: i for i, lab in enumerate(uniq)}
    disp = np.array([idx_map[l] + 1 for l in labels])
    hue_order = list(range(1, len(uniq) + 1))

    # 🔹 tăng chiều ngang cho legend + giữ bố cục cân
    fig = Figure(figsize=(7.5, 3.0), dpi=120)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)

    palette = ["#FAE4F2", "#C6ABC5", "#644E94", "#ffcc99"][:len(uniq)]

    sns.scatterplot(
        x=X2[:, 0], y=X2[:, 1],
        hue=disp,
        hue_order=hue_order,
        palette=palette,
        s=26, edgecolor="#3a2a68", linewidth=0.35, ax=ax
    )

    # viền & lưới
    for spine in ax.spines.values():
        spine.set_color("#7A5FA5")
    ax.grid(True, alpha=0.18, color="#7A5FA5", linewidth=0.7)

    # trục
    ax.set_xlabel("PC1", fontsize=9, labelpad=6)
    ax.set_ylabel("PC2", fontsize=9, labelpad=12)

    # 🔹 legend ra ngoài, nhưng ép figure chừa đủ khoảng trống
    font_legend = font_manager.FontProperties(family="Crimson Pro", size=8)

    leg = ax.legend(
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.06, 0.5),
        frameon=False,
        labelspacing=0.4,
        prop=font_legend,  # 👈 dùng prop thay vì font
        title_fontsize=9
    )

    # 🔹 chừa thêm khoảng trắng bên trái & phải để không đè chữ
    fig.subplots_adjust(left=0.16, right=0.82, bottom=0.18, top=0.90)

    return fig




# ========== SAVE OUTPUTS ==========
def save_outputs(
    df_cluster: pd.DataFrame,
    df_raw: pd.DataFrame,
    use_cols: List[str],
    pca_cols: Optional[List[str]] = None,
) -> dict:
    pca_cols = pca_cols or []

    # profile mean trên các biến dùng phân cụm
    cluster_profile = df_cluster.groupby("cluster")[use_cols].mean().round(3)
    cluster_profile["Count"] = df_cluster["cluster"].value_counts().sort_index().values

    # merge raw để có số liệu gốc
    if "CustomerID" in df_raw.columns:
        merged = pd.merge(df_raw, df_cluster[["CustomerID", "cluster"]], on="CustomerID", how="left")
    else:
        merged = df_raw.copy()
        merged["cluster"] = df_cluster["cluster"].values

    # mô tả PCA (thấp/trung/cao)
    def label_pca(v: float, thresholds=(-0.3, 0.3)) -> str:
        return "Thấp" if v <= thresholds[0] else ("Cao" if v >= thresholds[1] else "Trung bình")

    profile_real = merged.groupby("cluster")[list({*use_cols} - set(pca_cols))].mean().round(2)
    if pca_cols:
        pca_means = df_cluster.groupby("cluster")[pca_cols].mean().round(2)
        profile_real = pd.concat([profile_real, pca_means], axis=1)
    desc_profile = profile_real.copy()
    for c in pca_cols:
        desc_profile[c] = profile_real[c].apply(label_pca)
    desc_profile.rename(columns={
        "pca_convenience": "Mức độ coi trọng sự tiện lợi",
        "pca_service_issue": "Vấn đề dịch vụ",
        "pca_deal_sensitive": "Nhạy cảm ưu đãi/đánh giá",
    }, inplace=True)

    # Lưu CHUẨN
    out_full    = DATA_DIR / "df_cluster_full.csv"
    out_profile = DATA_DIR / "cluster_profile_scaled.csv"
    out_desc    = DATA_DIR / "cluster_characteristics_descriptive.csv"

    df_cluster.to_csv(out_full, index=False)
    cluster_profile.to_csv(out_profile)
    desc_profile.to_csv(out_desc, index=False, encoding="utf-8-sig")

    return {
        "df_cluster_full": out_full,
        "cluster_profile_scaled": out_profile,
        "cluster_characteristics_descriptive": out_desc,
    }

# ========== RUN PIPELINE ==========
def run_pipeline(
    k_final: int = 3,
    feature_file: Optional[Path] = None,
) -> dict:
    """load → PCA nhóm → chọn features → KMeans → lưu file → figures."""
    df_scaled_model = pd.read_csv(DATA_DIR / "df_scaled_model.csv")
    df_raw          = pd.read_csv(DATA_DIR / "df_raw_dashboard.csv")

    ids = df_scaled_model["CustomerID"] if "CustomerID" in df_scaled_model.columns else pd.Series(range(len(df_scaled_model)))
    df_cluster = df_scaled_model.drop(columns=["CustomerID"], errors="ignore").copy()

    # PCA nhóm
    df_cluster = attach_group_pca(df_cluster, random_state=42)

    # chọn features (ưu tiên CSV cột 'feature' nếu cung cấp)
    features = None
    if feature_file is None:
        env_ff = os.getenv("FEATURE_FILE")
        if env_ff:
            feature_file = Path(env_ff)
    if feature_file and Path(feature_file).exists():
        feat = pd.read_csv(feature_file)
        if "feature" in feat.columns:
            features = feat["feature"].dropna().astype(str).tolist()
        else:
            features = feat.iloc[:, 0].dropna().astype(str).tolist()

    X = select_X(df_cluster, features)
    cols_src = features if features is not None else CLUSTER_FEATURES
    use_cols = [c for c in cols_src if c in df_cluster.columns]

    # gợi ý K
    Ks, inertias, sils = evaluate_k_range(X, 2, 11)

    # fit cuối & gán nhãn
    labels = kmeans_labels(X, k_final, random_state=42, n_init=20)
    df_cluster["cluster"] = labels
    df_cluster["CustomerID"] = ids

    # lưu
    pca_cols = [c for c in ["pca_convenience", "pca_service_issue", "pca_deal_sensitive"] if c in df_cluster.columns]
    paths = save_outputs(df_cluster, df_raw, use_cols, pca_cols)

    # figures
    fig_elbow = figure_elbow_silhouette(X, 2, 11)
    fig_pie   = figure_cluster_distribution(labels, scale=1.2)
    fig_pca   = figure_pca_scatter(X, labels)

    return {
        "DATA_DIR": DATA_DIR,
        "use_cols": use_cols,
        "Ks": Ks,
        "inertias": inertias,
        "silhouettes": sils,
        "labels": labels,
        "fig_elbow": fig_elbow,
        "fig_pie": fig_pie,
        "fig_pca": fig_pca,
        "paths": paths,
    }

# ========== CLI ==========
if __name__ == "__main__":
    K = int(os.getenv("K_FINAL", "3"))
    FEATURE_FILE = os.getenv("FEATURE_FILE")
    info = run_pipeline(k_final=K, feature_file=Path(FEATURE_FILE) if FEATURE_FILE else None)

    print("DATA_DIR:", info["DATA_DIR"])
    print("Features dùng:", info["use_cols"])
    print("Silhouette gợi ý (k=2..10):", [round(v,3) for v in info["silhouettes"]])
    print("Đã lưu:")
    for k, p in info["paths"].items():
        print(" -", k, "->", p)

    # Vẽ nhanh nếu chạy standalone
    # AFTER

