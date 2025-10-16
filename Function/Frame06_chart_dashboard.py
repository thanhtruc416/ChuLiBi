# --- [ADD] matplotlib + numpy/pandas để vẽ vào Tkinter ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import mysql.connector
'''
server="localhost"
port=3306
database="ml_dashboard"
user="root"
password="Hoanganh22"

conn = mysql.connector.connect(
    host=server, port=port, database=database, user=user, password=password
)

# Đọc thẳng bảng vào pandas
query = "SELECT * FROM df_raw_dashboard;"
df = pd.read_sql(query, con=conn)

# Chuẩn hoá tên cột về snake_case
df.columns = [c.strip().replace(" ", "_").replace(".", "").replace("/", "_").lower() for c in df.columns]

conn.close()
'''


# --- Chuẩn hoá dữ liệu ---
def _normalize_gender(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ["male", "m", "nam"]: return "Male"
    if s in ["female", "f", "nu", "nữ"]: return "Female"
    return "Khác"

def _map_late_delivery(row):
    val = row.get("late_delivery", np.nan)
    enc = row.get("late_delivery_encoded", np.nan)
    if pd.notna(val):
        s = str(val).strip().lower()
        if any(k in s for k in ["agree", "yes"]):   return "Late deliveries"
        if any(k in s for k in ["disagree", "no"]): return "On-time deliveries"
        if "neutral" in s:                           return "Neutral/uncertain"
    if pd.notna(enc):
        try:
            e = float(enc)
            if e >= 1: return "Late"
            if e == 0: return "Not Late"
        except: pass
    return "Middle"

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["age", "no_of_orders_placed", "delivery_time"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["gender_norm"] = out["gender"].apply(_normalize_gender) if "gender" in out.columns else np.nan
    if ("late_delivery" in out.columns) or ("late_delivery_encoded" in out.columns):
        out["late_status"] = out.apply(_map_late_delivery, axis=1)
    else:
        out["late_status"] = np.nan
    return out

# --- [ADD] Các hàm vẽ *theo Axes* để nhúng vào Tkinter ---
def _plot_bar_occupation_gender(ax, dfp: pd.DataFrame, mode: str = "grouped"):
    data = dfp.dropna(subset=["occupation", "gender_norm"])
    if data.empty:
        ax.set_title("No data for Occupation × Gender"); return
    counts = data.groupby(["occupation", "gender_norm"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    for spine in ax.spines.values():
        spine.set_edgecolor("#B992B9")  # Màu xám nhạt
        spine.set_linewidth(1)

    x = np.arange(len(counts.index))
    width = 0.25
    genders = ["Male", "Female", "??"]
    colors  = {"Male": "#644E94", "Female": "#C6ABC5", "Khác": "gray"}
    available = [g for g in genders if g in counts.columns]

    ax.clear()
    if mode == "stacked":
        bottom = np.zeros(len(x))
        for g in available:
            ax.bar(x, counts[g].values, bottom=bottom, label=g, color=colors.get(g))
            bottom += counts[g].values
    else:
        n = len(available)
        offsets = (np.arange(n) - (n-1)/2) * width
        for i, g in enumerate(available):
            ax.bar(x + offsets[i], counts[g].values, width=width, label=g, color=colors.get(g))

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=0, ha="center", fontsize=9, color="#7A467A")
    ax.set_ylabel("Number of Customers", fontsize=11, fontfamily="Crimson Pro",color="#7A467A")
    ax.set_xlabel("Occupation", fontsize=11, fontfamily="Crimson Pro",color="#7A467A")
    ax.tick_params(axis="y", labelcolor="#644E94")
    legend = ax.legend(title="Gender", fontsize=11, title_fontsize=11)
    legend.get_title().set_color("#7A467A")

def _make_age_bins(age: pd.Series, bin_width=4, start_at=18):
    a = age.dropna()
    if a.empty: return pd.IntervalIndex([])
    import math
    min_age = max(start_at, int(np.floor(a.min())))
    max_age = int(np.ceil(a.max()))
    edges   = list(range(min_age, max_age + bin_width, bin_width))
    if edges[-1] < max_age: edges.append(max_age + bin_width)
    return pd.IntervalIndex.from_breaks(edges, closed="left")

def _plot_line_orders_by_age(ax, dfp: pd.DataFrame, bin_width=4, start_at=18):
    data = dfp.dropna(subset=["age", "no_of_orders_placed"])
    if data.empty:
        ax.set_title("No data for Age/Orders"); return
    bins = _make_age_bins(data["age"], bin_width, start_at)
    if len(bins) == 0:
        ax.set_title("No bins created for Age/Orders"); return

    data = data.assign(age_bin=pd.cut(data["age"], bins))
    agg  = data.groupby("age_bin")["no_of_orders_placed"].sum().reindex(bins, fill_value=0)

    for spine in ax.spines.values():
        spine.set_edgecolor("#B992B9")  # Màu xám nhạt
        spine.set_linewidth(1)

    x = np.arange(len(agg))
    ax.clear()
    ax.plot(x, agg.values, marker="o",color="#A08CB1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{iv.left}-{iv.right-1}" for iv in agg.index], rotation=20, ha="center", fontsize=9, color="#7A467A")
    ax.set_xlabel("(age group)", fontsize=11, font="Crimson Pro",color="#7A467A")
    ax.set_ylabel("orders", fontsize=11, fontfamily="Crimson Pro",color="#7A467A")
    ax.tick_params(axis="y", labelcolor="#644E94")
    peak_idx = int(np.argmax(agg.values)); peak_iv = agg.index[peak_idx]; peak_val = agg.values[peak_idx]
    ax.annotate(
        f"Peak: {int(peak_val)} orders ({peak_iv.left}-{peak_iv.right - 1})",
        xy=(peak_idx, peak_val),  # điểm đỉnh
        xytext=(12, 0),  # → nhích sang phải 12pt (thử 10–16)
        textcoords="offset points",
        ha="left", va="center",  # canh trái, giữa theo chiều dọc
        fontsize=11, color="#7A467A",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#7A467A"),
        clip_on=False
    )
    ax.margins(x=0.06, y=0.12)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_position(("outward", 1))  # nhích ra ngoài
        ax.spines[side].set_linewidth(1)

    # cho ticks hướng ra ngoài + thêm khoảng cách nhãn khỏi viền
    ax.tick_params(axis='both', which='major', length=8, width=1.0, direction='out', pad=4)
    # (tuỳ chọn) vạch chia phụ
    ax.tick_params(axis='both', which='minor', length=5, width=1.0, direction='out')

def _plot_pie_meal_share(ax, dfp: pd.DataFrame):
    col = "frequently_ordered_meal_category"
    if col not in dfp.columns:
        ax.set_title("Missing column: frequently_ordered_meal_category"); return
    data = dfp[col].dropna().astype(str).str.strip().str.title()
    mapping = {"Lunch":"Lunch", "Dinner":"Dinner", "Snacks":"Snacks", "Breakfast":"Breakfast"}
    data = data.map(lambda x: mapping.get(x, x))
    counts = data.value_counts()
    ax.clear()

    if counts.empty:
        ax.set_title("No data for meal category")
        return
    color_map = {
        "Breakfast": "#E3D4E0",
        "Lunch": "#C6ABC5",
        "Snacks": "#B992B9",
        "Dinner": "#706093"
    }
    colors = [color_map.get(label, "#CCCCCC") for label in counts.index]

    # Vẽ biểu đồ
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.7),
        textprops={"fontsize": 9, "color": "#7A467A", "fontfamily": "Crimson Pro"}
    )

    # Giữ hình tròn
    ax.set_aspect("equal")


def _plot_stacked_hist_delivery(ax, dfp: pd.DataFrame, bin_width=10):
    if "delivery_time" not in dfp.columns:
        ax.set_title("Missing column: delivery_time"); return
    data = dfp.dropna(subset=["delivery_time"]).copy()
    if data.empty:
        ax.set_title("No data for delivery time"); return
    data["late_status"] = data["late_status"].fillna("Neutral/uncertain")

    for spine in ax.spines.values():
        spine.set_edgecolor("#B992B9")  # Màu xám nhạt
        spine.set_linewidth(1)

    t = pd.to_numeric(data["delivery_time"], errors="coerce").dropna()
    if t.empty:
        ax.set_title("No numeric delivery time"); return

    import math
    tmin = int(math.floor(t.min() / bin_width) * bin_width)
    tmax = int(math.ceil(t.max()  / bin_width) * bin_width)
    edges = list(range(tmin, tmax + bin_width, bin_width))

    data["time_bin"] = pd.cut(data["delivery_time"], bins=edges, right=False, include_lowest=True)
    ct = pd.crosstab(data["time_bin"], data["late_status"]).reindex(
        pd.IntervalIndex.from_breaks(edges, closed="left"), fill_value=0
    )
    order = ["On-time deliveries", "Neutral/uncertain", "Late deliveries"]
    for s in order:
        if s not in ct.columns: ct[s] = 0
    ct = ct[order]

    x = np.arange(len(ct.index))
    bottom = np.zeros(len(x))
    colors = {"On-time deliveries":"#FAE4F2", "Neutral/uncertain":"#C6ABC5", "Late deliveries":"#644E94"}

    ax.clear()
    for s in order:
        vals = ct[s].values
        ax.bar(x, vals, bottom=bottom, label=s, color=colors[s])
        bottom += vals
    ax.set_xticks(x);
    ax.set_xticklabels([f"{iv.left}–{iv.right}" for iv in ct.index], rotation=20, ha="center",fontsize=9, color="#7A467A")
    ax.set_xlabel("(minutes)", fontsize=8, fontfamily="Crimson Pro", color="#7A467A", labelpad=6)
    # canh phải & dời toạ độ nhãn trục X (tọa độ theo trục axes)
    ax.xaxis.get_label().set_horizontalalignment("right")
    ax.xaxis.set_label_coords(1.13, -0.12)  # 1.0 = mép phải; -0.12 = thấp hơn trục (chỉnh tuỳ ý)
    ax.set_ylabel("(orders)",fontsize=9, fontfamily="Crimson Pro",color="#7A467A")
    ax.tick_params(axis="y", labelcolor="#644E94",labelsize=9)
    legend=ax.legend(
        title = "Status",
        fontsize=7,
        loc = "lower center",  # Đặt ở giữa phía dưới
        bbox_to_anchor = (0.5, -0.60),  # Dịch xuống dưới khung vẽ
        ncol = 3,  # Hiển thị theo hàng ngang
        title_fontsize = 7,)
    legend.get_title().set_color("#7A467A")
