# ui_content_Frame08_table.py
# Prediction table with scrollbar for Frame 8 (standalone-friendly)
# - API giữ nguyên: build_prediction_table(parent, width, height, df_result)
# - Có thêm build_cluster_filter_dropdown(parent, df_result, on_filter_change)
# - Có thể chạy riêng file này để preview

import tkinter as tk
from tkinter import ttk

HEADER_BG = "#644E94"   # purple (theme)
ROW_EVEN  = "#F7F4F7"
ROW_ODD   = "#FFFFFF"
TEXT      = "#2E2E2E"

COLUMNS = [
    ("Customer ID", 120),
    ("Cluster", 90),
    ("Churn Prob", 110),
    ("Churn Risk", 120),
    ("Cluster Description", 640),  # dài -> có wrap
]

# ----- helpers -----
def _col_width_to_chars(px: int) -> int:
    # heuristic: 1 char ~ 8px với font mặc định
    return max(1, int(px / 8))

def _label(parent, text, bg, fg=TEXT, bold=False, anchor="w", padx=12, pady=10, wrap=None):
    font = ("Crimson Pro", 12, "bold" if bold else "normal")
    lbl = tk.Label(parent, text=str(text), bg=bg, fg=fg, font=font, anchor=anchor, padx=padx, pady=pady, justify="left")
    if wrap:
        lbl.configure(wraplength=wrap)  # px
    return lbl


def build_prediction_table(parent: tk.Widget, width: int, height: int, df_result=None) -> tk.Canvas:
    """
    Create a scrollable canvas containing the churn prediction table.

    Args:
        parent: Container widget
        width:  Table width (px)
        height: Table height (px)
        df_result: DataFrame columns expected:
            Customer_ID, cluster, proba_churn_pct, pred_churn_label, cluster_label
    Returns:
        Canvas widget (viewport)
    """
    # Viewport canvas
    canvas = tk.Canvas(parent, bg="#FFFFFF", highlightthickness=0, bd=0, width=width, height=height)
    canvas.place(x=0, y=0, width=width, height=height)

    # Vertical scrollbar (gắn sát biên phải của canvas)
    vbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    vbar.place(in_=canvas, relx=1.0, rely=0.0, relheight=1.0, anchor="ne", width=14)
    canvas.configure(yscrollcommand=vbar.set)

    # Inner frame chứa bảng
    inner = tk.Frame(canvas, bg="#FFFFFF")
    inner_id = canvas.create_window(0, 0, window=inner, anchor="nw")

    # ---- optional: sort state khi click header ----
    sort_state = {"col": None, "asc": True}

    # ----- Header -----
    header = tk.Frame(inner, bg=HEADER_BG)
    header.pack(fill="x", padx=0, pady=(0, 2))

    def _set_header(label, idx, col_name):
        def _on_click(_=None):
            nonlocal sort_state
            if df_result is None or df_result.empty:
                return
            # map idx -> tên cột trong df
            name_map = {
                0: "Customer_ID",
                1: "cluster",
                2: "proba_churn_pct",
                3: "pred_churn_label",
                4: "cluster_label",
            }
            k = name_map.get(idx)
            if k is None or k not in df_result.columns:
                return
            asc = not (sort_state["col"] == k and sort_state["asc"])
            sort_state.update({"col": k, "asc": asc})
            # clear + rebuild
            for w in inner.pack_slaves():
                if w is not header:
                    w.destroy()
            _build_rows(df_result.sort_values(k, ascending=asc).reset_index(drop=True))
            _sync_scroll()
        label.bind("<Button-1>", _on_click)

    # dựng header nút
    for i, (title, col_w) in enumerate(COLUMNS):
        lbl = _label(header, title, HEADER_BG, fg="#FFFFFF", bold=True, anchor="w", padx=12, pady=10)
        lbl.pack(side="left")
        lbl.config(width=_col_width_to_chars(col_w))
        _set_header(lbl, i, title)

    # ----- Rows builder -----
    def _row_values(row, i):
        # Lấy giá trị có fallback hợp lý
        customer_id = row.get('Customer_ID', f'CUS{i + 1:03d}')
        cluster = row.get('cluster', 'N/A')
        proba_pct = row.get('proba_churn_pct', '0.0%')
        pred_label = row.get('pred_churn_label', 'Unknown')
        cluster_label = row.get('cluster_label', 'N/A')
        return (customer_id, f"Cluster {cluster} +1", proba_pct, pred_label, cluster_label)

    def _add_row(values, index):
        bg = ROW_EVEN if index % 2 == 0 else ROW_ODD
        row = tk.Frame(inner, bg=bg)
        row.pack(fill="x")

        for (title, col_w), val in zip(COLUMNS, values):
            wrap = col_w if title == "Cluster Description" else None
            lbl = _label(row, val, bg, wrap=wrap)
            lbl.pack(side="left")
            lbl.config(width=_col_width_to_chars(col_w))

    def _build_rows(df):
        if df is None or df.empty:
            # Sample rows
            sample = [
                (f"CUS{i:03d}", f"Cluster {(i % 3)}", f"{20 + i * 2}%",
                 "Có nguy cơ" if i % 3 == 0 else "Không rời bỏ",
                 "Khách hàng dịch vụ cao / nhạy cảm trải nghiệm" if i % 3 == 0
                 else "Khách hàng nhạy ưu đãi / ổn định")
                for i in range(1, 36)
            ]
            for idx, vals in enumerate(sample, start=1):
                _add_row(vals, idx)
            return

        # Có dữ liệu thật
        for i, row in df.iterrows():
            _add_row(_row_values(row, i), i + 1)

    # Build initial
    try:
        _build_rows(df_result)
        if df_result is not None and not df_result.empty:
            print(f"✓ Loaded {len(df_result)} prediction records into table")
        else:
            print("⚠ No prediction data available, showing sample data")
    except Exception as e:
        print(f"✗ Error loading prediction data: {e}")
        import traceback; traceback.print_exc()
        _build_rows(None)

    # ----- sync size & scrollregion -----
    def _sync_scroll(_=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # fit inner width to viewport
        try:
            canvas.itemconfigure(inner_id, width=canvas.winfo_width())
        except Exception:
            pass

    inner.bind("<Configure>", _sync_scroll)
    canvas.bind("<Configure>", _sync_scroll)

    # Mouse wheel
    def _on_wheel(event):
        delta = event.delta
        if delta:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")
    canvas.bind("<MouseWheel>", _on_wheel)              # Windows/macOS
    canvas.bind("<Button-4>", lambda e: _on_wheel(type("E", (), {"delta": 120})()))   # Linux up
    canvas.bind("<Button-5>", lambda e: _on_wheel(type("E", (), {"delta": -120})()))  # Linux down

    return canvas


def build_cluster_filter_dropdown(parent: tk.Widget, df_result=None, on_filter_change=None):
    """Right-aligned 'Filter by Cluster' (sát mép phải)."""
    filter_frame = tk.Frame(parent, bg="#FFFFFF", highlightthickness=0, bd=0)

    # options
    cluster_options = ["All Clusters"]
    if df_result is not None and hasattr(df_result, "columns") and "cluster" in df_result.columns:
        try:
            unique_clusters = sorted(map(int, set(df_result["cluster"].dropna().tolist())))
            cluster_options += [f"Cluster {c+1}" for c in unique_clusters]
        except Exception:
            cluster_options += ["Cluster 0", "Cluster 1", "Cluster 2"]
    else:
        cluster_options += ["Cluster 0", "Cluster 1", "Cluster 2"]

    # style combobox
    style = ttk.Style()
    style.configure("Custom.TCombobox",
                    fieldbackground="#FFFFFF",
                    background="#644E94",
                    foreground=TEXT)

    var = tk.StringVar(value=cluster_options[0])
    cmb = ttk.Combobox(filter_frame, values=cluster_options, textvariable=var,
                       state="readonly", width=20, font=("Crimson Pro", 12),
                       style="Custom.TCombobox")

    # pack theo thứ tự phải → trái, KHÔNG padding
    cmb.pack(side="right", padx=0, pady=0)
    label = _label(filter_frame, "Filter by Cluster:", "#FFFFFF", bold=True, anchor="e", padx=0, pady=0)
    label.pack(side="left", padx=(6, 0), pady=0)  # 6px giữa label & combo; muốn sát hơn: đổi 6 -> 0

    if on_filter_change:
        cmb.bind("<<ComboboxSelected>>", lambda e: on_filter_change(cmb.get()))

    return filter_frame



# ========= Standalone preview =========
# ========= Standalone preview =========
if __name__ == "__main__":
    import pandas as pd

    N = 60
    data = pd.DataFrame({
        "Customer_ID": [f"CUS{i:03d}" for i in range(1, N+1)],
        "cluster": [i % 4 for i in range(1, N+1)],
        "proba_churn_pct": [f"{20 + (i % 9) * 6}%" for i in range(1, N+1)],
        "pred_churn_label": ["Có nguy cơ" if i % 3 == 0 else "Không rời bỏ"
                             for i in range(1, N+1)],
        "cluster_label": ["Khách hàng dịch vụ cao / nhạy cảm trải nghiệm" if i % 3 == 0
                          else "Khách hàng nhạy ưu đãi / ổn định"
                          for i in range(1, N+1)]
    })

    root = tk.Tk()
    root.title("Preview - ui_content_Frame08_table")
    root.geometry("1024x600")
    root.configure(bg="#FFFFFF")

    container = tk.Frame(root, bg="#FFFFFF")
    container.pack(fill="both", expand=True, padx=12, pady=12)

    def on_change(sel):
        for w in table_holder.winfo_children():
            w.destroy()
        if sel == "All Clusters":
            df_show = data
        else:
            num = int(sel.split()[-1])
            df_show = data[data["cluster"] == num]
        build_prediction_table(table_holder, width=980, height=420, df_result=df_show)

    filter_bar = build_cluster_filter_dropdown(container, data, on_change)
    filter_bar.pack(anchor="w", pady=(0, 8))

    table_holder = tk.Frame(container, bg="#FFFFFF", bd=1, relief="solid")
    table_holder.pack(fill="both", expand=True)

    build_prediction_table(table_holder, width=980, height=420, df_result=data)
    root.mainloop()
