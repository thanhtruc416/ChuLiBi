# Nội dung vùng bảng có thanh cuộn dọc bên phải

import tkinter as tk
import pandas as pd
from pathlib import Path

HEADER_BG = "#CFAFCA"   # header tím nhạt như mock
ROW_EVEN = "#F7F4F7"
ROW_ODD  = "#FFFFFF"
TEXT     = "#374A5A"

COLUMNS = [
    ("ID", 120),
    ("Cluster", 180),
    ("Churn Risk", 160),
    ("Recommendation", 600),
]

def build_table(parent: tk.Widget, width: int, height: int, data: pd.DataFrame = None, search_term: str = None) -> tk.Canvas:
    """
    Tạo một Canvas chứa khung bảng cuộn. Trả về chính canvas để .pack() ở file khung.
    Thanh cuộn nằm SÁT mép phải vùng bảng, chỉ cuộn phần bảng.
    
    Args:
        parent: Parent widget
        width: Width of the table
        height: Height of the table
        data: DataFrame with recommendation data
        search_term: Search term to highlight (optional)
    """
    # Canvas làm viewport
    canvas = tk.Canvas(parent, bg="#FFFFFF", highlightthickness=0, bd=0)
    canvas.place(x=0, y=0, relwidth=1, relheight=1)

    # Scrollbar dọc: neo vào mép phải của canvas
    vbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    vbar.place(in_=canvas, relx=1.0, rely=0.0, relheight=1.0, anchor="ne", width=16)
    canvas.configure(yscrollcommand=vbar.set)

    # Khung thực sự chứa các hàng, nhúng vào canvas
    inner = tk.Frame(canvas, bg="#FFFFFF")
    inner_id = canvas.create_window(0, 0, window=inner, anchor="nw")

    # ----- Header -----
    header = tk.Frame(inner, bg=HEADER_BG)
    header.pack(fill="x", padx=0, pady=(0, 2))
    for title, col_w in COLUMNS:
        lbl = tk.Label(
            header, text=title, bg=HEADER_BG, fg="#FFFFFF",
            font=("Crimson Pro SemiBold", 16), anchor="w", padx=12, pady=8
        )
        lbl.pack(side="left")
        lbl.config(width=max(1, col_w // 9))

    # ----- Data rows (real or demo) -----
    def add_row(values, index, customer_id=None):
        # Highlight row if it matches search term
        is_match = False
        if search_term and customer_id:
            is_match = search_term.lower() in str(customer_id).lower()
        
        if is_match:
            bg = "#E8D4E3"  # Light purple highlight for matching rows
        else:
            bg = ROW_EVEN if index % 2 == 0 else ROW_ODD
        
        row = tk.Frame(inner, bg=bg)
        row.pack(fill="x")
        
        for (title, col_w), val in zip(COLUMNS, values):
            # Truncate long text for Recommendation column
            display_val = str(val)
            if title == "Recommendation" and len(display_val) > 70:
                display_val = display_val[:67] + "..."
            
            # Bold font for matching customer IDs
            font_style = "Crimson Pro SemiBold" if (is_match and title == "ID") else "Crimson Pro"
            
            lbl = tk.Label(
                row, text=display_val, bg=bg, fg=TEXT,
                font=(font_style, 14), anchor="w", padx=12, pady=10,
                wraplength=col_w - 24 if title == "Recommendation" else 0
            )
            lbl.pack(side="left")
            lbl.config(width=max(1, col_w // 9))

    # Load actual data or use sample
    if data is not None and not data.empty:
        # Use real recommendation data
        for i, (_, row_data) in enumerate(data.iterrows(), start=1):
            customer_id = str(row_data.get("Customer_ID", f"C{i:03d}"))
            cluster = str(row_data.get("Cluster", "Unknown"))
            churn_risk = str(row_data.get("Churn_Risk", "0%"))
            recommendation = str(row_data.get("action_name", "No action"))
            
            add_row((customer_id, cluster, churn_risk, recommendation), i, customer_id=customer_id)
    else:
        # Sample data for preview
        sample = [("C%03d" % i, "Premium", "20%", "Offer 10% discount on next purchase")
                  for i in range(1, 80)]
        for i, vals in enumerate(sample, start=1):
            add_row(vals, i, customer_id=vals[0])

    # ----- Đồng bộ kích thước và scrollregion -----
    def _on_inner_config(_=None):
        # cập nhật scrollregion theo nội dung
        canvas.configure(scrollregion=canvas.bbox("all"))
        # đảm bảo nội dung luôn tối thiểu bằng viewport theo chiều ngang
        w = max(width, inner.winfo_reqwidth())
        canvas.itemconfigure(inner_id, width=w)

    def _on_canvas_config(event):
        # khi viewport thay đổi kích thước, set lại width của inner cho khớp
        canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", _on_inner_config)
    canvas.bind("<Configure>", _on_canvas_config)

    # ----- Cuộn bằng bánh xe chuột (chỉ khi trỏ chuột trong bảng) -----
    def _on_wheel(event):
        # Windows/macOS: event.delta; Linux: <Button-4/5> bên dưới
        delta = event.delta
        if delta:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    canvas.bind("<MouseWheel>", _on_wheel)                      # Windows/macOS
    canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
    canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux

    return canvas
