# Nội dung vùng bảng có thanh cuộn dọc bên phải

import tkinter as tk

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

def build_table(parent: tk.Widget, width: int, height: int) -> tk.Canvas:
    """
    Tạo một Canvas chứa khung bảng cuộn. Trả về chính canvas để .pack() ở file khung.
    Thanh cuộn nằm SÁT mép phải vùng bảng, chỉ cuộn phần bảng.
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

    # ----- Demo data (thay bằng dữ liệu thật) -----
    def add_row(values, index):
        bg = ROW_EVEN if index % 2 == 0 else ROW_ODD
        row = tk.Frame(inner, bg=bg)
        row.pack(fill="x")
        for (title, col_w), val in zip(COLUMNS, values):
            lbl = tk.Label(
                row, text=val, bg=bg, fg=TEXT,
                font=("Crimson Pro", 14), anchor="w", padx=12, pady=10
            )
            lbl.pack(side="left")
            lbl.config(width=max(1, col_w // 9))

    sample = [("C%03d" % i, "Premium", "20%", "Offer 10% discount on next purchase")
              for i in range(1, 80)]
    for i, vals in enumerate(sample, start=1):
        add_row(vals, i)

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
