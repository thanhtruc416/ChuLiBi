# ui_content_Frame08_table.py
# Prediction table with scrollbar for Frame 8

import tkinter as tk
from tkinter import ttk

HEADER_BG = "#644E94"   # Purple to match Frame 8 theme
ROW_EVEN = "#F7F4F7"
ROW_ODD  = "#FFFFFF"
TEXT     = "#2E2E2E"
CHURN_HIGH = "#E74C3C"  # Red for high churn risk
CHURN_LOW = "#27AE60"   # Green for low churn risk

COLUMNS = [
    ("Customer ID", 90),
    ("Cluster", 80),
    ("Churn Prob", 80),
    ("Churn Risk", 100),
    ("Cluster Description", 800),  # Increased width for full text visibility
]

def build_prediction_table(parent: tk.Widget, width: int, height: int, df_result=None) -> tk.Canvas:
    """
    Create a scrollable canvas containing the churn prediction table.
    Similar to Frame 10's table implementation.
    
    Args:
        parent: Parent widget
        width: Table width
        height: Table height
        df_result: DataFrame with prediction results (columns: Customer_ID, cluster, 
                   proba_churn_pct, pred_churn_label, cluster_label)
    
    Returns:
        Canvas widget with embedded table
    """
    # Canvas as viewport
    canvas = tk.Canvas(parent, bg="#FFFFFF", highlightthickness=0, bd=0)
    canvas.place(x=0, y=0, relwidth=1, relheight=1)

    # Vertical scrollbar: anchored to right edge of canvas
    vbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    vbar.place(in_=canvas, relx=1.0, rely=0.0, relheight=1.0, anchor="ne", width=16)
    canvas.configure(yscrollcommand=vbar.set)

    # Inner frame to hold all rows, embedded in canvas
    inner = tk.Frame(canvas, bg="#FFFFFF")
    inner_id = canvas.create_window(0, 0, window=inner, anchor="nw")

    # ----- Header -----
    header = tk.Frame(inner, bg=HEADER_BG)
    header.pack(fill="x", padx=0, pady=(0, 2))
    for title, col_w in COLUMNS:
        lbl = tk.Label(
            header, text=title, bg=HEADER_BG, fg="#FFFFFF",
            font=("Crimson Pro SemiBold", 14), anchor="w", padx=12, pady=10
        )
        lbl.pack(side="left")
        lbl.config(width=max(1, col_w // 9))  # Use same width calculation as data rows

    # ----- Data rows -----
    def add_row(values, index):
        """Add a single data row to the table"""
        bg = ROW_EVEN if index % 2 == 0 else ROW_ODD
        row = tk.Frame(inner, bg=bg)
        row.pack(fill="x")
        
        for col_idx, ((title, col_w), val) in enumerate(zip(COLUMNS, values)):
            lbl = tk.Label(
                row, text=str(val), bg=bg, fg=TEXT,
                font=("Crimson Pro", 14), anchor="w", padx=12, pady=10
            )
            lbl.pack(side="left")
            lbl.config(width=max(1, col_w // 9))

    # ----- Load actual data or show sample -----
    if df_result is not None and not df_result.empty:
        try:
            # Use actual prediction data
            for i, row_data in df_result.iterrows():
                customer_id = row_data.get('Customer_ID', f'CUS{i+1:03d}')
                cluster = row_data.get('cluster', 'N/A')
                proba_pct = row_data.get('proba_churn_pct', '0.0%')
                pred_label = row_data.get('pred_churn_label', 'Unknown')
                cluster_label = row_data.get('cluster_label', 'N/A')
                
                values = (
                    str(customer_id),
                    f"Cluster {cluster}",
                    str(proba_pct),
                    str(pred_label),
                    str(cluster_label)
                )
                add_row(values, i + 1)
            
            print(f"✓ Loaded {len(df_result)} prediction records into table")
        except Exception as e:
            print(f"✗ Error loading prediction data: {e}")
            import traceback
            traceback.print_exc()
            # Add error row
            add_row(("Error loading data", str(e), "", "", ""), 1)
    else:
        # Show sample data if no real data available
        print("⚠ No prediction data available, showing sample data")
        sample = [
            (f"CUS{i:03d}", f"Cluster {(i % 3)}", f"{20 + i*2}%", 
             "Có nguy cơ" if i % 3 == 0 else "Không rời bỏ",
             "Khách hàng dịch vụ cao / nhạy cảm trải nghiệm" if i % 3 == 0 
             else "Khách hàng nhạy ưu đãi / ổn định")
            for i in range(1, 50)
        ]
        for i, vals in enumerate(sample, start=1):
            add_row(vals, i)

    # ----- Synchronize size and scrollregion -----
    def _on_inner_config(_=None):
        # Update scrollregion based on content
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Ensure content width matches viewport width
        w = max(width, inner.winfo_reqwidth())
        canvas.itemconfigure(inner_id, width=w)

    def _on_canvas_config(event):
        # When viewport changes size, adjust inner width to match
        canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", _on_inner_config)
    canvas.bind("<Configure>", _on_canvas_config)

    # ----- Mouse wheel scrolling (only when mouse is over table) -----
    def _on_wheel(event):
        # Windows/macOS: event.delta; Linux: <Button-4/5> below
        delta = event.delta
        if delta:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    canvas.bind("<MouseWheel>", _on_wheel)                      # Windows/macOS
    canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
    canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux

    return canvas


def build_cluster_filter_dropdown(parent: tk.Widget, df_result=None, on_filter_change=None):
    """
    Build a dropdown filter for cluster selection
    
    Args:
        parent: Parent widget
        df_result: DataFrame with prediction results
        on_filter_change: Callback function(cluster_value) when filter changes
    
    Returns:
        Combobox widget
    """
    filter_frame = tk.Frame(parent, bg="#FFFFFF")
    
    # Label
    label = tk.Label(
        filter_frame,
        text="Filter by Cluster:",
        bg="#FFFFFF",
        fg="#2E2E2E",
        font=("Young Serif", 14)
    )
    label.pack(side="left", padx=(0, 10))
    
    # Get unique clusters
    cluster_options = ["All Clusters"]
    if df_result is not None and not df_result.empty and 'cluster' in df_result.columns:
        unique_clusters = sorted(df_result['cluster'].dropna().unique())
        cluster_options += [f"Cluster {int(c)}" for c in unique_clusters]
    else:
        cluster_options += ["Cluster 0", "Cluster 1", "Cluster 2"]
    
    # Combobox
    style = ttk.Style()
    style.configure("Custom.TCombobox", 
                    fieldbackground="#FFFFFF",
                    background="#644E94",
                    foreground="#2E2E2E")
    
    combobox = ttk.Combobox(
        filter_frame,
        values=cluster_options,
        state="readonly",
        font=("Crimson Pro", 12),
        width=20,
        style="Custom.TCombobox"
    )
    combobox.current(0)  # Default to "All Clusters"
    combobox.pack(side="left")
    
    # Bind selection event
    if on_filter_change:
        combobox.bind("<<ComboboxSelected>>", 
                     lambda e: on_filter_change(combobox.get()))
    
    return filter_frame
