# ui_content_Frame08.py
# Enhanced version with direct integration with Frame08_churn.py
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Entry, Button, Frame, Label
import sys

# Add paths for imports
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets_Frame08_2")
PROJECT_ROOT = OUTPUT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import churn functions directly
try:
    from Function.Frame08_churn import (
        get_churn_data,
        _plot_churn_rate_by_segment,
        _plot_reasons_pie,
        _plot_feature_importance,
        _plot_shap_summary
    )
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    HAS_CHURN_FUNCTIONS = True
except ImportError as e:
    print(f"Warning: Could not import churn functions: {e}")
    HAS_CHURN_FUNCTIONS = False

# Import table builder
try:
    from Frame.Frame08.ui_content_Frame08_table import build_prediction_table, build_cluster_filter_dropdown

    HAS_TABLE_BUILDER = True
except ImportError as e:
    print(f"Warning: Could not import table builder: {e}")
    HAS_TABLE_BUILDER = False


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# Global data cache
_churn_data = None


def get_cached_data():
    """Load data once and cache it"""
    global _churn_data
    if _churn_data is None and HAS_CHURN_FUNCTIONS:
        print("Loading churn data...")
        _churn_data = get_churn_data()
        if _churn_data:
            print("✓ Churn data loaded successfully!")
    return _churn_data


def build_content(parent: tk.Widget, width: int, height: int) -> Canvas:
    """
    Build full content with real data integration
    """
    canvas = Canvas(
        parent,
        bg="#D4C5D2",
        width=width,
        height=height,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )
    canvas.pack(fill="both", expand=True)

    # Keep image references
    _img_refs = []

    def _img(name: str) -> PhotoImage:
        try:
            im = PhotoImage(file=relative_to_assets(name))
            _img_refs.append(im)
            return im
        except Exception as e:
            print(f"Could not load image {name}: {e}")
            return None

    canvas._img_refs = _img_refs

    # Load data
    data = get_cached_data()
    if data:
        print("✓ Using loaded churn data")
    else:
        print("✗ No churn data available, using defaults")

    # ========== TOP SECTION: KPIs ==========

    # Avg Churn KPI
    img_avg = _img("image_AvgChurn.png")
    if img_avg:
        canvas.create_image(179.0, 101.0, image=img_avg)

    canvas.create_text(130.0, 145.0, anchor="nw", text="Avg Churn",
                       fill="#000000", font=("Young Serif", 18))

    avg_churn_value = "44%"
    if data:
        try:
            avg_churn = data['avg_churn']
            avg_churn_value = f"{avg_churn * 100:.0f}%"
        except Exception as e:
            print(f"Error getting avg churn: {e}")

    canvas.create_text(140.0, 88.0, anchor="nw", text=avg_churn_value,
                       fill="#706093", font=("Kodchasan Regular", 40))

    img_icon_avg = _img("image_iconAvgChurn.png")
    if img_icon_avg:
        canvas.create_image(169.0, 58.0, image=img_icon_avg)

    # Clusters KPI
    img_clusters = _img("image_Clusters.png")
    if img_clusters:
        canvas.create_image(525.0, 101.0, image=img_clusters)

    canvas.create_text(484.0, 138.0, anchor="nw", text="Clusters",
                       fill="#000000", font=("Young Serif", 18))

    num_clusters = "4"
    if data:
        try:
            num_clusters = str(data['num_clusters'])
        except Exception as e:
            print(f"Error getting num clusters: {e}")

    canvas.create_text(509.0, 88.0, anchor="nw", text=num_clusters,
                       fill="#706093", font=("Kodchasan Regular", 45))

    img_icon_clusters = _img("image_iconClusters.png")
    if img_icon_clusters:
        canvas.create_image(526.0, 56.6, image=img_icon_clusters)

    # Best Model KPI
    img_best = _img("image_LogisticRegression.png")
    if img_best:
        canvas.create_image(880.0, 104.0, image=img_best)

    canvas.create_text(830.0, 47.0, anchor="nw", text="Best Model",
                       fill="#000000", font=("Young Serif", 18))

    best_model = "  Logistic\nRegression"
    if data:
        try:
            model_name = data['best_model']
            if "Logistic" in model_name:
                best_model = "  Logistic\nRegression"
            elif "Random" in model_name:
                best_model = "  Random\n   Forest"
            elif "XGBoost" in model_name:
                best_model = "XGBoost"
        except Exception as e:
            print(f"Error getting best model: {e}")

    canvas.create_text(800.0, 88.0, anchor="nw", text=best_model,
                       fill="#706093", font=("Young Serif", 32))

    # ========== CHARTS SECTION ==========

    # Churn Rate Chart - Background image first (placed behind)
    img_rate_bg = _img("image_ChurnRate.png")
    if img_rate_bg:
        canvas.create_image(354.0, 380.0, image=img_rate_bg)

    canvas.create_text(59.0, 213.0, anchor="nw",
                       text="Churn Rate by Customer Segment",
                       fill="#000000", font=("Young Serif", 20))

    # Create frame with transparent background to overlay on the image
    # Adjusted size and position to fit within the white background
    chart_frame_1 = Frame(canvas, bg="#FFFFFF", width=420, height=300)
    canvas.create_window(350, 390, window=chart_frame_1, anchor="center")

    if data and HAS_CHURN_FUNCTIONS:
        try:
            fig = Figure(figsize=(4.2, 3.0), dpi=100, facecolor='#FFFFFF')
            ax = fig.add_subplot(111)
            _plot_churn_rate_by_segment(ax, data['df_core'], data['df'])

            fig.tight_layout(pad=0.5)

            canvas_widget = FigureCanvasTkAgg(fig, master=chart_frame_1)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            print("✓ Churn rate chart created")
        except Exception as e:
            print(f"✗ Error creating churn rate chart: {e}")
            import traceback
            traceback.print_exc()

    # Reasons Chart - Background image first (placed behind)
    img_reason_bg = _img("image_ReasonsChart.png")
    if img_reason_bg:
        canvas.create_image(880.0, 380.0, image=img_reason_bg)

    canvas.create_text(740.0, 216.0, anchor="nw", text="Reasons Chart",
                       fill="#000000", font=("Young Serif", 20))

    # Adjusted size and position to fit within the white background
    chart_frame_2 = Frame(canvas, bg="#FFFFFF", width=420, height=300)
    canvas.create_window(880, 390, window=chart_frame_2, anchor="center")

    if data and data.get('feature_importance') is not None and HAS_CHURN_FUNCTIONS:
        try:
            # Adjusted figure size with proper padding for labels
            fig = Figure(figsize=(3.0, 2.8), dpi=100, facecolor="#FFFFFF")
            ax = fig.add_subplot(111)
            _plot_reasons_pie(ax, data['feature_importance'])

            fig.tight_layout(pad=0.8, rect=[0.05, 0.05, 0.95, 0.95])

            canvas_widget = FigureCanvasTkAgg(fig, master=chart_frame_2)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            print("✓ Reasons chart created")
        except Exception as e:
            print(f"✗ Error creating reasons chart: {e}")
            import traceback
            traceback.print_exc()

    # ========== MODEL EVALUATION TABLE ==========

    # Table background image first (placed behind)
    img_table_bg = _img("image_Table.png")
    if img_table_bg:
        canvas.create_image(538.0, 764.0, image=img_table_bg)

    canvas.create_text(69.0, 630.0, anchor="nw",
                       text="Model Evaluation Metrics Table",
                       fill="#000000", font=("Young Serif", 20))

    # Adjusted table frame to fit within background
    table_frame = Frame(canvas, bg='#FFFFFF', width=1000, height=240)
    canvas.create_window(538, 775, window=table_frame, anchor="center")

    # Create table with model comparison data
    if data and data.get('eval_metrics') is not None:
        try:
            import tkinter.font as tkfont
            eval_df = data['eval_metrics']

            # Define colors
            header_bg = "#644E94"
            header_fg = "#FFFFFF"
            row_bg_1 = "#F5F5F5"
            row_bg_2 = "#FFFFFF"
            text_color = "#2E2E2E"
            border_color = "#E0E0E0"

            # Column configuration - use the actual columns from eval_metrics
            columns = ['Model', 'AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
            col_widths = [200, 130, 130, 130, 130, 130]  # Increased widths

            # Container frame for table (no canvas wrapper for simplicity)
            table_container = Frame(table_frame, bg='#FFFFFF')
            table_container.pack(fill="both", expand=True, padx=10, pady=10)

            # Font settings
            header_font = tkfont.Font(family="Young Serif", size=12, weight="bold")
            cell_font = tkfont.Font(family="Crimson Pro", size=11)

            # Header row
            header_frame = Frame(table_container, bg=header_bg)
            header_frame.grid(row=0, column=0, columnspan=len(columns), sticky="ew")

            for col_idx, col_name in enumerate(columns):
                header_label = Label(
                    header_frame,
                    text=col_name,
                    bg=header_bg,
                    fg=header_fg,
                    font=header_font,
                    width=15 if col_idx == 0 else 10,  # Wider for Model column
                    anchor="center",
                    padx=8,
                    pady=10
                )
                header_label.grid(row=0, column=col_idx, sticky="ew", padx=1)

            # Data rows
            for row_idx, row_data in eval_df.iterrows():
                # Alternate row colors
                row_bg = row_bg_1 if row_idx % 2 == 0 else row_bg_2

                for col_idx, col_name in enumerate(columns):
                    # Format the value
                    value = row_data[col_name]
                    if col_name == 'Model':
                        display_value = str(value)
                    else:
                        # Format numeric values to 3 decimal places
                        try:
                            numeric_val = float(value)
                            display_value = f"{numeric_val:.3f}"
                        except:
                            display_value = str(value)

                    # Determine font for best model highlighting
                    cell_fg = text_color
                    cell_font_obj = cell_font
                    if row_idx == 0:  # Best model (first row)
                        cell_fg = "#644E94"
                        cell_font_obj = tkfont.Font(family="Crimson Pro", size=11, weight="bold")

                    # Create cell label directly in table_container
                    cell_label = Label(
                        table_container,
                        text=display_value,
                        bg=row_bg,
                        fg=cell_fg,
                        font=cell_font_obj,
                        width=15 if col_idx == 0 else 10,
                        anchor="center",
                        padx=8,
                        pady=8,
                        relief="flat"
                    )
                    cell_label.grid(row=row_idx + 1, column=col_idx, sticky="ew", padx=1, pady=1)

            print("✓ Model evaluation table created")
        except Exception as e:
            print(f"✗ Error creating evaluation table: {e}")
            import traceback
            traceback.print_exc()
            # Show error message in table frame
            error_label = Label(table_frame, text="Error loading model metrics",
                                bg='#FFFFFF', fg='#666666', font=("Crimson Pro", 12))
            error_label.pack(expand=True)
    else:
        # Show placeholder when no data
        placeholder_label = Label(table_frame, text="No model evaluation data available",
                                  bg='#FFFFFF', fg='#999999', font=("Crimson Pro", 12))
        placeholder_label.pack(expand=True)

    # ========== DETAIL ANALYSIS SECTION ==========

    canvas.create_text(48.0, 960.0, anchor="nw", text="Detail Analysis",
                       fill="#706093", font=("Young Serif", 32))

    # Feature Importance - Background image first (placed behind)
    img_feat_bg = _img("image_FeatureImportant.png")
    if img_feat_bg:
        canvas.create_image(280.0, 1275.0, image=img_feat_bg)

    canvas.create_text(55.0, 1047.0, anchor="nw", text="Feature Important",
                       fill="#000000", font=("Young Serif", 20))

    # Adjusted frame to fit within background
    fi_frame = Frame(canvas, bg='white', width=400, height=400)
    canvas.create_window(280, 1280, window=fi_frame, anchor="center")

    if data and data.get('feature_importance') is not None and HAS_CHURN_FUNCTIONS:
        try:
            fig = Figure(figsize=(4.0, 4.0), dpi=100, facecolor='white')
            ax = fig.add_subplot(111)
            _plot_feature_importance(ax, data['feature_importance'], top_n=10)

            fig.tight_layout(pad=0.3)

            canvas_widget = FigureCanvasTkAgg(fig, master=fi_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)

            print("✓ Feature importance chart created")
        except Exception as e:
            print(f"✗ Error creating feature importance: {e}")
            import traceback
            traceback.print_exc()

    # SHAP Summary Plot - Background image first (placed behind)
    img_shap_bg = _img("image_SHAP.png")
    if img_shap_bg:
        canvas.create_image(807.0, 1275.0, image=img_shap_bg)

    canvas.create_text(592.0, 1047.0, anchor="nw", text="SHAP Summary Plot",
                       fill="#000000", font=("Young Serif", 20))

    # Adjusted frame to fit within background
    shap_frame = Frame(canvas, bg='white', width=400, height=400)
    canvas.create_window(807, 1280, window=shap_frame, anchor="center")

    if data and data.get('bundle') and data.get('X') is not None and HAS_CHURN_FUNCTIONS:
        try:
            fig = Figure(figsize=(4.0, 4.0), dpi=100, facecolor='white')
            _plot_shap_summary(fig, data['bundle'], data['X'])

            fig.tight_layout(pad=0.3)

            canvas_widget = FigureCanvasTkAgg(fig, master=shap_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill="both", expand=True)

            print("✓ SHAP summary plot embedded")
        except Exception as e:
            print(f"✗ Error embedding SHAP plot: {e}")
            import traceback
            traceback.print_exc()

    # ========== PREDICTION TABLE SECTION ==========

    # Background images for table area
    image_BGPredict = _img("image_Table.png")
    if image_BGPredict:
        canvas.create_image(538.0, 1730.0, image=image_BGPredict)
        canvas.create_image(538.0, 1930.0, image=image_BGPredict)

    # Title
    canvas.create_text(69.0, 1580.0, anchor="nw",
                       text="Churn Prediction Results",
                       fill="#000000", font=("Young Serif", 20))

    # Filter dropdown frame
    filter_container = Frame(canvas, bg='#D4C5D2')
    canvas.create_window(200, 1610, window=filter_container, anchor="nw")

    # Table frame - positioned lower with more height
    TABLE_X = 69
    TABLE_Y = 1660
    TABLE_W = 950
    TABLE_H = 380

    table_holder = Frame(canvas, bg="#FFFFFF", bd=1, relief="solid")
    canvas.create_window(TABLE_X, TABLE_Y, window=table_holder, anchor="nw",
                         width=TABLE_W, height=TABLE_H)

    # Build the prediction table with real data
    if data and data.get('df_result') is not None and HAS_TABLE_BUILDER:
        try:
            df_result = data['df_result']

            # Build filter dropdown
            current_df = [df_result]  # Use list to maintain reference in closure

            def on_filter_change(cluster_selection):
                """Callback when filter dropdown changes"""
                # Clear existing table
                for widget in table_holder.winfo_children():
                    widget.destroy()

                # Filter data
                if cluster_selection == "All Clusters":
                    filtered_df = df_result
                else:
                    # Extract cluster number from "Cluster X"
                    cluster_num = int(cluster_selection.split()[-1])
                    filtered_df = df_result[df_result['cluster'] == cluster_num]

                print(f"Filtered to {len(filtered_df)} records for {cluster_selection}")

                # Rebuild table with filtered data
                table_canvas = build_prediction_table(table_holder, TABLE_W, TABLE_H, filtered_df)
                table_canvas.pack(fill="both", expand=True)

            # Create filter dropdown
            filter_dropdown = build_cluster_filter_dropdown(
                filter_container,
                df_result,
                on_filter_change
            )
            filter_dropdown.pack()

            # Initial table build with all data
            table_canvas = build_prediction_table(table_holder, TABLE_W, TABLE_H, df_result)
            table_canvas.pack(fill="both", expand=True)

            print(f"✓ Prediction table created with {len(df_result)} records")
        except Exception as e:
            print(f"✗ Error creating prediction table: {e}")
            import traceback
            traceback.print_exc()
            # Show error message
            error_label = Label(table_holder, text=f"Error loading predictions: {str(e)}",
                                bg='#FFFFFF', fg='#E74C3C', font=("Crimson Pro", 12))
            error_label.pack(expand=True)
    elif HAS_TABLE_BUILDER:
        # Show table with sample data if no real data
        print("⚠ Building prediction table with sample data")
        table_canvas = build_prediction_table(table_holder, TABLE_W, TABLE_H, None)
        table_canvas.pack(fill="both", expand=True)
    else:
        # Show placeholder if table builder not available
        placeholder_label = Label(table_holder, text="Prediction table unavailable",
                                  bg='#FFFFFF', fg='#999999', font=("Crimson Pro", 12))
        placeholder_label.pack(expand=True)

    # Add a spacer at the bottom to ensure scrolling works
    canvas.create_text(69.0, TABLE_Y + TABLE_H + 100, anchor="nw",
                       text="", fill="#D4C5D2")

    # Set scrollregion - must be done AFTER all widgets are created
    def _sync_scrollregion(_=None):
        # Force update to ensure all widgets are rendered
        canvas.update_idletasks()

        # Get bounding box of all canvas items
        bbox = canvas.bbox("all")
        if bbox is None:
            bbox = (0, 0, canvas.winfo_width(), canvas.winfo_height())

        # Add extra padding at bottom to ensure last items are visible
        x0, y0, x1, y1 = bbox
        y1 = max(y1, TABLE_Y + TABLE_H + 150)  # Ensure we can scroll past table

        canvas.configure(scrollregion=(x0, y0, x1, y1))
        print(f"✓ Scroll region updated: {(x0, y0, x1, y1)}")

    canvas.bind("<Configure>", _sync_scrollregion)
    # Call multiple times to ensure it updates properly
    canvas.after(100, _sync_scrollregion)
    canvas.after(500, _sync_scrollregion)
    canvas.after(1000, _sync_scrollregion)

    return canvas
