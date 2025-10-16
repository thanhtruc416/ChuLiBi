# ui_Frame06_merged.py
# Merged: UI + KPI display + Dropdown + Charts + Standalone runner
# - KPI loading via get_all_kpis()
# - Data loading & preprocessing for charts (dfp) using Function.clean_dashboard
# - Robust standalone preview entrypoint

from pathlib import Path
from tkinter import Frame, Canvas, Button, PhotoImage

import sys
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- chart functions & preprocessing ---
from Function.Frame06_chart_dashboard import (
    _plot_pie_meal_share,
    _plot_line_orders_by_age,
    _plot_stacked_hist_delivery,
    _plot_bar_occupation_gender,
    _preprocess,
)

# --- dropdown & KPI functions ---
from Function.dropdown_profile import DropdownMenu
from Function.Frame06_kpi_dashboard import get_all_kpis

# ---------------- Paths & Fonts ----------------
OUTPUT_PATH = Path(__file__).parent                      # e.g., .../Frame/Frame06
ASSETS_PATH = OUTPUT_PATH / "assets_Frame06"

# Project root (adjust if your repo layout differs)
# Expected structure:
#   ROOT/
#     Dataset/Output/df_raw_dashboard.csv
#     Font/Crimson_Pro/static/CrimsonPro-Regular.ttf
ROOT = Path(__file__).resolve().parents[2]
font_path = ROOT / "Font" / "Crimson_Pro" / "static" / "CrimsonPro-Regular.ttf"
CSV_PATH = ROOT / "Dataset" / "Output" / "df_raw_dashboard.csv"

if font_path.exists():
    try:
        fm.fontManager.addfont(str(font_path))
        rcParams["font.family"] = "Crimson Pro"
    except Exception:
        # Fallback silently if font registration fails
        pass

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path

# ---------------- KPI Font Sizes (tweak here) ----------------
KPI_FONT_MAIN  = 16  # was 20
KPI_FONT_UNIT  = 8   # was 10
KPI_FONT_LABEL = 9   # was 11

# ---------------- Data load & preprocess ----------------
try:
    df = pd.read_csv(CSV_PATH)
    df.columns = [
        c.strip().replace(" ", "_").replace(".", "").replace("/", "_").lower()
        for c in df.columns
    ]
    dfp = _preprocess(df)
except Exception as e:
    print(f"[Frame06] Không thể đọc/tiền xử lý dữ liệu dashboard: {e}")
    dfp = None


class Frame06(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#D4C5D2")

        # --- Canvas ---
        canvas = Canvas(
            self,
            bg="#D4C5D2",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        canvas.place(x=0, y=0)

        # --- Load KPIs ---
        try:
            self.kpis = get_all_kpis()
        except Exception as e:
            print(f"[Frame06] Lỗi get_all_kpis(): {e}")
            self.kpis = {}
        print("KPIs loaded:", self.kpis)

        # --- Images (UI background) ---
        # If any image missing, skip it to avoid crashing
        def _safe_img(name):
            try:
                return PhotoImage(file=relative_to_assets(name))
            except Exception:
                return None

        def _add_img(img, x, y):
            if img is not None:
                canvas.create_image(x, y, image=img)

        self.image_1  = _safe_img("image_1.png");  _add_img(self.image_1, 168.0, 512.0)
        self.image_2  = _safe_img("image_2.png");  _add_img(self.image_2, 1.0,   245.0)
        self.image_3  = _safe_img("image_3.png");  _add_img(self.image_3, 889.0, 202.0)
        self.image_4  = _safe_img("image_4.png");  _add_img(self.image_4, 452.0, 220.0)
        self.image_5  = _safe_img("image_5.png");  _add_img(self.image_5, 577.0, 220.0)
        self.image_6  = _safe_img("image_6.png");  _add_img(self.image_6, 702.0, 220.0)
        self.image_7  = _safe_img("image_7.png");  _add_img(self.image_7, 827.0, 220.0)
        self.image_8  = _safe_img("image_8.png");  _add_img(self.image_8, 952.0, 220.0)
        self.image_9  = _safe_img("image_9.png");  _add_img(self.image_9, 1077.0, 220.0)
        self.image_10 = _safe_img("image_10.png"); _add_img(self.image_10, 1202.0, 220.0)
        self.image_11 = _safe_img("image_11.png"); _add_img(self.image_11, 1327.0, 220.0)
        self.image_12 = _safe_img("image_12.png"); _add_img(self.image_12, 426.0, 199.0)
        self.image_13 = _safe_img("image_13.png"); _add_img(self.image_13, 550.0, 199.0)
        self.image_14 = _safe_img("image_14.png"); _add_img(self.image_14, 676.0, 199.0)
        self.image_15 = _safe_img("image_15.png"); _add_img(self.image_15, 801.0, 195.0)
        self.image_16 = _safe_img("image_16.png"); _add_img(self.image_16, 633.0, 469.0)
        self.image_17 = _safe_img("image_17.png"); _add_img(self.image_17, 1151.0, 469.0)
        self.image_18 = _safe_img("image_18.png"); _add_img(self.image_18, 664.0, 802.0)
        self.image_19 = _safe_img("image_19.png"); _add_img(self.image_19, 1184.0, 802.0)
        self.image_20 = _safe_img("image_20.png"); _add_img(self.image_20, 888.0, 42.0)
        self.image_21 = _safe_img("image_21.png"); _add_img(self.image_21, 162.0, 101.0)
        self.image_22 = _safe_img("image_22.png"); _add_img(self.image_22, 924.0, 198.0)
        self.image_23 = _safe_img("image_23.png"); _add_img(self.image_23, 1050.0, 197.0)
        self.image_24 = _safe_img("image_24.png"); _add_img(self.image_24, 1181.0, 196.0)
        self.image_25 = _safe_img("image_25.png"); _add_img(self.image_25, 1303.0, 196.0)
        self.image_26 = _safe_img("image_26.png"); _add_img(self.image_26, 1296.0, 42.0)

        # --- Texts ---
        canvas.create_text(98.0,  927.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4", font=("Rubik Burned", 35 * -1))
        canvas.create_text(1142.0, 972.0, anchor="nw", text="Copyright by ChuLiBi", fill="#706093", font=("Young Serif", 22 * -1))
        canvas.create_text(399.0, 125.0, anchor="nw", text="KPI cards", fill="#706093", font=("Young Serif", 24 * -1))
        canvas.create_text(542.0, 133.0, anchor="nw", text="Top Metrics", fill="#B992B9", font=("Crimson Pro", 17 * -1))

        # --- KPI Display ---
        kpi_positions = [
            (452, 220),  # 1 - Total Customers
            (577, 220),  # 2 - Average Age
            (702, 220),  # 3 - Total Orders
            (827, 220),  # 4 - High Frequency Rate
            (952, 220),  # 5 - Average Order Value
            (1077, 220), # 6 - Average Delivery Time
            (1202, 220), # 7 - Restaurant Rating
            (1327, 220), # 8 - Delivery Rating
        ]
        kpi_labels = [
            "Total customers",
            "Average Age",
            "Total Orders",
            "High-Frequency\nCustomer Rate",
            "Average\nOrder Value",
            "Average\nDelivery Time",
            "Average\nRestaurant Rating",
            "Average\nDelivery Rating",
        ]
        kpi_values = [
            f"{self.kpis.get('total_customers', 'N/A')}\n(people)",
            f"{self.kpis.get('avg_age', 'N/A')}\n(years old)",
            f"{self.kpis.get('total_orders', 'N/A')}\n(orders)",
            f"{self.kpis.get('high_frequency_customer_rate', 'N/A')}\n(%)",
            f"${self.kpis.get('avg_order_value', 'N/A')}",
            f"{self.kpis.get('avg_delivery_time', 'N/A')}\n(hours)",
            f"{self.kpis.get('avg_restaurant_rating', 'N/A')}",
            f"{self.kpis.get('avg_delivery_rating', 'N/A')}",
        ]

        for i, (x, y) in enumerate(kpi_positions):
            # NOTE: dùng "\n" để tách dòng đúng
            parts = kpi_values[i].split("\n")
            main_value = parts[0]
            unit = parts[1] if len(parts) > 1 else ""
            # Main value (smaller font)
            canvas.create_text(
                x + 18, y - 20,
                text=main_value,
                fill="#706093",
                font=("Kodchasan Bold", KPI_FONT_MAIN),
                anchor="center"
            )
            # Unit (smaller font)
            if unit:
                canvas.create_text(
                    x + 18, y,
                    text=unit,
                    fill="#B992B9",
                    font=("Kodchasan", KPI_FONT_UNIT),
                    anchor="center"
                )
            # Label (smaller font)
            canvas.create_text(
                x, y + 20,
                text=kpi_labels[i],
                fill="#644E94",
                font=("Kodchasan", KPI_FONT_LABEL),
                anchor="center",
                justify="center"
            )

        # Additional section titles
        canvas.create_text(405.0, 322.0, anchor="nw", text="Bar chart", fill="#706093", font=("Young Serif", 27 * -1))
        canvas.create_text(544.0, 335.0, anchor="nw", text="Customer group with the highest spending", fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(937.0, 324.0, anchor="nw", text="Histogram chart", fill="#706093", font=("Young Serif", 27 * -1))
        canvas.create_text(1180.0, 324.0, anchor="nw", text="Delivery Performance \nImpact of Late Delivery", fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(407.0, 659.0, anchor="nw", text="Line chart", fill="#706093", font=("Young Serif", 27 * -1))
        canvas.create_text(560.0, 674.0, anchor="nw", text="Order Trends by Age Group", fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(999.0, 659.0, anchor="nw", text="Pie chart", fill="#706093", font=("Young Serif", 27 * -1))
        canvas.create_text(1135.0, 673.0, anchor="nw", text="Proportion of Meal Types", fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(373.0, 16.0, anchor="nw", text="Dashboard", fill="#000000", font=("Young Serif", 40 * -1))

        # --- Dropdown/Profile button ---
        self.button_Profile_image = _safe_img("button_Profile.png")
        self.dropdown = DropdownMenu(self)
        self.button_Profile = Button(self, image=self.button_Profile_image, borderwidth=0,
                                     highlightthickness=0, command=self.dropdown.show, relief="flat")
        self.button_Profile.place(x=1332.0, y=16.0, width=57.0, height=51.0)

        # --- Sidebar buttons ---
        def _btn_img(name): return _safe_img(name)
        self.button_Dashboard_image = _btn_img("button_Dashboard.png")
        self.button_Dashboard = Button(self, image=self.button_Dashboard_image, borderwidth=0,
                                       highlightthickness=0, command=lambda: print("button_Dashboard clicked"),
                                       relief="flat")
        self.button_Dashboard.place(x=20.0, y=201.0, width=317.0, height=87.0)

        self.button_Customer_analysis_image = _btn_img("button_Customer_analysis.png")
        self.button_Customer_analysis = Button(self, image=self.button_Customer_analysis_image, borderwidth=0,
                                               highlightthickness=0, command=lambda: self.controller.show_frame("Frame07"),
                                               relief="flat")
        self.button_Customer_analysis.place(x=0.0, y=302.0, width=337.0, height=77.0)

        self.button_Churn_image = _btn_img("button_Churn.png")
        self.button_Churn = Button(self, image=self.button_Churn_image, borderwidth=0,
                                   highlightthickness=0, command=lambda: self.controller.show_frame("Frame08"),
                                   relief="flat")
        self.button_Churn.place(x=0.0, y=381.0, width=336.0, height=86.0)

        self.button_Recommendation_image = _btn_img("button_Recommendation.png")
        self.button_Recommendation = Button(self, image=self.button_Recommendation_image, borderwidth=0,
                                            highlightthickness=0, command=lambda: self.controller.show_frame("Frame010"),
                                            relief="flat")
        self.button_Recommendation.place(x=0.0, y=500.0, width=337.0, height=200.0)

        self.button_Delivery_image = _btn_img("button_Delivery.png")
        self.button_Delivery = Button(self, image=self.button_Delivery_image, borderwidth=0,
                                      highlightthickness=0, command=lambda: self.controller.show_frame("Frame09"),
                                      relief="flat")
        self.button_Delivery.place(x=0.0, y=468.0, width=336.0, height=82.0)

        self.button_Report_image = _btn_img("button_Report.png")
        self.button_Report = Button(self, image=self.button_Report_image, borderwidth=0,
                                    highlightthickness=0, command=lambda: print("button_Report clicked"),
                                    relief="flat")
        self.button_Report.place(x=0.0, y=646.0, width=338.0, height=88.0)

        # ---------------- Chart containers ----------------
        frame_bar_chart  = Frame(canvas, bg="#FFFFFF")
        frame_histogram  = Frame(canvas, bg="#FFFFFF")
        frame_line_chart = Frame(canvas, bg="#FFFFFF")
        frame_pie_chart  = Frame(canvas, bg="#FFFFFF")

        canvas.create_window(410, 360, window=frame_bar_chart,  anchor="nw", width=480, height=270)
        canvas.create_window(930, 375, window=frame_histogram,  anchor="nw", width=450, height=220)
        canvas.create_window(420, 700, window=frame_line_chart, anchor="nw", width=520, height=260)
        canvas.create_window(980, 700, window=frame_pie_chart,  anchor="nw", width=400, height=220)

        # ---------------- Draw charts if data available ----------------
        if dfp is not None:
            # Bar chart
            fig1 = Figure(figsize=(4, 2), dpi=100)
            fig1.subplots_adjust(bottom=0.18, top=0.98)
            ax1 = fig1.add_subplot(111)
            _plot_bar_occupation_gender(ax1, dfp, mode="grouped")
            canvas1 = FigureCanvasTkAgg(fig1, master=frame_bar_chart)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)

            # Histogram
            fig2 = Figure(figsize=(8, 6), dpi=120)
            fig2.subplots_adjust(bottom=0.35, top=0.98)
            ax2 = fig2.add_subplot(111)
            _plot_stacked_hist_delivery(ax2, dfp, bin_width=10)
            canvas2 = FigureCanvasTkAgg(fig2, master=frame_histogram)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)

            # Line chart
            fig3 = Figure(figsize=(5, 3), dpi=100)
            fig3.subplots_adjust(bottom=0.3, top=0.88)
            ax3 = fig3.add_subplot(111)
            _plot_line_orders_by_age(ax3, dfp, bin_width=4)
            canvas3 = FigureCanvasTkAgg(fig3, master=frame_line_chart)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill="both", expand=True)

            # Pie chart
            fig4 = Figure(figsize=(6, 4), dpi=140)
            fig4.subplots_adjust(left=0.2, right=0.80, top=0.95, bottom=0.01)
            ax4 = fig4.add_subplot(111)
            _plot_pie_meal_share(ax4, dfp)
            canvas4 = FigureCanvasTkAgg(fig4, master=frame_pie_chart)
            canvas4.draw()
            canvas4.get_tk_widget().pack(fill="both", expand=True)
        else:
            print("[Frame06] Bỏ qua vẽ chart vì không có dữ liệu dfp.")

# -------------------------
# Standalone preview runner
# -------------------------
if __name__ == "__main__":
    # Ensure project imports work even when running this file directly
    try:
        from Function.dropdown_profile import DropdownMenu  # validate import
    except ModuleNotFoundError:
        ROOT_LOCAL = Path(__file__).parent.resolve()
        if str(ROOT_LOCAL) not in sys.path:
            sys.path.insert(0, str(ROOT_LOCAL))
        from Function.dropdown_profile import DropdownMenu

    import tkinter as tk

    class _DummyController:
        def show_frame(self, *args, **kwargs):
            pass

    root = tk.Tk()
    root.title("Dashboard - Frame06 (Merged)")
    root.geometry("1440x1024")
    root.configure(bg="#D4C5D2")
    app = Frame06(root, _DummyController())
    app.pack(fill="both", expand=True)
    # root.resizable(False, False)  # lock if desired
    root.mainloop()
