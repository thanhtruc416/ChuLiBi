# -*- coding: utf-8 -*-
# File: class10.py
# Converted from Tkinter Designer to Frame10 class + backend display
# Assets: ./assets_frame10/

from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox

from Function.dropdown_profile import DropdownMenu

# ==== Import backend recommendation ====
try:
    from Function.Frame10_Recommend import get_recommendation_data
except Exception as e:
    print("[WARN] Không thể import Function.Frame10_Recommend:", e)


class Frame10(tk.Frame):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent, bg="#ECE7EB")
        self.controller = controller
        self.lower()

        # ---- Đường dẫn tương đối ----
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("assets_frame10")

        def relative_to_assets(path: str) -> Path:
            return ASSETS_PATH / Path(path)

        # ---- Canvas chính ----
        self.canvas = Canvas(
            self,
            bg="#ECE7EB",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # ================= IMAGE & BUTTONS =================
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(889.0, 266.0, image=self.image_image_1)

        self.image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(168.0, 512.0, image=self.image_image_2)

        self.image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.image_3 = self.canvas.create_image(888.0, 42.0, image=self.image_image_3)

        self.image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        self.image_4 = self.canvas.create_image(580.0, 42.0, image=self.image_image_4)

        self.button_image_Profile = PhotoImage(file=relative_to_assets("button_Profile.png"))
        self.dropdown = DropdownMenu(self,controller=self.controller)
        self.button_Profile = Button(
            self,
            image=self.button_image_Profile,
            borderwidth=0,
            highlightthickness=0,
            command=self.dropdown.show,
            relief="flat"
        )
        self.button_Profile.place(x=1361.18, y=17.03, width=44.18, height=44.69)

        self.image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        self.image_5 = self.canvas.create_image(889.0, 527.0, image=self.image_image_5)

        self.button_image_search = PhotoImage(file=relative_to_assets("button_search.png"))
        self.button_search = Button(
            self,
            image=self.button_image_search,
            borderwidth=0,
            highlightthickness=0,
            command=self._filter_by_search,
            relief="flat"
        )
        self.button_search.place(x=1273.0, y=506.0, width=105.0, height=46.0)

        self.button_image_Noti = PhotoImage(file=relative_to_assets("button_Noti.png"))
        self.button_Noti = Button(
            self,
            image=self.button_image_Noti,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("Noti clicked"),
            relief="flat"
        )
        self.button_Noti.place(x=1292.0, y=17.0, width=49.0, height=49.0)

        # ================= TEXT =================
        self.canvas.create_text(
            375.0, 461.0, anchor="nw",
            text="Customer research",
            fill="#374A5A", font=("Young Serif", -24)
        )

        self.canvas.create_text(
            375.0, 563.0, anchor="nw",
            text="Search by Customer ID or Recommendation only",
            fill="#9282AA", font=("Young Serif", -15)
        )

        self.image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        self.image_6 = self.canvas.create_image(818.0, 527.0, image=self.image_image_6)

        # ================= ENTRY =================
        self.entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(809.0, 529.0, image=self.entry_image_1)
        self.entry_1 = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,font=("Crimson Pro", 15, "bold"))
        self.entry_1.place(x=386.0, y=518.0, width=846.0, height=20.0)

        # ================= FOOTER =================
        self.image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        self.image_7 = self.canvas.create_image(889.0, 807.0, image=self.image_image_7)

        self.canvas.create_text(
            375.0, 630.0, anchor="nw",
            text="Recommendations Table",
            fill="#374A5A", font=("Young Serif", -24)
        )

        self.canvas.create_text(
            99.0, 956.0, anchor="nw",
            text="ChuLiBi",
            fill="#FDE5F4", font=("Rubik Burned Regular", -35)
        )

        self.image_image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        self.image_8 = self.canvas.create_image(163.0, 130.0, image=self.image_image_8)

        # ================= SIDEBAR =================
        # (không đổi cấu trúc)
        self.button_image_CustomerAnalysis = PhotoImage(file=relative_to_assets("button_CustomerAnalysis.png"))
        self.button_CustomerAnalysis = Button(self, image=self.button_image_CustomerAnalysis, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame07"), relief="flat")
        self.button_CustomerAnalysis.place(x=0.0, y=305.0, width=337.0, height=78.0)

        self.button_image_Recommendation = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
        self.button_Recommendation = Button(self, image=self.button_image_Recommendation, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame10"), relief="flat")
        self.button_Recommendation.place(x=0.0, y=547.0, width=336.0, height=79.0)

        self.button_image_Dashboard = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
        self.button_Dashboard = Button(self, image=self.button_image_Dashboard, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame06"), relief="flat")
        self.button_Dashboard.place(x=1.0, y=222.0, width=336.0, height=88.0)

        self.button_image_El = PhotoImage(file=relative_to_assets("button_El.png"))
        self.button_El = Button(self, image=self.button_image_El, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame09_EL"), relief="flat")
        self.button_El.place(x=1.0, y=469.0, width=336.0, height=78.0)

        self.button_image_Churn = PhotoImage(file=relative_to_assets("button_Churn.png"))
        self.button_Churn = Button(self, image=self.button_image_Churn, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame08"), relief="flat")
        self.button_Churn.place(x=1.0, y=383.0, width=336.0, height=86.0)

        self.button_image_PredictCustomer = PhotoImage(file=relative_to_assets("button_PredictCustomer.png"))
        self.button_PredictCustomer = Button(self, image=self.button_image_PredictCustomer, borderwidth=0,
            highlightthickness=0, command=lambda: self.controller.show_frame("Frame11"), relief="flat")
        self.button_PredictCustomer.place(x=0.0, y=626.0, width=336.0, height=75.0)

        # ================= KPI (có hình + text phần trăm) =================
        # ================= KPI =================
        # ---- Vẽ 8 ảnh khung KPI trước ----
        self.image_service = PhotoImage(file=relative_to_assets("image_9.png"))
        self.canvas.create_image(480.0, 213.0, image=self.image_service)

        self.image_sla = PhotoImage(file=relative_to_assets("image_11.png"))
        self.canvas.create_image(751.0, 213.0, image=self.image_sla)

        self.image_quality = PhotoImage(file=relative_to_assets("image_13.png"))
        self.canvas.create_image(1020.0, 213.0, image=self.image_quality)

        self.image_coupon = PhotoImage(file=relative_to_assets("image_15.png"))
        self.canvas.create_image(1291.0, 213.0, image=self.image_coupon)

        self.image_loyalty = PhotoImage(file=relative_to_assets("image_10.png"))
        self.canvas.create_image(480.0, 351.0, image=self.image_loyalty)

        self.image_care = PhotoImage(file=relative_to_assets("image_12.png"))
        self.canvas.create_image(751.0, 351.0, image=self.image_care)

        self.image_remind = PhotoImage(file=relative_to_assets("image_14.png"))
        self.canvas.create_image(1020.0, 351.0, image=self.image_remind)

        self.image_edu = PhotoImage(file=relative_to_assets("image_16.png"))
        self.canvas.create_image(1291.0, 351.0, image=self.image_edu)

        # ---- Sau đó vẽ chữ nhãn KPI ----
        self.canvas.create_text(375.0, 115.0, anchor="nw", text="Service Package",
                                fill="#374A5A", font=("Young Serif", -25))
        self.canvas.create_text(441.0, 171.0, anchor="nw", text="SLA_UP",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(665.0, 171.0, anchor="nw", text="QUALITY SWITCH",
                                fill="#374A5A", font=("Young Serif", -16))
        self.canvas.create_text(976.0, 171.0, anchor="nw", text="COUPON",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(1247.0, 171.0, anchor="nw", text="LOYALTY",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(423.0, 312.0, anchor="nw", text="CARE CALL",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(688.0, 312.0, anchor="nw", text="REMIND APP",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(949.0, 312.0, anchor="nw", text="EDU CONTENT",
                                fill="#374A5A", font=("Young Serif", -18))
        self.canvas.create_text(1234.0, 312.0, anchor="nw", text="NO ACTION",
                                fill="#374A5A", font=("Young Serif", -18))

        # ---- Vẽ số KPI (phần trăm, auto canh giữa khung) ----
        # Lấy đúng vị trí giữa các ảnh KPI
        kpi_centers = {
            "SLA_UP": (480.0, 213.0),
            "QUALITY_SWITCH": (751.0, 213.0),
            "COUPON10": (1020.0, 213.0),
            "LOYALTY": (1291.0, 213.0),
            "CARE_CALL": (480.0, 351.0),
            "REMIND_APP": (751.0, 351.0),
            "EDU_CONTENT": (1020.0, 351.0),
            "NO_ACTION": (1291.0, 351.0),
        }

        self.kpi_labels = {}
        for key, (cx, cy) in kpi_centers.items():
            self.kpi_labels[key] = self.canvas.create_text(
                cx, cy +12,  # nhích lên 8px
                text="0%",
                fill="#794679",
                font=("Kodchasan Regular", -36),
                anchor="center",  # canh giữa tuyệt đối
            )

        # ==== Table holder ====
        self.table_holder = tk.Frame(self, bg="#FFFFFF")
        self.table_holder.place(x=380, y=670, width=980, height=300)
        self.table_widget = None

    # ===================================================
    # ==== BACKEND & KPI UPDATE ====
    # ===================================================
    def _load_recommendations(self):
        """Load dữ liệu gốc từ backend và hiển thị bảng ban đầu"""
        try:
            df, df_rec, thr = get_recommendation_data()
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi tải dữ liệu: {e}")
            return

        if df_rec is None or df_rec.empty:
            messagebox.showinfo("Thông báo", "Không có dữ liệu khuyến nghị.")
            return

        # Lưu dữ liệu gốc
        self.df_rec_raw = df_rec.copy()  # dữ liệu gốc
        self.df_rec_original = df_rec.copy()
        self._render_table(df_rec)
        self._update_kpi(df_rec)

    def _filter_by_search(self):
        """Lọc theo Customer_ID hoặc Recommendation"""
        keyword = self.entry_1.get().strip().lower()
        self.active_mode = "search"

        if not hasattr(self, "df_rec_raw") or self.df_rec_raw is None:
            self._load_recommendations()
            return

        df_full = self.df_rec_raw.copy()  # dữ liệu gốc

        if not keyword:
            messagebox.showinfo("Thông báo", "Vui lòng nhập Customer ID hoặc gói cần tìm.")
            return

        df_filtered = df_full[
            df_full["Customer_ID"].astype(str).str.lower().str.contains(keyword)
            | df_full.get("action_name", "").astype(str).str.lower().str.contains(keyword)
            ]

        if df_filtered.empty:
            messagebox.showinfo("Kết quả tìm kiếm", f"Không tìm thấy khách hàng hoặc gói phù hợp với '{keyword}'.")
            return

        # ✅ Hiển thị kết quả search, KHÔNG render lại dropdown
        self._render_table(df_filtered, show_filter=True)

        print(f"[SEARCH] Found {len(df_filtered)} rows matching '{keyword}'")

    def _render_table(self, df_rec, show_filter=True):
        import tkinter as tk
        from tkinter import ttk
        import pandas as pd

        # Xóa bảng cũ
        for w in self.table_holder.winfo_children():
            w.destroy()

        HEADER_BG = "#B79AC8"
        ROW_EVEN = "#FFFFFF"
        ROW_ODD = "#F7F4F7"
        TEXT = "#2E2E2E"

        # ==== Chuẩn hoá dữ liệu hiển thị ban đầu ====
        df_show = df_rec.copy()
        df_show["ID"] = df_show.get("Customer_ID", "")
        if "priority_score" in df_show.columns:
            df_show["Expected Loss"] = (df_show["priority_score"].astype(float) * 100).round(1).astype(str) + "%"
        else:
            df_show["Expected Loss"] = "0.0%"
        df_show["Cluster"] = "Cluster 1"
        df_show["Recommendation"] = df_show.apply(
            lambda r: f"{r.get('action_id', '')} – {r.get('action_name', '')}", axis=1
        )
        df_show = df_show[["ID", "Cluster", "Expected Loss", "Recommendation"]]
        self.df_full = df_show.copy()

        # ==== Khung filter (dropdown) ====
        if show_filter:
            filter_outer = tk.Frame(self.table_holder, bg="#ECE7EB")
            filter_outer.pack(fill="x", padx=0, pady=(0, 4))

            filter_inner = tk.Frame(filter_outer, bg="#FFFFFF", bd=0, relief="flat", height=36)
            filter_inner.pack(fill="x")
            filter_inner.pack_propagate(False)

            right_wrap = tk.Frame(filter_inner, bg="#FFFFFF")
            right_wrap.place(relx=1.0, rely=0.5, anchor="e", x=-20)

            lbl = tk.Label(
                right_wrap,
                text="Filter by Recommendation:",
                bg="#FFFFFF",
                fg="#374A5A",
                font=("Crimson Pro", 12, "bold")
            )
            lbl.pack(side="left", padx=(0, 6))

            base_df = getattr(self, "df_rec_raw", df_rec)
            rec_packages = sorted(base_df["action_id"].dropna().unique().tolist())
            rec_opts = ["All Packages"] + rec_packages
            self.selected_package = tk.StringVar(value="All Packages")

            self.cmb_package = ttk.Combobox(
                right_wrap,
                values=rec_opts,
                textvariable=self.selected_package,
                state="readonly",
                width=25,
                font=("Crimson Pro", 12)
            )
            self.cmb_package.pack(side="left")

        # ==== Khung table chính ====
        table_frame = tk.Frame(self.table_holder, bg="#FFFFFF")
        table_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(table_frame, bg="#FFFFFF", highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True, side="left")

        scrollbar = tk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas, bg="#FFFFFF")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _label(parent, text, bg, fg=TEXT, bold=False, anchor="w", padx=12, pady=8, width=None):
            font = ("Crimson Pro", 12, "bold" if bold else "normal")
            lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                           anchor=anchor, padx=padx, pady=pady, justify="left")
            if width:
                lbl.config(width=int(width / 8))
            return lbl

        # Header
        header = tk.Frame(inner, bg=HEADER_BG)
        header.pack(fill="x")
        COLUMNS = [
            ("ID", 100),
            ("Cluster", 120),
            ("Expected Loss", 160),
            ("Recommendation", 580),
        ]
        for col, col_w in COLUMNS:
            lbl = _label(header, col, HEADER_BG, fg="#FFFFFF", bold=True)
            lbl.pack(side="left")
            lbl.config(width=int(col_w / 8))

        # Row function
        def _add_row(values, index):
            bg = ROW_EVEN if index % 2 == 0 else ROW_ODD
            row = tk.Frame(inner, bg=bg)
            row.pack(fill="x")
            for (col, col_w), val in zip(COLUMNS, values):
                anchor = "center" if col == "Expected Loss" else "w"
                lbl = _label(row, str(val), bg, width=col_w, anchor=anchor)
                lbl.pack(side="left")

        # ==== Filter dropdown logic ====
        def render_filtered_table(sel_pkg):
            base_df = getattr(self, "df_rec_original", None)
            if base_df is None:
                return

            # Clear search box khi chọn dropdown
            self.entry_1.delete(0, tk.END)

            data = base_df.copy()
            data["ID"] = data.get("Customer_ID", "")
            if "priority_score" in data.columns:
                data["Expected Loss"] = (data["priority_score"].astype(float) * 100).round(1).astype(str) + "%"
            else:
                data["Expected Loss"] = "0.0%"
            data["Cluster"] = "Cluster 1"
            data["Recommendation"] = data.apply(
                lambda r: f"{r.get('action_id', '')} – {r.get('action_name', '')}", axis=1
            )
            data = data[["ID", "Cluster", "Expected Loss", "Recommendation", "action_id"]]

            if sel_pkg != "All Packages":
                data = data[data["action_id"].astype(str).str.strip() == sel_pkg]

            # clear old rows
            for w in inner.winfo_children():
                if w != header:
                    w.destroy()

            for i, row in enumerate(data.itertuples(index=False), start=1):
                _add_row(list(row), i)

            _sync_scroll()
            print(f"[UI] Filter dropdown applied: {sel_pkg} → {len(data)} rows")

        # ==== Scroll sync ====
        def _sync_scroll(_=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _sync_scroll)

        # ==== Hiển thị dữ liệu ban đầu (theo dataset truyền vào) ====
        for i, row in enumerate(df_show.itertuples(index=False), start=1):
            _add_row(list(row), i)
        _sync_scroll()

        # ==== Bind dropdown ====
        if show_filter and hasattr(self, "cmb_package"):
            self.cmb_package.bind("<<ComboboxSelected>>",
                                  lambda e: render_filtered_table(self.selected_package.get()))
            # ⚡ Nếu đây là kết quả search, KHÔNG render lại full
            if getattr(self, "active_mode", "") != "search":
                render_filtered_table("All Packages")

    def _update_kpi(self, df_rec):
        counts = df_rec["action_id"].value_counts(normalize=True).mul(100).to_dict()
        for aid, item in self.kpi_labels.items():
            pct = counts.get(aid, 0)
            self.canvas.itemconfigure(item, text=f"{pct:.1f}%")

    # ===================================================
    def _on_profile_clicked(self):
        messagebox.showinfo("Profile", "Profile clicked")

    def on_show(self):
        self._load_recommendations()


# =========================
# Test độc lập
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Demo – Frame10")
    root.geometry("1440x1024")
    root.configure(bg="#ECE7EB")

    # Mock controller
    class MockController:
        def show_frame(self, name, **kwargs):
            print(f"[MOCK] show_frame({name}, {kwargs})")
        def get_current_user(self):
            return {"username": "test_user"}
        def clear_current_user(self):
            print("[MOCK] clear_current_user()")

    mock_controller = MockController()
    app = Frame10(root, controller=mock_controller)
    app.pack(fill="both", expand=True)
    app.on_show()
    root.mainloop()

