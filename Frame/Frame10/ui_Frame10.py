from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox

from Function.dropdown_profile import DropdownMenu
from QMess.Qmess_calling import Qmess
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
        self.image_2 = self.canvas.create_image(169.0, 512.0, image=self.image_image_2)

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
        self.button_search.place(x=1282.0, y=506.0, width=105.0, height=46.0)

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
            fill="#706093", font=("Crimson Pro Bold", -30)
        )

        self.canvas.create_text(
            375.0, 563.0, anchor="nw",
            text="Search by Customer ID or Recommendation only",
            fill="#9282AA", font=("Crimson Pro Bold", -20)
        )

        self.image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        self.image_6 = self.canvas.create_image(818.0, 527.0, image=self.image_image_6)

        # ================= ENTRY =================
        self.entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(809.0, 529.0, image=self.entry_image_1)
        self.entry_1 = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,font=("Crimson Pro", 15, "bold"))
        self.entry_1.place(x=386.0, y=518.0, width=846.0, height=20.0)
        self.entry_1.bind("<Return>", lambda event: self.button_search.invoke())

        # ================= FOOTER =================
        self.image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        self.image_7 = self.canvas.create_image(889.0, 807.0, image=self.image_image_7)

        self.canvas.create_text(
            385.0, 635.0, anchor="nw",
            text="Recommendations Table",
            fill="#706093", font=("Crimson Pro Bold", -33)
        )

        self.canvas.create_text(
            99.0, 956.0, anchor="nw",
            text="ChuLiBi",
            fill="#FDE5F4", font=("Rubik Burned Regular", -35)
        )

        self.image_image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        self.image_8 = self.canvas.create_image( 162.0, 101.0, image=self.image_image_8)

        # --- Sidebar buttons ---
        # Dashboard
        self.button_image_Dashboard = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
        self.button_Dashboard = Button(
            self,
            image=self.button_image_Dashboard,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame06"),
            relief="flat"
        )
        self.button_Dashboard.place(
            x=0.0,
            y=223.0,
            width=338.0,
            height=81.0
        )

        # Customer Analysis
        self.button_image_CustomerAnalysis = PhotoImage(file=relative_to_assets("button_CustomerAnalysis.png"))
        self.button_CustomerAnalysis = Button(
            self,
            image=self.button_image_CustomerAnalysis,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame07"),
            relief="flat"
        )
        self.button_CustomerAnalysis.place(
            x=0.0,
            y=304.0,
            width=338.0,
            height=81.0
        )

        # Churn
        self.button_image_Churn = PhotoImage(file=relative_to_assets("button_Churn.png"))
        self.button_Churn = Button(
            self,
            image=self.button_image_Churn,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame08"),
            relief="flat"
        )
        self.button_Churn.place(
            x=0.0,
            y=385.0,
            width=338.0,
            height=81.0
        )

        # Expected Loss
        self.button_image_El = PhotoImage(file=relative_to_assets("button_El.png"))
        self.button_El = Button(
            self,
            image=self.button_image_El,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame09_EL"),
            relief="flat"
        )
        self.button_El.place(
            x=0.0,
            y=466.0,
            width=338.0,
            height=81.0
        )

        # Recommendation
        self.button_image_Recommendation = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
        self.button_Recommendation = Button(
            self,
            image=self.button_image_Recommendation,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame10"),
            relief="flat"
        )
        self.button_Recommendation.place(
            x=0.0,
            y=547.0,
            width=338.0,
            height=81.0
        )

        # Predict Customer
        self.button_image_PredictCustomer = PhotoImage(file=relative_to_assets("button_PredictCustomer.png"))
        self.button_PredictCustomer = Button(
            self,
            image=self.button_image_PredictCustomer,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame11"),
            relief="flat"
        )
        self.button_PredictCustomer.place(
            x=0.0,
            y=628.0,
            width=338.0,
            height=81.0
        )
        # ================= KPI (có hình + text phần trăm) =================
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
        self.canvas.create_text(375.0, 110.0, anchor="nw", text="Service Package",
                                fill="#706093", font=("Crimson Pro Bold", -32))
        self.canvas.create_text(441.0, 171.0, anchor="nw", text="SLA_UP",fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(665.0, 171.0, anchor="nw", text="QUALITY SWITCH",fill="#374A5A", font=("Crimson Pro Bold", -22))
        self.canvas.create_text(976.0, 171.0, anchor="nw", text="COUPON",fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(1247.0, 171.0, anchor="nw", text="LOYALTY",fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(423.0, 312.0, anchor="nw", text="CARE CALL",fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(688.0, 312.0, anchor="nw", text="REMIND APP", fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(949.0, 312.0, anchor="nw", text="EDU CONTENT",fill="#374A5A", font=("Crimson Pro Bold", -23))
        self.canvas.create_text(1234.0, 312.0, anchor="nw", text="NO ACTION",fill="#374A5A", font=("Crimson Pro Bold", -23))

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
        self.table_holder.place(x=365, y=670, width=1030, height=300)
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
        # Merge cluster từ df_cluster_full.csv (df) vào df_rec
        # ================= DEBUG CỘT TRƯỚC KHI MERGE =================
        print("[DEBUG] df_cluster_full columns:", df.columns.tolist())
        print("[DEBUG] df_rec columns:", df_rec.columns.tolist())

        # ================= MERGE CLUSTER CHÍNH XÁC =================
        try:
            # Trường hợp 1: file cluster dùng Customer_ID_STD
            if "cluster" in df.columns and "Customer_ID_STD" in df.columns and "Customer_ID" in df_rec.columns:
                print("[INFO] Dùng Customer_ID_STD để merge cluster")
                df_rec = df_rec.merge(
                    df[["Customer_ID_STD", "cluster"]],
                    left_on="Customer_ID",
                    right_on="Customer_ID_STD",
                    how="left"
                )
                df_rec.drop(columns=["Customer_ID_STD"], inplace=True)

            # Trường hợp 2: file cluster dùng Customer_ID
            elif "cluster" in df.columns and "Customer_ID" in df.columns and "Customer_ID" in df_rec.columns:
                print("[INFO] Dùng Customer_ID để merge cluster")
                df_rec = df_rec.merge(df[["Customer_ID", "cluster"]], on="Customer_ID", how="left")

            # Trường hợp 3: cột bị viết hoa (Cluster)
            elif "Cluster" in df.columns and "Customer_ID" in df.columns:
                print("[INFO] Dùng cột Cluster (viết hoa) để merge")
                df_rec = df_rec.merge(df[["Customer_ID", "Cluster"]], on="Customer_ID", how="left")

            else:
                print("[WARN] Không tìm thấy cột hợp lệ để merge Cluster")
        except Exception as e:
            print(f"[ERROR] Merge cluster thất bại: {e}")

        # ================= SAU KHI MERGE =================
        # Đảm bảo đồng bộ tên cột
        if "Cluster" not in df_rec.columns and "cluster" in df_rec.columns:
            df_rec.rename(columns={"cluster": "Cluster"}, inplace=True)

        # In ra xem có merge thành công không
        print(f"[CHECK] Cluster merged? {df_rec['Cluster'].notna().sum()} / {len(df_rec)} rows có Cluster")

        # ================= LƯU DỮ LIỆU & HIỂN THỊ =================
        self.df_rec_raw = df_rec.copy()
        self.df_rec_original = df_rec.copy()
        self._render_table(df_rec)
        self._update_kpi(df_rec)

    def _filter_by_search(self):
        keyword = self.entry_1.get().strip().lower()
        self.active_mode = "search"

        if not hasattr(self, "df_rec_raw") or self.df_rec_raw is None:
            self._load_recommendations()
            return

        df_full = self.df_rec_raw.copy()

        if not keyword:
            Qmess.popup_29(parent=self,
                        title="Warning",
                        subtitle="Please enter Customer ID or package to search.")
            return

        df_filtered = df_full[
            df_full["Customer_ID"].astype(str).str.lower().str.contains(keyword)
            | df_full.get("action_name", df_full.get("Action_Name", "")).astype(str).str.lower().str.contains(keyword)
            ]

        if df_filtered.empty:
            Qmess.popup_23(parent=self,
                        title="Warning",
                        subtitle=f"Do not found customer or package matching '{keyword}'.")
            return

        self.df_search_result = df_filtered.copy()
        self._render_table(df_filtered, show_filter=True)

        print(f"[SEARCH] Found {len(df_filtered)} rows matching '{keyword}'")

    def _render_table(self, df_rec, show_filter=True):
        import tkinter as tk
        from tkinter import ttk
        import pandas as pd

        # --- preserve raw/original
        if not hasattr(self, "df_rec_raw") or getattr(self, "df_rec_raw") is None:
            self.df_rec_raw = df_rec.copy()

        # chỉ cập nhật df_rec_original nếu KHÔNG phải search mode
        mode = getattr(self, "active_mode", "")
        if mode != "search":
            self.df_rec_original = df_rec.copy()

        for w in getattr(self, "table_holder").winfo_children():
            w.destroy()
        # --- Dọn sạch toàn bộ table cũ (canvas, frame, scroll...) ---
        if hasattr(self, "table_holder") and self.table_holder is not None:
            for widget in self.table_holder.winfo_children():
                widget.destroy()
                print("[DEBUG] Table inner recreated, rows:", len(df_rec))

        print("[DEBUG] Table inner recreated, rows:", len(df_rec))

        HEADER_BG = "#B79AC8"
        ROW_EVEN = "#FFFFFF"
        ROW_ODD = "#F7F4F7"
        TEXT = "#2E2E2E"

        df_show = df_rec.copy()
        df_show["ID"] = df_show.get("Customer_ID", "")
        if "priority_score" in df_show.columns:
            df_show["Expected Loss"] = (df_show["priority_score"].astype(float) * 100).round(1).astype(str) + "%"
        else:
            df_show["Expected Loss"] = "0.0%"

        # --- Đảm bảo cột Cluster tồn tại đúng ---
        if "Cluster" in df_rec.columns:
            df_show["Cluster"] = df_rec["Cluster"]
        elif "cluster" in df_rec.columns:
            df_show["Cluster"] = df_rec["cluster"]
        else:
            df_show["Cluster"] = ""

        # --- Format lại thành "Cluster 1", "Cluster 2" ---
        def format_cluster_display(v):
            try:
                return f"Cluster {int(float(v)) + 1}"
            except:
                return ""

        df_show["Cluster"] = df_show["Cluster"].apply(format_cluster_display)

        df_show["Recommendation"] = df_show.apply(
            lambda r: f"{r.get('action_id', '')} – {r.get('action_name', '')}", axis=1
        )

        df_show = df_show[["ID", "Cluster", "Expected Loss", "Recommendation"]]
        self.df_full = df_show.copy()

        # DỊCH FILTER & TABLE SANG TRÁI
        if show_filter:
            filter_outer = tk.Frame(self.table_holder, bg="#ECE7EB")
            filter_outer.pack(fill="x", padx=(20, 0), pady=(0, 4))

            filter_inner = tk.Frame(filter_outer, bg="#FFFFFF", height=36)
            filter_inner.pack(fill="x")
            filter_inner.pack_propagate(False)

            right_wrap = tk.Frame(filter_inner, bg="#FFFFFF")
            right_wrap.place(relx=1.0, rely=0.5, anchor="e", x=4, y=-2)

            lbl = tk.Label(right_wrap, text="Filter by Recommendation:",
                           font=("Crimson Pro", 15, "bold"), bg="#FFFFFF", fg="#374A5A")
            lbl.pack(side="left", padx=(0, 6))

            all_packages_full = ["SLA_UP", "QUALITY_SWITCH", "COUPON", "LOYALTY",
                                 "CARE_CALL", "REMIND_APP", "EDU_CONTENT", "NO_ACTION"]

            base_df = df_rec.copy()
            print(f"[DEBUG] Rendering table with {len(base_df)} rows (input df_rec={len(df_rec)})")

            existing = (
                base_df.get("action_id", pd.Series(dtype=str))
                .astype(str).str.strip()
                .replace(["", "nan", "None", "NaN"], None).dropna().unique().tolist()
            )

            rec_opts = ["All Packages"] + all_packages_full
            self.selected_package = tk.StringVar(value="All Packages")

            self.btn_package = tk.Label(
                right_wrap,
                text=f"{self.selected_package.get()} ▼",
                font=("Crimson Pro", 12),
                bg="#FFFFFF",
                fg="#374A5A",
                relief="solid", bd=1, padx=10, pady=4,
                highlightbackground="#C2A8C2",
                highlightthickness=1,
                cursor="hand2"
            )
            self.btn_package.pack(side="left", padx=(0, 10))
            self.btn_package.bind("<Button-1>",
                                  lambda e: self.show_dropdown_custom(right_wrap, self.selected_package, rec_opts))

        # ==== TABLE FRAME (dịch trái)
        table_frame = tk.Frame(self.table_holder, bg="#FFFFFF")
        table_frame.pack(fill="both", expand=True, padx=(20, 0))

        canvas = tk.Canvas(table_frame, bg="#FFFFFF", highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True, side="left")

        # === Custom pastel scrollbar (giống Frame09) ===
        scroll_canvas = tk.Canvas(
            table_frame,
            width=6,
            bg="#F7F4F7",
            highlightthickness=0,
            bd=0
        )
        scroll_canvas.pack(side="right", fill="y", padx=(2, 6), pady=6)

        scroll_thumb = scroll_canvas.create_rectangle(
            1, 0, 5, 40,
            outline="",
            fill="#C9C4CE",
            width=0
        )

        self._scroll_drag_start = None
        self._scroll_start_pos = None

        def update_thumb(first, last):
            """Cập nhật thumb khi Treeview (ở đây là Canvas) cuộn"""
            height = scroll_canvas.winfo_height()
            first, last = float(first), float(last)
            thumb_len = max(30, (last - first) * height)
            y1 = first * height
            y2 = y1 + thumb_len
            scroll_canvas.coords(scroll_thumb, 1, y1 + 3, 5, y2 - 3)

        def scroll(*args):
            """Cuộn canvas + cập nhật thumb"""
            canvas.yview(*args)
            update_thumb(*canvas.yview())

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            update_thumb(*canvas.yview())
            return "break"

        # === Kéo thả thanh scroll bằng chuột ===
        def on_thumb_press(event):
            self._scroll_drag_start = event.y
            self._scroll_start_pos = canvas.yview()[0]

        def on_thumb_drag(event):
            if self._scroll_drag_start is None:
                return
            dy = event.y - self._scroll_drag_start
            height = scroll_canvas.winfo_height()
            first, last = canvas.yview()
            visible = last - first
            thumb_len = max(30, visible * height)
            scrollable_height = height - thumb_len
            if scrollable_height <= 0:
                return  # không cuộn được nếu không có nội dung
            delta_fraction = dy / scrollable_height * (1 - visible)

            new_first = max(0, min(1 - visible, self._scroll_start_pos + delta_fraction))
            canvas.yview_moveto(new_first)
            update_thumb(*canvas.yview())

        def on_thumb_release(event):
            self._scroll_drag_start = None
            self._scroll_start_pos = None

        # === Bind events ===
        scroll_canvas.tag_bind(scroll_thumb, "<ButtonPress-1>", on_thumb_press)
        scroll_canvas.tag_bind(scroll_thumb, "<B1-Motion>", on_thumb_drag)
        scroll_canvas.tag_bind(scroll_thumb, "<ButtonRelease-1>", on_thumb_release)
        canvas.configure(yscrollcommand=update_thumb)
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        inner = tk.Frame(canvas, bg="#FFFFFF")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        COLUMNS = [
            ("Customer ID", 120),
            ("Cluster", 140),
            ("Expected Loss", 160),
            ("Recommendation", 600),
        ]

        # ==== HEADER (dùng grid, không pack) ====
        header = tk.Frame(inner, bg=HEADER_BG)
        header.pack(fill="x")

        for j, (col, col_w) in enumerate(COLUMNS):
            anchor = "c" if col == "Recommendation" else "center"
            lbl = tk.Label(
                header,
                text=col,
                bg=HEADER_BG,
                fg="#FFFFFF",
                font=("Crimson Pro", 15, "bold"),
                anchor=anchor,
                pady=8
            )
            lbl.grid(row=0, column=j, sticky="nsew", padx=(2, 0))
            header.grid_columnconfigure(j, minsize=col_w)

        # ==== ROW CHUẨN, KHÔNG BỊ LỆCH, GRID FULL CELL ====
        def _add_row(values, index):
            bg = ROW_EVEN if index % 2 == 0 else ROW_ODD

            row = tk.Frame(inner, bg=bg, height=42)
            row.pack(fill="x")
            row.pack_propagate(False)

            for j, ((col, col_w), val) in enumerate(zip(COLUMNS, values)):
                cell = tk.Frame(row, bg=bg, width=col_w, height=42)
                cell.pack(side="left", fill="y")
                cell.pack_propagate(False)

                lbl = tk.Label(
                    cell,
                    text=str(val),
                    bg=bg,
                    fg=TEXT,
                    font=("Crimson Pro", 13),
                    anchor="w" if col == "Recommendation" else "center"
                )
                lbl.place(
                    x=80 if col == "Recommendation" else 0,
                    relx=0.5 if col != "Recommendation" else 0.0,
                    rely=0.5,
                    anchor="center" if col != "Recommendation" else "w"
                )

        # --- Hiển thị các dòng dữ liệu ---
        for i, row in enumerate(df_show.itertuples(index=False), start=1):
            vals = list(row)
            _add_row(vals, i)
        print(f"[DEBUG] Rendered {len(df_show)} rows into table.")

        # ==== Filter dropdown render logic (dùng để re-render khi chọn filter) ====
        def render_filtered_table(sel_pkg):
            # Luôn lọc từ dữ liệu gốc, KHÔNG phụ thuộc vào kết quả search
            base_df = getattr(self, "df_rec_original", None)
            print(f"[DEBUG] Filter luôn dùng dataset gốc ({len(base_df) if base_df is not None else 0} rows)")

            if base_df is None or base_df.empty:
                print("[WARN] Không có dữ liệu để filter")
                return

            # Clear search box nếu có
            if hasattr(self, "entry_1") and getattr(self, "entry_1") is not None:
                try:
                    self.entry_1.delete(0, "end")
                except Exception:
                    pass

            data = base_df.copy()
            data["ID"] = data.get("Customer_ID", "")
            if "priority_score" in data.columns:
                data["Expected Loss"] = (data["priority_score"].astype(float) * 100).round(1).astype(str) + "%"
            else:
                data["Expected Loss"] = "0.0%"

            if "Cluster" not in data.columns and "cluster" in data.columns:
                data.rename(columns={"cluster": "Cluster"}, inplace=True)
            elif "Cluster" not in data.columns:
                data["Cluster"] = ""

            data["Cluster"] = data["Cluster"].apply(format_cluster_display)

            data["Recommendation"] = data.apply(
                lambda r: f"{r.get('action_id', '')} – {r.get('action_name', '')}", axis=1
            )
            # thêm action_id để filter
            data = data[["ID", "Cluster", "Expected Loss", "Recommendation", "action_id"]]

            if sel_pkg != "All Packages":
                data = data[data["action_id"].astype(str).str.strip() == sel_pkg]

            # clear old rows (giữ header)
            for w in inner.winfo_children():
                if w is not header:
                    w.destroy()

            for i, row in enumerate(data.itertuples(index=False), start=1):
                # loại bỏ field action_id khi render (itertuples trả cả action_id)
                vals = list(row)[:4]
                _add_row(vals, i)

            _sync_scroll()
            print(f"[UI] Filter dropdown applied: {sel_pkg} → {len(data)} rows")

        # Thêm dòng này:
        self.render_filtered_table = render_filtered_table

        # ==== Scroll sync ====
        def _sync_scroll(_=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _sync_scroll)

        # Fix: chỉ render lại bảng All Packages khi KHÔNG ở search mode
        mode = getattr(self, "active_mode", "")
        if mode == "search":
            print("[DEBUG] Đang ở chế độ search — vẫn hiển thị filter đầy đủ, KHÔNG tự động lọc lại.")
            _sync_scroll()
            # KHÔNG return nữa — để filter dropdown được khởi tạo bình thường

        else:
            sel = getattr(self, "selected_package", tk.StringVar(value="All Packages")).get()
            render_filtered_table(sel)

    def show_dropdown_custom(self, parent_widget, var, options):
        """Dropdown tùy chỉnh — style pastel tím, font Crimson Pro."""
        import tkinter as tk

        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.config(bg="#FFFFFF", bd=1, highlightthickness=1, highlightbackground="#B992B9")

        # Lấy vị trí hiển thị popup từ btn_package nếu có
        btn = getattr(self, "btn_package", None)
        if btn:
            x = btn.winfo_rootx()
            y = btn.winfo_rooty() + btn.winfo_height() + 2
            width = max(btn.winfo_width(), 165)
        else:
            # fallback
            x = parent_widget.winfo_rootx()
            y = parent_widget.winfo_rooty()
            width = 165

        popup.geometry(f"{width}x{max(30, len(options) * 30)}+{x}+{y}")

        def on_select(value):
            var.set(value)
            if btn:
                btn.config(text=f"{value} ▼")
            popup.destroy()
            # Khi chọn, gọi handler lọc
            self._filter_dropdown_selected(value)

        container = tk.Frame(popup, bg="#FFFFFF")
        container.pack(fill="both", expand=True)

        for opt in options:
            lbl = tk.Label(
                container,
                text=opt,
                font=("Crimson Pro", 11),
                bg="#FFFFFF",
                fg="#B992B9",
                anchor="w",
                padx=10,
                pady=3,
            )

            lbl.pack(fill="x")

            lbl.bind("<Enter>", lambda e, l=lbl: l.config(bg="#EDE6F9", fg="#2E1E5B"))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(bg="#FFFFFF", fg="#B992B9"))
            lbl.bind("<Button-1>", lambda e, v=opt: on_select(v))

        # Fade-in nhẹ (non-blocking)
        try:
            popup.attributes("-alpha", 0.0)
            for i in range(1, 11):
                popup.after(i * 10, lambda a=i: popup.attributes("-alpha", a / 10))
        except Exception:
            pass

        popup.focus_force()
        popup.bind("<FocusOut>", lambda e: popup.destroy())

    def _filter_dropdown_selected(self, sel_pkg):
        """Gọi khi chọn item trong dropdown filter."""
        self.active_mode = "filter"

        import tkinter as tk

        # Ghi lại lựa chọn
        if not hasattr(self, "selected_package"):
            self.selected_package = tk.StringVar(value=sel_pkg)
        else:
            self.selected_package.set(sel_pkg)

        # Cập nhật text trên nút dropdown
        if hasattr(self, "btn_package"):
            self.btn_package.config(text=f"{sel_pkg} ▼")

        # QUAN TRỌNG: Chỉ gọi render_filtered_table, KHÔNG gọi _render_table
        if hasattr(self, "render_filtered_table"):
            self.render_filtered_table(sel_pkg)
        else:
            print("[WARNING] render_filtered_table chưa được tạo – kiểm tra _render_table() có khai báo chưa?")

    def _update_kpi(self, df_rec):
        """
        Cập nhật các KPI (tỉ lệ % cho từng action_id) lên canvas labels.
        self.kpi_labels expected to be dict: {action_id: canvas_item_id}
        """
        counts = {}
        if "action_id" in df_rec.columns:
            counts = df_rec["action_id"].value_counts(normalize=True).mul(100).to_dict()

        for aid, item in getattr(self, "kpi_labels", {}).items():
            pct = counts.get(aid, 0)
            try:
                self.canvas.itemconfigure(item, text=f"{pct:.1f}%")
            except Exception:
                # nếu self.canvas/item không tồn tại thì bỏ qua
                pass

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

