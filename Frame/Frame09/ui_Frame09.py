# -*- coding: utf-8 -*-
# ui_Frame09.py (chỉ sửa font thành Crimson Pro Bold)

from pathlib import Path
import tkinter as tk
from tkinter import Frame, Canvas, Entry, Button, PhotoImage

from Function.dropdown_profile import DropdownMenu

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets_Frame09"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path


class Frame09(Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#ECE7EB")
        self.frame_dualmap = Frame(self, bg="#ECE7EB")
        self.frame_bar = Frame(self, bg="#ECE7EB")
        self.frame_table = Frame(self, bg="#ECE7EB")

        # NOTE: controller sẽ override these callbacks in Frame09_EL
        # Keep placeholders here (no-op) to avoid accidental call errors.
        self.on_search_customer = lambda event=None: print("[UI] on_search_customer placeholder")
        self.on_cluster_dropdown = lambda cluster=None: print("[UI] on_cluster_dropdown placeholder", cluster)

        # --- Canvas ---
        self.canvas = Canvas(
            self, bg="#ECE7EB", height=1024, width=1440,
            bd=0, highlightthickness=0, relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # --- helpers ---
        def _safe_img(name):
            try:
                return PhotoImage(file=relative_to_assets(name))
            except Exception:
                return None

        def _add_img(img, x, y):
            if img is not None:
                self.canvas.create_image(x, y, image=img)

        # --- Images nền / decor ---
        self.image_image_1 = _safe_img("image_1.png");  _add_img(self.image_image_1, 168.0, 512.0)
        self.image_image_2 = _safe_img("image_2.png");  _add_img(self.image_image_2, 888.0, 42.0)
        self.image_image_3 = _safe_img("image_3.png");  _add_img(self.image_image_3, 580.0, 42.0)
        # đổi chữ + ẩn Table
        self.canvas.create_rectangle(410, 10, 820, 80, fill="#D9CDD8", outline="")

        self.canvas.create_text(
            370.0, 10.0,  # x, y (bạn có thể tinh chỉnh sau)
            anchor="nw",
            text="Expected Loss",
            fill="#000000",  # màu chữ
            font=("Young Serif", -40)
        )

        self.image_image_4 = _safe_img("image_4.png");  _add_img(self.image_image_4, 876.0, 813.0)
        self.image_image_5 = _safe_img("image_5.png");  _add_img(self.image_image_5, 430.0, 671.0)
        #đổi chữ + ẩn Table
        self.canvas.create_rectangle(390, 655, 520, 685, fill="#FFFFFF", outline="")

        self.canvas.create_text(
            385.0, 650.0,  # x, y (bạn có thể tinh chỉnh sau)
            anchor="nw",
            text="Top 50 Customers",
            fill="#706093",  # màu chữ
            font=("Crimson Pro Bold", -30)
        )

        self.image_image_6 = _safe_img("image_6.png");  _add_img(self.image_image_6, 163.0, 130.0)
        self.image_image_7 = _safe_img("image_7.png");  _add_img(self.image_image_7, 482.0, 190.0)
        self.image_image_8 = _safe_img("image_8.png");  _add_img(self.image_image_8, 779.0, 190.0)
        self.image_image_9 = _safe_img("image_9.png");  _add_img(self.image_image_9, 629.0, 458.0)
        self.image_image_10 = _safe_img("image_10.png"); _add_img(self.image_image_10, 1158.0, 360.0)

        self.image_image_11 = _safe_img("image_11.png"); _add_img(self.image_image_11, 549.0, 338.0)
        # Ẩn chữ cũ "Mean Risk per Cluster"
        self.canvas.create_rectangle(370, 325, 730, 410, fill="#FFFFFF", outline="")
        # Viết chữ mới "Top 50 Customers"
        self.canvas.create_text(
            532.0, 340.0,
            anchor="center",
            text="Mean Risk per Cluster",
            fill="#706093",
            font=("Crimson Pro Bold", -31)
        )

        self.image_image_12 = _safe_img("image_12.png")
        _add_img(self.image_image_12, 1084.0, 140.0)
        # Ẩn chữ cũ "Dual Risk Map" (vẽ khối trắng đè lên vùng bên phải)
        self.canvas.create_rectangle(950, 125, 1320, 180, fill="#FFFFFF", outline="")

        # Viết chữ mới (ví dụ "Dual Risk Map")
        self.canvas.create_text(
            965.0, 115.0,
            anchor="nw",
            text="Dual Risk Map",
            fill="#706093",
            font=("Crimson Pro Bold", -33)
        )

        # ================== PROFILE / NOTI ==================
        self.button_image_profile = _safe_img("button_Profile.png")
        self.button_Profile = Button(
            self,
            image=self.button_image_profile,
            borderwidth=0, highlightthickness=0,
            relief="flat",
            cursor="hand2"
        )
        self.button_Profile.place(x=1361.1848, y=17.0305, width=44.1827, height=44.6934)

        # Dropdown menu
        self.dropdown = DropdownMenu(self, controller, x_offset=1270, y_offset=72, width=149)
        self.button_Profile.configure(command=lambda: self.dropdown.show())

        self.button_image_noti = _safe_img("button_Noti.png")
        self.button_Noti = Button(
            self,
            image=self.button_image_noti,
            borderwidth=0, highlightthickness=0,
            command=lambda: print("Notification clicked"),
            relief="flat"
        )
        self.button_Noti.place(x=1292.0, y=17.0, width=49.0, height=49.0)

        # --- Texts ---
        self.canvas.create_text(683.0, 695.0, anchor="nw",
                           text="Cluster", fill="#B992B9",
                           font=("Crimson Pro Bold", -20))
        self.canvas.create_text(390.0, 696.0, anchor="nw",
                           text="ID Customer", fill="#B992B9",
                           font=("Crimson Pro Bold", -18))
        self.canvas.create_text(99.0, 956.0, anchor="nw",
                           text="ChuLiBi", fill="#FDE5F4",
                           font=("Rubik Burned Regular", -35))
        self.text_highrisk_value = self.canvas.create_text(
            455.0, 185.0, anchor="nw",
            text="100", fill="#FFFFFF",
            font=("Kodchasan Regular", -40)
        )
        self.canvas.create_text(
            420.0, 130.0, anchor="nw",
            text="High Risk\nof Customer", fill="#FFEDFA",
            font=("Crimson Pro Bold", -25), width=200, justify="center"
        )
        self.canvas.create_text(
            703.0, 130.0, anchor="nw",
            text="Total\nExpected Loss",
            fill="#374A5A",
            font=("Crimson Pro Bold", -25),
            width=190,
            justify="center"
        )

        self.text_totalel_value = self.canvas.create_text(
            710.0, 185.0, anchor="nw",
            text="100", fill="#794679",
            font=("Kodchasan Regular", -40)
        )

        # ================== SIDEBAR ==================
        self.button_image_1 = _safe_img("button_CustomerAnalysis.png")
        self.button_CustomerAnalysis = Button(
            self,
            image=self.button_image_1,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame07"),
            relief="flat"
        )
        self.button_CustomerAnalysis.place(x=1.0, y=310.0, width=335.0, height=77.0)

        self.button_image_4 = _safe_img("button_Dashboard.png")
        self.button_Dashboard = Button(
            self,
            image=self.button_image_4,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame06"),
            relief="flat"
        )
        self.button_Dashboard.place(x=0.0, y=222.0, width=335.2258, height=88.0)

        self.button_image_5 = _safe_img("button_EL.png")
        self.button_EL = Button(
            self,
            image=self.button_image_5,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame09_EL"),
            relief="flat"
        )
        self.button_EL.place(x=0.0, y=473.0, width=336.0, height=81.0)

        self.button_image_6 = _safe_img("button_Churn.png")
        self.button_Churn = Button(
            self,
            image=self.button_image_6,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame08"),
            relief="flat"
        )
        self.button_Churn.place(x=0.0, y=387.0, width=334.0, height=86.0)

        self.button_image_8 = _safe_img("button_Recommendation.png")
        self.button_Recommendation = Button(
            self,
            image=self.button_image_8,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame10"),
            relief="flat"
        )
        self.button_Recommendation.place(x=0.0, y=554.0, width=336.0, height=82.0)

        self.button_image_7 = _safe_img("button_PredictCustomer.png")
        self.button_PredictCustomer = Button(
            self,
            image=self.button_image_7,
            borderwidth=0, highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame11"),
            relief="flat"
        )
        self.button_PredictCustomer.place(x=0.0, y=636.0, width=336.0, height=75.0)
        # ====================================================================

        # --- Entries ---
        self.entry_image_1 = _safe_img("entry_1.png")
        if self.entry_image_1:
            _add_img(self.entry_image_1, 842.0, 705.5)

        # --- Entry giả làm ô Cluster ---
        self.entry_cluster = tk.Label(
            self,
            text="",
            anchor="center",
            bg="#FFFFFF",
            fg="#000716",
            font=("Crimson Pro Bold", 13),
            bd=0,
            relief="flat"
        )
        self.entry_cluster.place(x=751.0, y=692.0, width=182.0, height=25.0)

        # Không cho người dùng gõ hay xóa trong ô Cluster
        self.entry_cluster.bind("<Key>", lambda e: "break")
        self.entry_cluster.bind("<BackSpace>", lambda e: "break")
        self.entry_cluster.bind("<Delete>", lambda e: "break")

        self.entry_image_2 = _safe_img("entry_2.png")
        if self.entry_image_2:
            _add_img(self.entry_image_2, 579.0, 705.0)

        self.entry_id = Entry(
            self,
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro Bold", 13),
            justify="center"
        )
        self.entry_id.place(x=500.0, y=694.0, width=158.0, height=20.0)

        # Khi người dùng nhấn Enter -> gọi callback (controller sẽ override)
        self.entry_id.bind("<Return>", self.on_search_customer)

        # ===== RÀNG BUỘC VÀ STYLE TƯƠNG TÁC ENTRY vs DROPDOWN =====
        def on_id_change(event=None):
            """Nếu người dùng gõ ID → vô hiệu hóa dropdown cluster."""
            text = self.entry_id.get().strip()
            if text:
                # Làm mờ dropdown
                self.entry_cluster.configure(bg="#F8F5F8", fg="#888888")
                self.button_dropdown.configure(state="disabled", cursor="arrow")
            else:
                self.entry_cluster.configure(bg="#FFFFFF", fg="#000716")
                self.button_dropdown.configure(state="normal", cursor="hand2")

        def on_cluster_select(value):
            """Khi chọn cluster → khóa ô CustomerID."""
            self.entry_cluster.configure(text=f"Cluster {value}", bg="#FFFFFF", fg="#000716")
            self.entry_id.delete(0, tk.END)
            self.entry_id.configure(state="disabled", disabledbackground="#F8F5F8", disabledforeground="#888888")

        def clear_cluster_if_empty(event=None):
            """Khi người dùng click vào ô Cluster → xoá và bật lại ID."""
            if self.entry_cluster.cget("text"):  # chỉ xoá nếu đang có cluster hiển thị
                self.entry_cluster.configure(text="")
                self.entry_id.configure(state="normal", bg="#FFFFFF", fg="#000716")
                self.entry_id.focus_set()

        # Gán sự kiện click để xóa cluster
        self.entry_cluster.bind("<Button-1>", clear_cluster_if_empty)

        def _on_cluster_dropdown():
            clusters = ["All Cluster", 1, 2, 3]

            # --- Tạo menu tùy chỉnh ---
            menu = tk.Menu(self, tearoff=0, font=("Crimson Pro Bold", 12), bg="#FFFFFF",
                           fg="#B992B9", activebackground="#EDE6F9", activeforeground="#2E1E5B",
                           borderwidth=0, relief="flat")

            # --- Custom style hover & border ---
            menu.configure(
                borderwidth=1,
                relief="solid",
                activeborderwidth=0
            )

            # --- Add menu items ---
            for c in clusters:
                label = "All Cluster" if c == "All Cluster" else f"Cluster {c}"
                val = c

                def _cmd(v=val):
                    on_cluster_select(v)
                    self.on_cluster_dropdown(v)

                # Dùng separator mảnh giữa nhóm
                menu.add_command(label=label, command=_cmd)

            # --- Hiển thị menu dưới nút dropdown ---
            x = self.button_dropdown.winfo_rootx() - 100  # lệch nhẹ sang trái nếu cần
            y = self.button_dropdown.winfo_rooty() + self.button_dropdown.winfo_height() + 6
            menu.tk_popup(x, y)
            menu.grab_release()

            # --- Tô nhẹ vùng popup (tránh bị “đơ trắng”) ---
            self.after(50, lambda: menu.config(bg="#FFFFFF"))

        # --- Create dropdown button ONCE and wire to local menu function ---
        self.button_image_9 = _safe_img("button_dropdown.png")
        self.button_dropdown = Button(
            self,
            image=self.button_image_9,
            borderwidth=0, highlightthickness=0,
            command=_on_cluster_dropdown,   # calls local menu creator
            relief="flat",
            cursor="hand2"
        )
        self.button_dropdown.place(x=906.0, y=700.0, width=14.0, height=12.0)

        # --- Frames hiển thị biểu đồ / bảng ---
        self.frame_dualmap = Frame(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="flat")
        self.frame_dualmap.place(x=950, y=153, width=412, height=462)
        self.frame_dualmap.pack_propagate(False)

        self.frame_bar = Frame(self, bg="#FFFFFF", bd=0, highlightthickness=0, relief="flat")
        self.frame_bar.place(x=370, y=375, width=510, height=220)

        self.frame_table = Frame(self, bg="#FFFFFF", highlightbackground="#D6C3D6", highlightthickness=0)
        self.frame_table.place(x=375, y=740, width=1000, height=230)

        # giữ reference ảnh (tránh GC)
        self._img_refs = [
            self.image_image_1, self.image_image_2, self.image_image_3, self.image_image_4,
            self.image_image_5, self.image_image_6, self.image_image_7, self.image_image_8,
            self.image_image_9, self.image_image_10, self.image_image_11, self.image_image_12,
            self.button_image_profile, self.button_image_noti,
            self.button_image_1, self.button_image_4, self.button_image_5,
            self.button_image_6, self.button_image_8, self.button_image_7,
            self.entry_image_1, self.entry_image_2, self.button_image_9
        ]

    # ================== CALLBACK PLACEHOLDERS ==================
    # kept as methods for clarity; controller will override these.
    def on_search_customer(self, event=None):
        print("[UI] on_search_customer triggered — override in Frame09_EL.")

    def on_cluster_dropdown(self, cluster=None):
        print("[UI] on_cluster_dropdown triggered — override in Frame09_EL.", cluster)


# --- Preview độc lập ---
if __name__ == "__main__":
    import tkinter as tk
    from Function.Frame09_EL import Frame09_EL

    root = tk.Tk()
    root.title("Frame09 Preview")
    root.geometry("1440x1024")

    app = Frame09_EL(root)
    app.pack(fill="both", expand=True)

    root.resizable(False, False)
    root.mainloop()
