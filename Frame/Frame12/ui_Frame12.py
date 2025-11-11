from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage

from Function.dropdown_profile import DropdownMenu
from QMess.Qmess_calling import Qmess
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH /Path("assets_Frame12")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame12(tk.Frame):
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")
        self.configure(bg="#FFFFFF")

        # ========== CONTAINER ==========
        self.main_container = tk.Frame(self, bg="#FFFFFF")
        self.main_container.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = Canvas(
            self.main_container,
            bg="#FFFFFF",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # ========== IMAGES (Top layer background) ==========
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(
            892.0,
            555.0,
            image=self.image_image_1
        )

        self.image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(
            891.0,
            555.0,
            image=self.image_image_2
        )

        self.image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.image_3 = self.canvas.create_image(
            890.0,
            330.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        self.image_4 = self.canvas.create_image(
            890.0,
            796.0,
            image=self.image_image_4
        )

        self.image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        self.image_5 = self.canvas.create_image(
            890.0,
            154.0,
            image=self.image_image_5
        )

        self.canvas.create_text(
            420.0,
            125.0,
            anchor="nw",
            text="Change Information",
            fill="#000000",
            font=("Crimson Pro Bold", 45 * -1)
        )

        # ========== ENTRIES ==========
        self.entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            559.0,
            296.5,
            image=self.entry_image_1
        )
        self.entry_FullName = Entry(
            self.main_container,
            bd=0,
            bg="#D9D9D9",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro", 16, "bold")
        )
        self.entry_FullName.place(
            x=432.0,
            y=273.0,
            width=254.0,
            height=45.0
        )

        self.entry_image_2 = PhotoImage(file=relative_to_assets("entry_2.png"))
        self.entry_bg_2 = self.canvas.create_image(
            559.0,
            418.5,
            image=self.entry_image_2
        )
        self.canvas.create_rectangle(
            432.0, 395.0, 432.0 + 254.0, 395.0 + 45.0,
            fill="#D9D9D9", outline="", width=0
        )
        self.label_Username_value = tk.Label(
            self.main_container,
            text="thanhtruc1",
            bg="#D9D9D9",
            fg="#6E6E6E",
            font=("Crimson Pro", 16, "bold"),
            anchor="w",
            padx=10
        )
        self.label_Username_value.place(x=432.0, y=395.0, width=254.0, height=45.0)
        self.entry_image_3 = PhotoImage(file=relative_to_assets("entry_3.png"))
        self.entry_bg_3 = self.canvas.create_image(
            891.0,
            296.5,
            image=self.entry_image_3
        )
        self.entry_Business_Name = Entry(
            self.main_container,
            bd=0,
            bg="#D9D9D9",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro", 16, "bold")
        )
        self.entry_Business_Name.place(
            x=764.0,
            y=273.0,
            width=254.0,
            height=45.0
        )

        self.entry_image_4 = PhotoImage(file=relative_to_assets("entry_4.png"))
        self.entry_bg_4 = self.canvas.create_image(
            891.0,
            418.5,
            image=self.entry_image_4
        )
        self.entry_Password = Entry(
            self.main_container,
            bd=0,
            bg="#D9D9D9",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro", 16, "bold")
        )
        self.entry_Password.place(
            x=764.0,
            y=395.0,
            width=254.0,
            height=45.0
        )

        self.entry_image_5 = PhotoImage(file=relative_to_assets("entry_5.png"))
        self.entry_bg_5 = self.canvas.create_image(
            1222.0,
            296.5,
            image=self.entry_image_5
        )
        self.entry_Your_Role = Entry(
            self.main_container,
            bd=0,
            bg="#D9D9D9",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro", 16, "bold")

        )
        self.entry_Your_Role.place(
            x=1095.0,
            y=273.0,
            width=254.0,
            height=45.0
        )

        self.entry_image_6 = PhotoImage(file=relative_to_assets("entry_6.png"))
        self.entry_bg_6 = self.canvas.create_image(
            1222.0,
            418.5,
            image=self.entry_image_6
        )
        self.canvas.create_rectangle(
            1095.0, 395.0, 1095.0 + 254.0, 395.0 + 45.0,
            fill="#D9D9D9", outline="", width=0
        )

        # Entry giả Label (có scroll, không focus được)
        self.entry_Gmail = tk.Entry(
            self.main_container,
            bd=0,
            bg="#D9D9D9",
            fg="#6E6E6E",
            font=("Crimson Pro", 16, "bold"),
            highlightthickness=0,
            relief="flat",
            exportselection=0,
            justify="left"
        )
        self.entry_Gmail.place(x=1095.0, y=395.0, width=254.0, height=45.0)
        self.entry_Gmail.insert(0, "truccct23416@st.uel.edu.vn")

        # Khóa toàn bộ thao tác bàn phím và chuột
        def block_all(event): return "break"

        self.entry_Gmail.bind("<Key>", block_all)
        self.entry_Gmail.bind("<Button-1>", lambda e: "break")  # chặn click focus

        # Nhưng cho phép cuộn ngang (nếu text dài)
        def on_mousewheel(event):
            self.entry_Gmail.xview_scroll(int(-1 * (event.delta / 120)), "units")

        self.entry_Gmail.bind("<MouseWheel>", on_mousewheel)

        # Tự động scroll nếu text dài
        self.entry_Gmail.configure(xscrollcommand=lambda *args: None)
        self.entry_Gmail.xview_moveto(0)
        # Labels for entry fields
        self.canvas.create_text(
            467.0,
            239.0,
            anchor="nw",
            text="FullName",
            fill="#000000",
            font=("Crimson Pro Bold", 28 * -1)
        )

        self.canvas.create_text(
            467.0,
            363.0,
            anchor="nw",
            text="Username",
            fill="#000000",
            font=("Crimson Pro Bold", 28 * -1)
        )
        # Icons beside labels
        self.image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        self.image_6 = self.canvas.create_image(442.0, 255.0, image=self.image_image_6)

        self.image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        self.image_7 = self.canvas.create_image(442.0, 379.0, image=self.image_image_7)

        self.image_image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        self.image_8 = self.canvas.create_image(443.0, 237.0, image=self.image_image_8)

        self.image_image_9 = PhotoImage(file=relative_to_assets("image_9.png"))
        self.image_9 = self.canvas.create_image(443.0, 361.0, image=self.image_image_9)

        # Business info
        self.canvas.create_text(
            799.0,
            238.0,
            anchor="nw",
            text="Business Name",
            fill="#000000",
            font=("Crimson Pro Bold", 28 * -1)
        )

        self.image_image_10 = PhotoImage(file=relative_to_assets("image_10.png"))
        self.image_10 = self.canvas.create_image(774.0, 253.0, image=self.image_image_10)

        self.image_image_11 = PhotoImage(file=relative_to_assets("image_11.png"))
        self.image_11 = self.canvas.create_image(774.0, 255.0, image=self.image_image_11)

        self.image_image_12 = PhotoImage(file=relative_to_assets("image_12.png"))
        self.image_12 = self.canvas.create_image(774.0, 244.0, image=self.image_image_12)

        self.canvas.create_text(
            1137.0,
            241.0,
            anchor="nw",
            text="Your Role",
            fill="#000000",
            font=("Crimson Pro Bold", 28 * -1)
        )

        self.image_image_13 = PhotoImage(file=relative_to_assets("image_13.png"))
        self.image_13 = self.canvas.create_image(1107.8125, 243.0, image=self.image_image_13)

        self.image_image_14 = PhotoImage(file=relative_to_assets("image_14.png"))
        self.image_14 = self.canvas.create_image(1108.0, 256.5, image=self.image_image_14)

        self.image_image_15 = PhotoImage(file=relative_to_assets("image_15.png"))
        self.image_15 = self.canvas.create_image(1120.75, 248.90625, image=self.image_image_15)

        self.canvas.create_text(
            800.0,
            364.0,
            anchor="nw",
            text="Password",
            fill="#000000",
            font=("Crimson Pro SemiBold", 28 * -1)
        )

        self.image_image_16 = PhotoImage(file=relative_to_assets("image_16.png"))
        self.image_16 = self.canvas.create_image(776.0, 378.0, image=self.image_image_16)

        self.canvas.create_text(
            1124.0,
            363.0,
            anchor="nw",
            text="Gmail",
            fill="#000000",
            font=("Crimson Pro Bold", 28 * -1)
        )

        self.image_image_17 = PhotoImage(file=relative_to_assets("image_17.png"))
        self.image_17 = self.canvas.create_image(776.0, 367.0, image=self.image_image_17)

        self.image_image_18 = PhotoImage(file=relative_to_assets("image_18.png"))
        self.image_18 = self.canvas.create_image(1105.0, 372.0, image=self.image_image_18)

        self.image_image_19 = PhotoImage(file=relative_to_assets("image_19.png"))
        self.image_19 = self.canvas.create_image(1105.0, 375.0, image=self.image_image_19)

        # Our Team section
        self.image_image_20 = PhotoImage(file=relative_to_assets("image_20.png"))
        self.image_20 = self.canvas.create_image(890.0, 620.0, image=self.image_image_20)

        self.canvas.create_text(
            420.0,
            590.0,
            anchor="nw",
            text="Our Team",
            fill="#000000",
            font=("Crimson Pro Bold", 45 * -1)
        )

        # Save button
        self.button_image_1 = PhotoImage(file=relative_to_assets("button_Save.png"))
        self.button_Save = Button(  # <-- đổi chữ S thành thường để đồng nhất
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self._on_save_clicked(),
            relief="flat"
        )
        self.button_Save.place(
            x=1227.0,
            y=459.0,
            width=137.0,
            height=54.0
        )

        # --- Team members ---
        self.image_image_21 = PhotoImage(file=relative_to_assets("image_21.png"))
        self.image_21 = self.canvas.create_image(1092.0, 883.0, image=self.image_image_21)
        self.canvas.create_text(
            1040.0,
            879.0,
            anchor="nw",
            text="Nhật Bản",
            fill="#706093",
            font=("Crimson Pro Bold", 26 * -1)
        )

        self.image_image_22 = PhotoImage(file=relative_to_assets("image_22.png"))
        self.image_22 = self.canvas.create_image(1291.0, 881.0, image=self.image_image_22)
        self.canvas.create_text(
            1245.0,
            879.0,
            anchor="nw",
            text="Mỹ Linh",
            fill="#706093",
            font=("Crimson Pro Bold", 26 * -1)
        )

        self.image_image_23 = PhotoImage(file=relative_to_assets("image_23.png"))
        self.image_23 = self.canvas.create_image(893.0, 883.0, image=self.image_image_23)
        self.canvas.create_text(
            830.0,
            879.0,
            anchor="nw",
            text="Hoàng Anh",
            fill="#706093",
            font=("Crimson Pro Bold", 26 * -1)
        )

        self.image_image_24 = PhotoImage(file=relative_to_assets("image_24.png"))
        self.image_24 = self.canvas.create_image(689.0, 890.0, image=self.image_image_24)
        self.canvas.create_text(
            630.0,
            879.0,
            anchor="nw",
            text="Quỳnh Chi",
            fill="#706093",
            font=("Crimson Pro Bold", 26 * -1)
        )

        # Trúc
        self.image_image_33 = PhotoImage(file=relative_to_assets("image_33.png"))
        self.image_33 = self.canvas.create_image(492.0, 890.0, image=self.image_image_33)
        self.canvas.create_text(
            430.0,
            880.0,
            anchor="nw",
            text="Thanh Trúc",
            fill="#706093",
            font=("Crimson Pro Bold", 26 * -1)
        )

        # --- Roles ---
        self.image_image_34 = PhotoImage(file=relative_to_assets("image_34.png"))
        self.image_34 = self.canvas.create_image(491.0, 791.0, image=self.image_image_34)
        self.canvas.create_text(
            440.0,
            920.0,
            anchor="nw",
            text="Project Manager",
            fill="#000000",
            font=("Crimson Pro Bold", 16 * -1)
        )
        self.canvas.create_text(
            633.5,
            920.0,
            anchor="nw",
            text="Data & ML Lead",
            fill="#000000",
            font=("Crimson Pro Bold", 16 * -1)
        )
        self.canvas.create_text(
            840.0,
            920.0,
            anchor="nw",
            text="UI/UX Designer",
            fill="#000000",
            font=("Crimson Pro Bold", 16 * -1)
        )
        self.canvas.create_text(
            1020.0,
            920.0,
            anchor="nw",
            text="Front-End Developer",
            fill="#000000",
            font=("Crimson Pro Bold", 16 * -1)
        )
        self.canvas.create_text(
            1240.0,
            920.0,
            anchor="nw",
            text="Technical Writer",
            fill="#000000",
            font=("Crimson Pro Bold", 16 * -1)
        )
        # --- Side bar, logo, and top bar ---
        self.image_image_25 = PhotoImage(file=relative_to_assets("image_25.png"))
        self.image_25 = self.canvas.create_image(891.0, 43.0, image=self.image_image_25)

        self.image_image_26 = PhotoImage(file=relative_to_assets("image_26.png"))
        self.image_26 = self.canvas.create_image(1321.0, 39.0, image=self.image_image_26)

        # Profile button
        self.button_image_2 = PhotoImage(file=relative_to_assets("button_Profile.png"))
        self.dropdown = DropdownMenu(self, controller=self.controller)
        self.button_Profile = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.dropdown.show(),
            relief="flat"
        )
        self.button_Profile.place(
            x=1362.0,
            y=17.0,
            width=44.0,
            height=45.0
        )

        # Sidebar
        self.image_image_27 = PhotoImage(file=relative_to_assets("image_27.png"))
        self.image_27 = self.canvas.create_image(171.0, 512.0, image=self.image_image_27)

        self.canvas.create_text(
            100.0,
            892.0,
            anchor="nw",
            text="ChuLiBi",
            fill="#FDE5F4",
            font=("Rubik Burned Regular", 35 * -1)
        )

        self.image_image_28 = PhotoImage(file=relative_to_assets("image_28.png"))
        self.image_28 = self.canvas.create_image( 162.0, 101.0, image=self.image_image_28)

        # Sidebar buttons
        # --- Sidebar buttons ---

        # Dashboard
        self.button_image_5 = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
        self.button_Dashboard = Button(
            self,
            image=self.button_image_5,
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
        self.button_image_3 = PhotoImage(file=relative_to_assets("button_CustomerAnalysis.png"))
        self.button_CustomerAnalysis = Button(
            self,
            image=self.button_image_3,
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
        self.button_image_7 = PhotoImage(file=relative_to_assets("button_Churn.png"))
        self.button_Churn = Button(
            self,
            image=self.button_image_7,
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

        # Expected Loss (EL)
        self.button_image_6 = PhotoImage(file=relative_to_assets("button_EL.png"))
        self.button_EL = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame09_EL"),
            relief="flat"
        )
        self.button_EL.place(
            x=0.0,
            y=466.0,
            width=338.0,
            height=81.0
        )

        # Recommendation
        self.button_image_4 = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
        self.button_Recommendation = Button(
            self,
            image=self.button_image_4,
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
        self.button_image_8 = PhotoImage(file=relative_to_assets("button_PredictCustomer.png"))
        self.button_PredictCustomer = Button(
            self,
            image=self.button_image_8,
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
        # Background decor around team
        self.image_image_29 = PhotoImage(file=relative_to_assets("image_29.png"))
        self.image_29 = self.canvas.create_image(695.0, 788.0, image=self.image_image_29)

        self.image_image_30 = PhotoImage(file=relative_to_assets("image_30.png"))
        self.image_30 = self.canvas.create_image(1092.0, 783.0, image=self.image_image_30)

        self.image_image_31 = PhotoImage(file=relative_to_assets("image_31.png"))
        self.image_31 = self.canvas.create_image(1292.0, 780.0, image=self.image_image_31)

        self.image_image_32 = PhotoImage(file=relative_to_assets("image_32.png"))
        self.image_32 = self.canvas.create_image(898.0, 788.0, image=self.image_image_32)
    # =====================================================
    # THÊM CHỨC NĂNG XỬ LÝ NGƯỜI DÙNG TRỰC TIẾP TRONG UI
    # =====================================================

    def on_show(self, user_data=None):
        """Khi Frame12 được hiển thị → điền thông tin user."""
        if not user_data and self.controller:
            try:
                user_data = self.controller.get_current_user()
            except Exception:
                user_data = {}

        user_data = user_data or {}
        self.label_Username_value.config(text=user_data.get("username", ""))


        def set_entry(entry, value, readonly=False):
            entry.configure(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, value)
            if readonly:
                entry.configure(state="readonly", fg="#6f6f6f")

        set_entry(self.entry_FullName, user_data.get("full_name", ""))
        set_entry(self.entry_Gmail, user_data.get("email", ""))
        set_entry(self.entry_Business_Name, user_data.get("business_name", ""))
        set_entry(self.entry_Your_Role, user_data.get("role", ""))


        self.entry_Password.configure(state="normal")
        self.entry_Password.delete(0, tk.END)
        print(f"[Frame12] Hiển thị thông tin user: {user_data.get('username', '')}")

    def _on_save_clicked(self):
        """Khi nhấn nút Save → cập nhật thông tin người dùng"""
        from tkinter import messagebox
        from Function.user_repository import update_user_info
        import re

        print("[DEBUG] Save button clicked")

        full_name = self.entry_FullName.get().strip()
        business_name = self.entry_Business_Name.get().strip()
        role = self.entry_Your_Role.get().strip()
        password = self.entry_Password.get().strip()
        username = self.label_Username_value.cget("text").strip()
        email = self.entry_Gmail.get().strip()

        # --- Kiểm tra thông tin bắt buộc ---
        if not full_name or not business_name or not role:
            Qmess.popup_24(parent=self, title="Warning",
                           subtitle="Please fill all the following information below")
            return

        # --- Rào điều kiện password ---
        if password:
            if len(password) < 8:
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle="Password must be at least 8 characters long.")
                return
            if not re.search(r"[A-Z]", password):
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle="Password must include at least one uppercase letter.")
                return
            if not re.search(r"[a-z]", password):
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle="Password must include at least one lowercase letter.")
                return
            if not re.search(r"\d", password):
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle="Password must include at least one number.")
                return
            if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-\+=/\\\[\]`~]", password):
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle="Password must include at least one special character.")
                return

        # --- Cập nhật dữ liệu ---
        try:
            update_user_info(username, full_name, business_name, role, email,
                             password if password else None)
            Qmess.popup_22(parent=self, title="Success",
                           subtitle="User information has been saved successfully!")
            self.entry_Password.delete(0, tk.END)
        except Exception as e:
            Qmess.popup_24(parent=self, title="Warning",
                           subtitle=f"An error occurred while saving: {e}")
            return