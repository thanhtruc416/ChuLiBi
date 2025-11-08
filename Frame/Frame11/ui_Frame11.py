# Frame11.py
# Giữ nguyên layout gốc, chuyển sang dạng class
# Với controller logic + on_show() + run_this_frame_only()

from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox

from Function.dropdown_profile import DropdownMenu
from QMess.Qmess_calling import Qmess

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH /Path("assets_Frame11")
from Function.Frame11_Predict import predict_customer
from Function.Frame11_Predict import validate_customer_input


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Frame11(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # --- Khởi tạo window layout ---
        self.configure(bg="#ECE7EB")

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

        # --- Hàm in entry khi click ---
        def on_entry_click(event):
            widget = event.widget
            print(f"Bạn vừa click vào ô nhập: {widget}")

        self.on_entry_click = on_entry_click
        # -----------------------------
        # PHẦN HÌNH ẢNH (IMAGE ELEMENTS)
        # -----------------------------
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(168.0, 512.0, image=self.image_image_1)

        self.image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(888.0, 42.0, image=self.image_image_2)

        self.image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.image_3 = self.canvas.create_image(163.0, 130.0, image=self.image_image_3)

        self.image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        self.image_4 = self.canvas.create_image(888.0, 903.0, image=self.image_image_4)

        self.image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        self.image_5 = self.canvas.create_image(567.0, 903.0, image=self.image_image_5)

        self.image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        self.image_6 = self.canvas.create_image(707.0, 954.0, image=self.image_image_6)

        self.image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        self.image_7 = self.canvas.create_image(1204.0, 954.0, image=self.image_image_7)

        self.image_image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        self.image_8 = self.canvas.create_image(948.0, 903.0, image=self.image_image_8)

        self.image_image_9 = PhotoImage(file=relative_to_assets("image_9.png"))
        self.image_9 = self.canvas.create_image(1331.0, 903.0, image=self.image_image_9)

        self.image_image_10 = PhotoImage(file=relative_to_assets("image_10.png"))
        self.image_10 = self.canvas.create_image(888.0, 447.0, image=self.image_image_10)
        # -----------------------------
        # PHẦN TEXT LABELS
        # -----------------------------
        self.canvas.create_text(380.0, 8.0, anchor="nw",
            text="Predict New Customer", fill="#000000",
            font=("Young Serif Regular", 40 * -1))

        self.canvas.create_text(99.0, 956.0, anchor="nw",
            text="ChuLiBi", fill="#FDE5F4",
            font=("Rubik Burned Regular", 35 * -1))

        self.canvas.create_text(385.0, 820.0, anchor="nw",
            text="Predict", fill="#374A5A",
            font=("Crimson Pro Bold", 36 * -1))

        self.canvas.create_text(387.0, 889.0, anchor="nw",
            text="Cluster", fill="#634D94",
            font=("Crimson Pro Bold", 20 * -1))
        self.canvas.create_text(720.0, 889.0, anchor="nw",
            text="Churn risk", fill="#634D94",
            font=("Crimson Pro Bold", 20 * -1))
        self.canvas.create_text(1086.0, 889.0, anchor="nw",
            text="Expected Loss", fill="#634D94",
            font=("Crimson Pro Bold", 20 * -1))
        self.canvas.create_text(387.0, 943.0, anchor="nw",
            text="Recommend", fill="#634D94",
            font=("Crimson Pro Bold", 20 * -1))
        self.canvas.create_text(933.0, 943.0, anchor="nw",
            text="Channel", fill="#634D94",
            font=("Crimson Pro Bold", 20 * -1))

        self.canvas.create_text(558.0, 881.0, anchor="nw",
                                text="0", fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text")
        self.canvas.create_text(915, 881.0, anchor="nw",
                                text="50.0%", fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text")
        self.canvas.create_text(1310.0, 881.0, anchor="nw",
                                text="1.00", fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text")
        self.canvas.create_text(548.0, 943.0, anchor="nw",
                                text="Đăng ký loyalty/điểm thưởng quay lại",
                                fill="#000000", font=("Crimson Pro Bold", 17 * -1), tags="result_text")
        self.canvas.create_text(1078.0, 943.0, anchor="nw",
                                text="In-app", fill="#000000", font=("Crimson Pro Bold", 17 * -1), tags="result_text")

        self.canvas.create_text(385.0, 108.0, anchor="nw",
            text="Input part", fill="#374A5A",
            font=("Crimson Pro Bold", 38 * -1))
        self.canvas.create_text(385.0, 155.0, anchor="nw",
            text="Please enter all required information to proceed the prediction",
            fill="#B992B9", font=("Crimson Pro Regular", 18 * -1))

        # (Các label 1.Age ... 23.Influence of rating)
        text_labels = [
            ("1.Age", 387, 202), ("2.Gender", 897, 202),
            ("3.Marriage Status", 387, 250), ("4.Occupation", 897, 250),
            ("5.Educational Qualifications", 389, 298), ("6.Family Size", 897, 298),
            ("7.Frequently Used Medium", 389, 346), ("8.Frequently Ordered Meal Category", 897, 346),
            ("9.Preference", 389, 394), ("10.Restaurant Rating", 897, 394),
            ("11.Delivery Rating", 389, 442), ("12.No. of Orders Placed", 897, 442),
            ("13.Delivery Time", 389, 490), ("14.Order Value", 897, 490),
            ("15.Ease and Convenient", 389, 538), ("16.Self Cooking", 897, 538),
            ("17.Health Concern", 389, 586), ("18.Late Delivery", 897, 586),
            ("19.Poor Hygiene", 387, 634),
            ("20.Bad Past Experience", 897, 634),
            ("21.More Offers and Discount", 385, 682), ("22.Maximum Wait Time", 897, 682),
            ("23.Influence of Rating", 387, 730)
        ]
        for text, x, y in text_labels:
            self.canvas.create_text(x, y, anchor="nw", text=text,
                                    fill="#634D94", font=("Crimson Pro Bold", 19 * -1))
        # -----------------------------
        # PHẦN ENTRY (Ô NHẬP DỮ LIỆU)
        # -----------------------------
        def make_entry(img_name, x_img, y_img, x_place, y_place, w, h):
            img = PhotoImage(file=relative_to_assets(img_name))
            self.canvas.create_image(x_img, y_img, image=img)
            entry = Entry(
                self,
                bd=0,
                bg="#FFFFFF",
                fg="#000716",
                highlightthickness=0,
                font=("Crimson Pro SemiBold", 15 * -1)  # 👈 Thêm dòng này
            )
            entry.place(x=x_place, y=y_place, width=w, height=h)
            entry.bind("<FocusIn>", self.on_entry_click)
            return entry, img

        # --- Các entry bên trái ---
        self.entry_age, self.entry_img_1 = make_entry("entry_1.png", 747.5, 215.0, 655.0, 203.0, 185.0, 22.0)
        self.entry_Marriage, self.entry_img_13 = make_entry("entry_13.png", 747.5, 264.0, 655.0, 252.0, 185.0, 22.0)
        self.entry_EducationalQualification, self.entry_img_14 = make_entry("entry_14.png", 747.5, 311.0, 655.0, 299.0, 185.0, 22.0)
        self.entry_Frenquently_user_Medium, self.entry_img_15 = make_entry("entry_15.png", 747.5, 359.0, 655.0, 347.0, 185.0, 22.0)
        self.entry_Preference, self.entry_img_16 = make_entry("entry_16.png", 747.5, 406.0, 655.0, 394.0, 185.0, 22.0)
        self.entry_Delivery_Rating, self.entry_img_17 = make_entry("entry_17.png", 747.5, 453.0, 655.0, 441.0, 185.0, 22.0)
        self.entry_Delivery_Time, self.entry_img_18 = make_entry("entry_18.png", 747.5, 503.0, 655.0, 491.0, 185.0, 22.0)
        self.entry_Ease_and_Convenient, self.entry_img_19 = make_entry("entry_19.png", 747.5, 551.0, 655.0, 539.0, 185.0, 22.0)
        self.entry_Health_Concern, self.entry_img_20 = make_entry("entry_20.png", 747.5, 599.0, 655.0, 587.0, 185.0, 22.0)
        self.entry_Poor_Hygiene, self.entry_img_21 = make_entry("entry_21.png", 747.5, 647.0, 655.0, 635.0, 185.0, 22.0)
        self.entry_More_Offers_and_Discount, self.entry_img_22 = make_entry("entry_22.png", 747.5, 695.0, 655.0, 683.0, 185.0, 22.0)
        self.entry_Influence_of_Rating, self.entry_img_23 = make_entry("entry_23.png", 747.5, 743.0, 655.0, 731.0, 185.0, 22.0)

        # --- Các entry bên phải ---
        self.entry_Gender, self.entry_img_2 = make_entry("entry_2.png", 1256.5, 215.0, 1164.0, 203.0, 185.0, 22.0)
        self.entry_Occupation, self.entry_img_3 = make_entry("entry_3.png", 1256.5, 264.0, 1164.0, 252.0, 185.0, 22.0)
        self.entry_Family_size, self.entry_img_4 = make_entry("entry_4.png", 1254.5, 311.0, 1162.0, 299.0, 185.0, 22.0)
        self.entry_Frequently_ordered_Meal_category, self.entry_img_5 = make_entry("entry_5.png", 1277.5, 359.0, 1208.0, 347.0, 139.0, 22.0)
        self.entry_Restaurant_Rating, self.entry_img_6 = make_entry("entry_6.png", 1254.5, 407.0, 1162.0, 395.0, 185.0, 22.0)
        self.entry_No_of_orders_placed, self.entry_img_7 = make_entry("entry_7.png", 1254.5, 453.0, 1162.0, 441.0, 185.0, 22.0)
        self.entry_Order_Value, self.entry_img_8 = make_entry("entry_8.png", 1254.5, 503.0, 1162.0, 491.0, 185.0, 22.0)
        self.entry_Self_Cooking, self.entry_img_9 = make_entry("entry_9.png", 1254.5, 551.0, 1162.0, 539.0, 185.0, 22.0)
        self.entry_Late_Delivery, self.entry_img_10 = make_entry("entry_10.png", 1252.5, 599.0, 1160.0, 587.0, 185.0, 22.0)
        self.entry_Bad_past_Experience, self.entry_img_11 = make_entry("entry_11.png", 1252.5, 647.0, 1160.0, 635.0, 185.0, 22.0)
        self.entry_Maximum_wait_time, self.entry_img_12 = make_entry("entry_12.png", 1252.5, 695.0, 1160.0, 683.0, 185.0, 22.0)

        # -----------------------------
        # PHẦN BUTTON
        # -----------------------------
        def make_button(img_name, x, y, w, h, cmd_text):
            img = PhotoImage(file=relative_to_assets(img_name))
            btn = Button(self, image=img, borderwidth=0, highlightthickness=0,
                         command=lambda: print(f"{cmd_text} clicked"), relief="flat")
            btn.image = img
            btn.place(x=x, y=y, width=w, height=h)
            return btn

        # Các button chính
        # -----------------------------
        # PHẦN BUTTON
        # -----------------------------
        # 🧠 Profile
        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.dropdown = DropdownMenu(self)
        self.button_Profile = Button(self, image=self.button_image_1, borderwidth=0,
                                     highlightthickness=0, command=self.dropdown.show, relief="flat")
        self.button_Profile.place(x=1361.18, y=17.03, width=44.18, height=44.69)

        # ⚙️ Nút phụ thứ 2 (icon tròn nhỏ)
        self.button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        button_2 = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_2 clicked"),
            relief="flat"
        )
        button_2.place(x=1292.0, y=17.0, width=49.0, height=49.0)

        # 📊 Dashboard
        self.button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
        button_Dashboard = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame06"),
            relief="flat"
        )
        button_Dashboard.place(x=0.0, y=222.0, width=335.22, height=88.0)

        # 👤 Customer Analysis
        self.button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        button_CustomerAnalysis = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame07"),
            relief="flat"
        )
        button_CustomerAnalysis.place(x=1.0, y=331.0, width=335.0, height=77.0)

        # 📉 Churn
        self.button_image_5 = PhotoImage(file=relative_to_assets("button_5.png"))
        button_Churn = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame08"),
            relief="flat"
        )
        button_Churn.place(x=2.0, y=409.0, width=334.0, height=86.0)

        # 📦 El (giả sử là Elastic / Explore)
        self.button_image_8 = PhotoImage(file=relative_to_assets("button_8.png"))
        button_el = Button(
            self,
            image=self.button_image_8,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame09"),
            relief="flat"
        )
        button_el.place(x=0.0, y=497.0, width=336.0, height=78.0)

        # 💡 Recommendation
        self.button_image_6 = PhotoImage(file=relative_to_assets("button_6.png"))
        button_Recommendation = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame10"),
            relief="flat"
        )
        button_Recommendation.place(x=1.0, y=596.0, width=336.0, height=82.0)

        # 🔮 Predict Customer
        self.button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
        button_PredictCustomer = Button(
            self,
            image=self.button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame11"),
            relief="flat"
        )
        button_PredictCustomer.place(x=11.0, y=690.0, width=325.0, height=76.0)

        # 🧾 Predict (nút chạy model)
        self.button_image_9 = PhotoImage(file=relative_to_assets("button_9.png"))
        button_Predict = Button(
            self,
            image=self.button_image_9,
            borderwidth=0,
            highlightthickness=0,
            command=self.on_predict_clicked,
            relief="flat"
        )
        button_Predict.place(x=1206.0, y=722.0, width=149.0, height=54.0)

        # Các nút nhỏ chọn dữ liệu (icon 10px)
        btn_small_specs = [
            ("button_10.png", 830, 259, "Marriage"), ("button_11.png", 1338, 210, "Gender"),
            ("button_12.png", 1338, 259, "Occupation"), ("button_13.png", 1338, 354, "Meal_Category"),
            ("button_14.png", 1338, 402, "Restaurant_Rating"), ("button_15.png", 1338, 546, "Self_Cooking"),
            ("button_16.png", 1338, 594, "Late_Delivery"), ("button_17.png", 1338, 642, "Bad_past_Experience"),
            ("button_18.png", 1338, 690, "Maximum_wait_time"), ("button_19.png", 830, 306, "Education"),
            ("button_20.png", 830, 355, "Medium"), ("button_21.png", 830, 401, "Preference"),
            ("button_22.png", 830, 448, "Delivery_Rating"), ("button_23.png", 830, 546, "Ease_and_Convenient"),
            ("button_24.png", 830, 594, "Health_Concern"), ("button_25.png", 830, 643, "Poor_Hygiene"),
            ("button_26.png", 830, 690, "More_Offers_and_Discount"), ("button_27.png", 830, 738, "Influence_of_Rating")
        ]
        for img, x, y, name in btn_small_specs:
            make_button(img, x, y, 10.0, 10.0, name)
            # =====================================================
            #  DROPDOWN XỔ XUỐNG ĐẸP, DÙNG HÌNH NÚT ICON 10PX
            # =====================================================
            self.dropdown_values = {
                "Gender": ["Male", "Female", "Prefer not to say"],
                "Marriage": ["Single", "Married", "Prefer not to say"],
                "Occupation": ["Student", "Employee", "Self Employeed", "House wife"],
                "Education": ["School", "Graduate", "Post Graduate", "Ph.D", "Uneducated"],
                "Medium": ["Food delivery apps", "Walk-in", "Direct call", "Web browser"],
                "Meal_Category": ["Breakfast", "Lunch", "Dinner", "Snacks", "Sweets", "Bakery items (snacks)"],
                "Preference": ["1", "2", "3", "4", "5"],
                "Restaurant_Rating": ["1", "2", "3", "4", "5"],
                "Delivery_Rating": ["1", "2", "3", "4", "5"],
                "Ease_and_Convenient": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Self_Cooking": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Health_Concern": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Late_Delivery": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Poor_Hygiene": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Bad_past_Experience": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "More_Offers_and_Discount": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                "Maximum_wait_time": ["15 minutes", "30 minutes", "45 minutes", "60 minutes", "More than 60 minutes"],
                "Influence_of_Rating": ["Yes", "No", "Maybe"],
                "Order_Value": ["1", "2", "3"]
            }

            # mapping entry cho dropdown
            entry_map = {
                "Gender": self.entry_Gender,
                "Marriage": self.entry_Marriage,
                "Occupation": self.entry_Occupation,
                "Education": self.entry_EducationalQualification,
                "Medium": self.entry_Frenquently_user_Medium,
                "Meal_Category": self.entry_Frequently_ordered_Meal_category,
                "Preference": self.entry_Preference,
                "Restaurant_Rating": self.entry_Restaurant_Rating,
                "Delivery_Rating": self.entry_Delivery_Rating,
                "Ease_and_Convenient": self.entry_Ease_and_Convenient,
                "Self_Cooking": self.entry_Self_Cooking,
                "Health_Concern": self.entry_Health_Concern,
                "Late_Delivery": self.entry_Late_Delivery,
                "Poor_Hygiene": self.entry_Poor_Hygiene,
                "Bad_past_Experience": self.entry_Bad_past_Experience,
                "Order_Value": self.entry_Order_Value,
                "More_Offers_and_Discount": self.entry_More_Offers_and_Discount,
                "Maximum_wait_time": self.entry_Maximum_wait_time,
                "Influence_of_Rating": self.entry_Influence_of_Rating
            }
            # ===== ĐẶT ENTRY CÓ DROPDOWN KHÔNG CHO NHẬP (CHỈ CHỌN TỪ MENU) =====
            readonly_keys = [
                "Gender", "Marriage", "Occupation", "Education", "Medium",
                "Meal_Category", "Preference", "Restaurant_Rating", "Delivery_Rating",
                "Ease_and_Convenient", "Self_Cooking", "Health_Concern",
                "Late_Delivery", "Poor_Hygiene", "Bad_past_Experience",
                "More_Offers_and_Discount", "Maximum_wait_time", "Influence_of_Rating","Order_Value"
            ]

            for key in readonly_keys:
                entry_widget = entry_map.get(key)
                if entry_widget:
                    # 🔒 Không cho nhập, vẫn giữ nền trắng (ăn theo ảnh PNG)
                    entry_widget.configure(
                        state="disabled",
                        disabledbackground="#FFFFFF",
                        disabledforeground="#000716"
                    )

            # tạo các icon nhỏ
            btn_small_specs = [

                ("button_10.png", 1338, 498, "Order_Value"),
                ("button_10.png", 830, 259, "Marriage"), ("button_11.png", 1338, 210, "Gender"),
                ("button_12.png", 1338, 259, "Occupation"), ("button_13.png", 1338, 354, "Meal_Category"),
                ("button_14.png", 1338, 402, "Restaurant_Rating"), ("button_15.png", 1338, 546, "Self_Cooking"),
                ("button_16.png", 1338, 594, "Late_Delivery"), ("button_17.png", 1338, 642, "Bad_past_Experience"),
                ("button_18.png", 1338, 690, "Maximum_wait_time"), ("button_19.png", 830, 306, "Education"),
                ("button_20.png", 830, 355, "Medium"), ("button_21.png", 830, 401, "Preference"),
                ("button_22.png", 830, 448, "Delivery_Rating"), ("button_23.png", 830, 546, "Ease_and_Convenient"),
                ("button_24.png", 830, 594, "Health_Concern"), ("button_25.png", 830, 643, "Poor_Hygiene"),
                ("button_26.png", 830, 690, "More_Offers_and_Discount"),
                ("button_27.png", 830, 738, "Influence_of_Rating")
            ]

            for img, x, y, key in btn_small_specs:
                icon = PhotoImage(file=relative_to_assets(img))
                btn = Button(
                    self,
                    image=icon,
                    borderwidth=0,
                    highlightthickness=0,
                    relief="flat",
                    command=lambda k=key: self.show_dropdown(entry_map[k], k)
                )
                btn.image = icon
                btn.place(x=x, y=y, width=10, height=10)

    def show_dropdown(self, entry_widget, key):
        """Hiện dropdown custom ngay dưới entry — màu & font theo ChuLiBi style."""
        values = self.dropdown_values.get(key, [])
        if not values:
            print(f"Không có giá trị dropdown cho {key}")
            return

        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.config(bg="#FFFFFF")

        x = self.winfo_rootx() + entry_widget.winfo_x() + 25
        y = self.winfo_rooty() + entry_widget.winfo_y() + entry_widget.winfo_height()

        # --- Nếu là "Frequently ordered Meal category" thì dịch riêng ---
        if key == "Meal_Category" or key == "Frequently ordered Meal category":
            x = max(0, x - 45)

        popup.geometry(f"170x{len(values) * 30}+{x}+{y}")

        def on_select(val):
            entry_widget.configure(state="normal")
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, val)
            entry_widget.configure(
                state="disabled",
                disabledbackground="#FFFFFF",
                disabledforeground="#000716"
            )
            popup.destroy()

        # === Style đẹp theo yêu cầu ===
        for val in values:
            lbl = tk.Label(
                popup,
                text=val,
                bg="#FFFFFF",
                fg="#745fa3",
                font=("Crimson Pro SemiBold", 12),
                anchor="w",
                padx=12,
                pady=3
            )
            lbl.pack(fill="x")

            lbl.bind("<Button-1>", lambda e, v=val: on_select(v))
            lbl.bind("<Enter>", lambda e, l=lbl: l.config(
                bg="#EDE6F9",
                fg="#2E1E5B"
            ))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(
                bg="#FFFFFF",
                fg="#B992B9"
            ))

        popup.focus_force()
        popup.bind("<FocusOut>", lambda e: popup.destroy())

    def on_predict_clicked(self):
            customer = {
                'Age': self.entry_age.get(),
                'Gender': self.entry_Gender.get(),
                'Marriage Status': self.entry_Marriage.get(),
                'Occupation': self.entry_Occupation.get(),
                'Educational Qualifications': self.entry_EducationalQualification.get(),
                'Family size': self.entry_Family_size.get(),
                'Frequently used Medium': self.entry_Frenquently_user_Medium.get(),
                'Frequently ordered Meal category': self.entry_Frequently_ordered_Meal_category.get(),
                'Preference': self.entry_Preference.get(),
                'Restaurant Rating': self.entry_Restaurant_Rating.get(),
                'Delivery Rating': self.entry_Delivery_Rating.get(),
                'No. of orders placed': self.entry_No_of_orders_placed.get(),
                'Delivery Time': self.entry_Delivery_Time.get(),
                'Order Value': self.entry_Order_Value.get(),
                'Ease and Convenient': self.entry_Ease_and_Convenient.get(),
                'Self Cooking': self.entry_Self_Cooking.get(),
                'Health Concern': self.entry_Health_Concern.get(),
                'Late Delivery': self.entry_Late_Delivery.get(),
                'Poor Hygiene': self.entry_Poor_Hygiene.get(),
                'Bad past experience': self.entry_Bad_past_Experience.get(),
                'More Offers and Discount': self.entry_More_Offers_and_Discount.get(),
                'Maximum wait time': self.entry_Maximum_wait_time.get(),
                'Influence of rating': self.entry_Influence_of_Rating.get()
            }

            # Kiểm tra dữ liệu đầu vào
            is_valid, errors = validate_customer_input(customer)
            if not is_valid:
                error_text = "\n".join(errors)
                Qmess.popup_24(parent=self, title="Warning",
                               subtitle=f"Invalid Input")
                return  # Dừng lại, không chạy model

            # Nếu hợp lệ, chạy mô hình
            try:
                result = predict_customer(customer)

                # Xóa kết quả cũ
                self.canvas.delete("result_text")

                # --- Hiển thị kết quả ---
                self.canvas.create_text(
                    558.0, 881.0, anchor="nw",
                    text=str(result['cluster']),
                    fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text"
                )
                self.canvas.create_text(
                    915.0, 881.0, anchor="nw",
                    text=result['churn']['churn_risk_pct'],
                    fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text"
                )
                self.canvas.create_text(
                    1310.0, 881.0, anchor="nw",
                    text=f"{result['expected_loss']['ExpectedLoss_real']:.2f}",
                    fill="#794679", font=("Kodchasan Regular", 24 * -1), tags="result_text"
                )

                # Recommend + Channel
                x_rec, y_rec = self.canvas.coords(self.image_6)
                self.canvas.create_text(
                    x_rec, y_rec,
                    text=result['recommendation']['action_name'],
                    fill="#000000", font=("Crimson Pro Bold", 17 * -1),
                    anchor="center", width=380, tags="result_text"
                )
                x_chn, y_chn = self.canvas.coords(self.image_7)
                self.canvas.create_text(
                    x_chn, y_chn,
                    text=result['recommendation']['channel'],
                    fill="#000000", font=("Crimson Pro Bold", 17 * -1),
                    anchor="center", width=180, tags="result_text"
                )

                Qmess.popup_22(parent=self, title="Success",
                               subtitle=f"Prediction completed successfully!")

            except Exception as e:
                Qmess.popup_02(parent=self, title="System Error", subtitle=f"Authentication module not found: {str(e)}")

        # -----------------------------
        # PHẦN LOGIC HIỂN THỊ FRAME
        # -----------------------------
    def on_entry_click(self, event):
            widget = event.widget
            print(f"Bạn vừa click vào ô nhập: {widget}")

    def on_show(self):
            """Gọi khi frame này được hiển thị."""
            print("Frame11 đang được hiển thị...")

        # -----------------------------
        # CHẠY FRAME11 RIÊNG LẺ
        # -----------------------------
def run_this_frame_only():
            import tkinter as tk
            root = tk.Tk()
            root.title("ChuLiBi – Predict New Customer")
            root.geometry("1440x1024")
            root.configure(bg="#ECE7EB")
            root.resizable(False, False)

            # Tạo và chạy frame
            app = Frame11(root)
            app.pack(fill="both", expand=True)

            if hasattr(app, "on_show"):
                app.on_show()

            root.mainloop()

if __name__ == "__main__":
    run_this_frame_only()




