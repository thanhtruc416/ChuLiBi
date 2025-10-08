from pathlib import Path
from tkinter import Frame, Canvas, Button, PhotoImage
from Function.dropdown_profile import DropdownMenu  # import module dropdown đã tạo

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame06")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


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
            relief="ridge"
        )
        canvas.place(x=0, y=0)

        # --- Images ---
        self.image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        canvas.create_image(168.0, 512.0, image=self.image_1)

        self.image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        canvas.create_image(1.0, 245.0, image=self.image_2)

        self.image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        canvas.create_image(889.0, 202.0, image=self.image_3)

        self.image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        canvas.create_image(452.0, 220.0, image=self.image_4)

        self.image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        canvas.create_image(577.0, 220.0, image=self.image_5)

        self.image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        canvas.create_image(702.0, 220.0, image=self.image_6)

        self.image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        canvas.create_image(827.0, 220.0, image=self.image_7)

        self.image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        canvas.create_image(952.0, 220.0, image=self.image_8)

        self.image_9 = PhotoImage(file=relative_to_assets("image_9.png"))
        canvas.create_image(1077.0, 220.0, image=self.image_9)

        self.image_10 = PhotoImage(file=relative_to_assets("image_10.png"))
        canvas.create_image(1202.0, 220.0, image=self.image_10)

        self.image_11 = PhotoImage(file=relative_to_assets("image_11.png"))
        canvas.create_image(1327.0, 220.0, image=self.image_11)

        self.image_12 = PhotoImage(file=relative_to_assets("image_12.png"))
        canvas.create_image(426.0, 199.0, image=self.image_12)

        self.image_13 = PhotoImage(file=relative_to_assets("image_13.png"))
        canvas.create_image(550.0, 199.0, image=self.image_13)

        self.image_14 = PhotoImage(file=relative_to_assets("image_14.png"))
        canvas.create_image(676.0, 199.0, image=self.image_14)

        self.image_15 = PhotoImage(file=relative_to_assets("image_15.png"))
        canvas.create_image(801.0, 195.0, image=self.image_15)

        self.image_16 = PhotoImage(file=relative_to_assets("image_16.png"))
        canvas.create_image(633.0, 469.0, image=self.image_16)

        self.image_17 = PhotoImage(file=relative_to_assets("image_17.png"))
        canvas.create_image(1151.0, 469.0, image=self.image_17)

        self.image_18 = PhotoImage(file=relative_to_assets("image_18.png"))
        canvas.create_image(664.0, 802.0, image=self.image_18)

        self.image_19 = PhotoImage(file=relative_to_assets("image_19.png"))
        canvas.create_image(1184.0, 802.0, image=self.image_19)

        self.image_20 = PhotoImage(file=relative_to_assets("image_20.png"))
        canvas.create_image(888.0, 42.0, image=self.image_20)

        self.image_21 = PhotoImage(file=relative_to_assets("image_21.png"))
        canvas.create_image(162.0, 101.0, image=self.image_21)

        self.image_22 = PhotoImage(file=relative_to_assets("image_22.png"))
        canvas.create_image(924.0, 198.0, image=self.image_22)

        self.image_23 = PhotoImage(file=relative_to_assets("image_23.png"))
        canvas.create_image(1050.0, 197.0, image=self.image_23)

        self.image_24 = PhotoImage(file=relative_to_assets("image_24.png"))
        canvas.create_image(1181.0, 196.0, image=self.image_24)

        self.image_25 = PhotoImage(file=relative_to_assets("image_25.png"))
        canvas.create_image(1303.0, 196.0, image=self.image_25)

        self.image_26 = PhotoImage(file=relative_to_assets("image_26.png"))
        canvas.create_image(1296.0, 42.0, image=self.image_26)

        # --- Texts ---
        canvas.create_text(98.0, 927.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4",
                           font=("Rubik Burned", 35 * -1))
        canvas.create_text(1142.0, 972.0, anchor="nw", text="Copyright by ChuLiBi", fill="#706093",
                           font=("Young Serif", 22 * -1))
        canvas.create_text(399.0, 125.0, anchor="nw", text="KPI cards", fill="#706093",
                           font=("Young Serif", 24 * -1))
        canvas.create_text(542.0, 133.0, anchor="nw", text="Top Metrics", fill="#B992B9",
                           font=("Crimson Pro", 17 * -1))
        canvas.create_text(405.0, 322.0, anchor="nw", text="Bar chart", fill="#706093",
                           font=("Young Serif", 27 * -1))
        canvas.create_text(544.0, 335.0, anchor="nw", text="Customer group with the highest spending",
                           fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(937.0, 324.0, anchor="nw", text="Histogram chart", fill="#706093",
                           font=("Young Serif", 27 * -1))
        canvas.create_text(1180.0, 324.0, anchor="nw", text="Delivery Performance \nImpact of Late Delivery",
                           fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(407.0, 659.0, anchor="nw", text="Line chart", fill="#706093",
                           font=("Young Serif", 27 * -1))
        canvas.create_text(560.0, 674.0, anchor="nw", text="Order Trends by Age Group",
                           fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(999.0, 659.0, anchor="nw", text="Pie chart", fill="#706093",
                           font=("Young Serif", 27 * -1))
        canvas.create_text(1135.0, 673.0, anchor="nw", text="Proportion of Meal Types",
                           fill="#B992B9", font=("Crimson Pro", 18 * -1))
        canvas.create_text(373.0, 16.0, anchor="nw", text="Dashboard", fill="#000000",
                           font=("Young Serif", 40 * -1))

        # --- Buttons ---
        self.button_Dashboard_image = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
        self.button_Dashboard = Button(self, image=self.button_Dashboard_image, borderwidth=0,
                                       highlightthickness=0, command=lambda: print("button_Dashboard clicked"),
                                       relief="flat")
        self.button_Dashboard.place(x=20.0, y=201.0, width=317.0, height=87.0)

        self.button_Profile_image = PhotoImage(file=relative_to_assets("button_Profile.png"))
        # dùng dropdown từ module
        self.dropdown = DropdownMenu(self)
        self.button_Profile = Button(self, image=self.button_Profile_image, borderwidth=0,
                                     highlightthickness=0, command=self.dropdown.show,
                                     relief="flat")
        self.button_Profile.place(x=1332.0, y=16.0, width=57.0, height=51.0)

        self.button_Customer_analysis_image = PhotoImage(file=relative_to_assets("button_Customer_analysis.png"))
        self.button_Customer_analysis = Button(self, image=self.button_Customer_analysis_image, borderwidth=0,
                                               highlightthickness=0,
                                               command=lambda: print("button_Customer_analysis clicked"),
                                               relief="flat")
        self.button_Customer_analysis.place(x=0.0, y=302.0, width=337.0, height=77.0)

        self.button_Churn_image = PhotoImage(file=relative_to_assets("button_Churn.png"))
        self.button_Churn = Button(self, image=self.button_Churn_image, borderwidth=0,
                                   highlightthickness=0, command=lambda: print("button_Churn clicked"),
                                   relief="flat")
        self.button_Churn.place(x=0.0, y=381.0, width=336.0, height=86.0)

        self.button_Recommendation_image = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
        self.button_Recommendation = Button(self, image=self.button_Recommendation_image, borderwidth=0,
                                            highlightthickness=0,
                                            command=lambda: print("button_Recommendation clicked"),
                                            relief="flat")
        self.button_Recommendation.place(x=0.0, y=468.0, width=336.0, height=82.0)

        self.button_Delivery_image = PhotoImage(file=relative_to_assets("button_Delivery.png"))
        self.button_Delivery = Button(self, image=self.button_Delivery_image, borderwidth=0,
                                      highlightthickness=0, command=lambda: print("button_Delivery clicked"),
                                      relief="flat")
        self.button_Delivery.place(x=0.0, y=552.0, width=337.0, height=90.0)

        self.button_Report_image = PhotoImage(file=relative_to_assets("button_Report.png"))
        self.button_Report = Button(self, image=self.button_Report_image, borderwidth=0,
                                    highlightthickness=0, command=lambda: print("button_Report clicked"),
                                    relief="flat")
        self.button_Report.place(x=0.0, y=646.0, width=338.0, height=88.0)
