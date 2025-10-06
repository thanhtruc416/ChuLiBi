# frame03.py
from pathlib import Path
from tkinter import Frame, Canvas, Entry, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame03")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame03(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")

        # --- Canvas ---
        canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        canvas.place(x=0, y=0)

        # --- Images ---
        self.image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        canvas.create_image(360.0, 512.0, image=self.image_1)

        self.image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        canvas.create_image(360.0, 512.0, image=self.image_2)

        self.image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        canvas.create_image(1081.0, 533.0, image=self.image_3)

        self.image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        canvas.create_image(720.0, 512.0, image=self.image_4)

        self.image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        canvas.create_image(360.0, 334.0, image=self.image_5)

        self.image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        canvas.create_image(872.0, 427.0, image=self.image_6)

        self.image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        canvas.create_image(873.0, 409.0, image=self.image_7)

        self.image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        canvas.create_image(872.0, 561.8333740234375, image=self.image_8)

        self.image_9 = PhotoImage(file=relative_to_assets("image_9.png"))
        canvas.create_image(872.0, 563.291748046875, image=self.image_9)

        self.image_10 = PhotoImage(file=relative_to_assets("image_10.png"))
        canvas.create_image(871.25, 551.0, image=self.image_10)

        self.image_11 = PhotoImage(file=relative_to_assets("image_11.png"))
        canvas.create_image(872.08349609375, 561.0833740234375, image=self.image_11)

        self.image_12 = PhotoImage(file=relative_to_assets("image_12.png"))
        canvas.create_image(868.8125, 691.0, image=self.image_12)

        self.image_13 = PhotoImage(file=relative_to_assets("image_13.png"))
        canvas.create_image(869.0, 704.5, image=self.image_13)

        self.image_14 = PhotoImage(file=relative_to_assets("image_14.png"))
        canvas.create_image(881.75, 696.90625, image=self.image_14)

        # --- Text ---
        canvas.create_text(913.0, 277.0, anchor="nw", text="Your Profile", fill="#000000",
                           font=("Young Serif", 55 * -1))
        canvas.create_text(948.0, 179.0, anchor="nw", text="Complete ", fill="#000000",
                           font=("Young Serif", 55 * -1))
        canvas.create_text(153.0, 466.0, anchor="nw", text="Almost there!", fill="#FFFFFF",
                           font=("Young Serif", 60 * -1))
        canvas.create_text(39.0, 565.0, anchor="nw",
                           text=" A few details help ChuLiBi personalize ", fill="#FFFFFF",
                           font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(132.0, 615.0, anchor="nw", text="insights for your business.",
                           fill="#FFFFFF", font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(908.0, 686.0, anchor="nw", text="Your Role", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(908, 409, anchor="nw", text="FullName", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(908.0, 545.0, anchor="nw", text="Business Name", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))

        # --- Entries ---
        self.entry_image_yourrole = PhotoImage(file=relative_to_assets("entry_full_name.png"))
        canvas.create_image(1081.5, 755.5, image=self.entry_image_yourrole)
        self.entry_your_role = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                     font=("Crimson Pro Regular", 26 * -1))
        self.entry_your_role.place(x=857.0, y=721.0, width=449.0, height=67.0)

        self.entry_image_fullname = PhotoImage(file=relative_to_assets("entry_full_name.png"))
        canvas.create_image(1081.5, 484.5, image=self.entry_image_fullname)
        self.entry_full_name = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                     font=("Crimson Pro Regular", 26 * -1))
        self.entry_full_name.place(x=857.0, y=450.0, width=449.0, height=67.0)

        self.entry_image_business_name = PhotoImage(file=relative_to_assets("entry_business_name.png"))
        canvas.create_image(1081.5, 618.5, image=self.entry_image_business_name)
        self.entry_business_name = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                         font=("Crimson Pro Regular", 26 * -1))
        self.entry_business_name.place(x=857.0, y=584.0, width=449.0, height=67.0)

        # --- Buttons ---
        self.button_image_continue = PhotoImage(file=relative_to_assets("button_continue.png"))
        self.button_continue = Button(self, image=self.button_image_continue, borderwidth=0, highlightthickness=0,
                                      command=lambda: self.controller.show_frame("Frame01"), relief="flat")
        self.button_continue.place(x=843.0, y=822.0, width=482.0, height=69.0)
