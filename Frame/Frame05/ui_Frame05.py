# frame05.py
from pathlib import Path
from tkinter import Frame, Canvas, Entry, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame05")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame05(Frame):
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
        canvas.create_image(1081.0, 513.0, image=self.image_3)

        self.image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        canvas.create_image(720.0, 512.0, image=self.image_4)

        self.image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        canvas.create_image(877.0, 435.0, image=self.image_5)

        self.image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        canvas.create_image(876.0, 573.0, image=self.image_6)

        self.image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        canvas.create_image(878.0, 422.0, image=self.image_7)

        self.image_8 = PhotoImage(file=relative_to_assets("image_8.png"))
        canvas.create_image(877.0, 560.0, image=self.image_8)

        self.image_9 = PhotoImage(file=relative_to_assets("image_9.png"))
        canvas.create_image(875.0, 287.0, image=self.image_9)

        self.image_10 = PhotoImage(file=relative_to_assets("image_10.png"))
        canvas.create_image(875.0, 270.0, image=self.image_10)

        self.image_11 = PhotoImage(file=relative_to_assets("image_11.png"))
        canvas.create_image(875.875, 692.875, image=self.image_11)

        self.image_12 = PhotoImage(file=relative_to_assets("image_12.png"))
        canvas.create_image(876.458, 694.25, image=self.image_12)

        self.image_13 = PhotoImage(file=relative_to_assets("image_13.png"))
        canvas.create_image(869.75, 693.25, image=self.image_13)

        self.image_14 = PhotoImage(file=relative_to_assets("image_14.png"))
        canvas.create_image(876.208, 693.25, image=self.image_14)

        self.image_15 = PhotoImage(file=relative_to_assets("image_15.png"))
        canvas.create_image(882.667, 693.25, image=self.image_15)

        self.image_16 = PhotoImage(file=relative_to_assets("image_16.png"))
        canvas.create_image(357.0, 350.0, image=self.image_16)

        self.button_image_resetpassword = PhotoImage(file=relative_to_assets("button_resetpassword.png"))
        self.button_image_eyes_password = PhotoImage(file=relative_to_assets("button_eyes.png"))
        self.button_image_eyes = PhotoImage(file=relative_to_assets("button_eyes.png"))

        # --- Text ---
        canvas.create_text(839.0, 115.0, anchor="nw", text="Reset Password", fill="#000000",
                           font=("Young Serif", 55 * -1))
        canvas.create_text(906.0, 418.0, anchor="nw", text="New Password", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(900.0, 559.0, anchor="nw", text="Confirm password", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(900.0, 682.0, anchor="nw", text="OTP-Verify", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(280.0, 900.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4",
                           font=("Rubik Burned Regular", 35 * -1))
        canvas.create_text(21.0, 477.0, anchor="nw", text="Secure your account", fill="#FFFFFF",
                           font=("Young Serif", 65 * -1))
        canvas.create_text(162.0, 597.0, anchor="nw", text="Reset your password ", fill="#FFFFFF",
                           font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(39.0, 647.0, anchor="nw", text="continue your AI journey with ChuLiBi",
                           fill="#FFFFFF", font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(903.0, 271.0, anchor="nw", text="Username", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))

        # --- Entry ---
        self.entry_image_username = PhotoImage(file=relative_to_assets("entry_username.png"))
        canvas.create_image(1083.5, 341.0, image=self.entry_image_username)
        self.entry_username = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                    font=("Crimson Pro Regular", 26 * -1))
        self.entry_username.place(x=859.0, y=306.5, width=449.0, height=67.0)

        self.entry_image_newpassword = PhotoImage(file=relative_to_assets("entry_newpassword.png"))
        canvas.create_image(1083.5, 488.5, image=self.entry_image_newpassword)
        self.entry_newpassword = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                       font=("Crimson Pro Regular", 26 * -1))
        self.entry_newpassword.place(x=859.0, y=454.0, width=449.0, height=67.0)

        self.entry_image_confirm_password = PhotoImage(file=relative_to_assets("entry_confirm_password.png"))
        canvas.create_image(1083.5, 627.5, image=self.entry_image_confirm_password)
        self.entry_confirm_newpassword = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                                               font=("Crimson Pro Regular", 26 * -1))
        self.entry_confirm_newpassword.place(x=859.0, y=593.0, width=449.0, height=67.0)

        self.entry_image_OTP = PhotoImage(file=relative_to_assets("entry_OTP.png"))
        canvas.create_image(1083.5, 750.5, image=self.entry_image_OTP)
        self.entry_OTP = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
                               font=("Crimson Pro Regular", 26 * -1))
        self.entry_OTP.place(x=859.0, y=716.0, width=449.0, height=67.0)

        # --- Buttons ---
        self.button_eyes_password = Button(self, image=self.button_image_eyes_password, borderwidth=0,
                                           highlightthickness=0, command=lambda: print("button_eyes _password clicked"),
                                           relief="flat")
        self.button_eyes_password.place(x=1270.0, y=476.0, width=30.0, height=21.0)

        self.button_eyes = Button(self, image=self.button_image_eyes, borderwidth=0, highlightthickness=0,
                                  command=lambda: print("button_eyes 2 clicked"), relief="flat")
        self.button_eyes.place(x=1270.0, y=617.0, width=30.0, height=21.0)

        self.button_resetpassword = Button(self, image=self.button_image_resetpassword, borderwidth=0,
                                           highlightthickness=0, command=lambda: self.controller.show_frame("Frame01"),
                                           relief="flat")
        self.button_resetpassword.place(x=844.0, y=817.0, width=484.0, height=67.0)
