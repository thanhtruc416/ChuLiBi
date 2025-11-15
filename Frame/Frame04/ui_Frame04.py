from pathlib import Path
from tkinter import Frame, Canvas, Entry, Button, PhotoImage
from Function.Frame04_ForgetPassword import send_otp_if_email_exists
from QMess.Qmess_calling import Qmess
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame04")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame04(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")
        self.lower()

        # --- Canvas ---
        canvas = Canvas(
            self, bg="#FFFFFF", height=1024, width=1440, bd=0,
            highlightthickness=0, relief="ridge"
        )
        canvas.place(x=0, y=0)

        # --- Images ---
        self.image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        canvas.create_image(360.0, 512.0, image=self.image_1)

        self.image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        canvas.create_image(360.0, 512.0, image=self.image_2)

        self.image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        canvas.create_image(1079.0, 512.0, image=self.image_3)

        self.image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        canvas.create_image(720.0, 512.0, image=self.image_4)

        self.image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
        canvas.create_image(357.0, 351.0, image=self.image_5)

        self.image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
        canvas.create_image(882.0, 441.0, image=self.image_6)

        self.image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
        canvas.create_image(882.0, 444.0, image=self.image_7)

        self.button_image_sendOTP = PhotoImage(file=relative_to_assets("button_sendOTP.png"))
        self.button_image_login = PhotoImage(file=relative_to_assets("button_login.png"))

        # --- Text ---
        canvas.create_text(918.0, 681.0, anchor="nw", text="Already have an account ?", fill="#000000",
                           font=("Crimson Pro SemiBold", 23 * -1))
        canvas.create_text(837.0, 293.0, anchor="nw", text="Forget Password", fill="#000000",
                           font=("Young Serif", 55 * -1))
        canvas.create_text(277.0, 885.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4",
                           font=("Rubik Burned", 35 * -1))
        canvas.create_text(60.0, 637.0, anchor="nw", text="we’ll send you a verification code",
                           fill="#FFFFFF", font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(202.0, 586.0, anchor="nw", text="Enter your email ", fill="#FFFFFF",
                           font=("Crimson Pro SemiBold", 40 * -1))
        canvas.create_text(910.0, 428.0, anchor="nw", text="Email", fill="#000000",
                           font=("Crimson Pro SemiBold", 28 * -1))
        canvas.create_text(76.0, 476.0, anchor="nw", text="Oops, it happens!", fill="#FFFFFF",
                           font=("Young Serif", 65 * -1))

        # --- Entry ---
        self.entry_image_email = PhotoImage(file=relative_to_assets("entry_email.png"))
        canvas.create_image(1081.5, 492.0, image=self.entry_image_email)

        self.entry_email = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_email.place(x=857.0, y=462.0, width=449.0, height=58.0)

        # --- Buttons ---
        self.button_sendOTP = Button(
            self,
            image=self.button_image_sendOTP,
            borderwidth=0,
            highlightthickness=0,
            command=self.on_click_send_otp,
            relief="flat"
        )
        self.button_sendOTP.place(x=842.0, y=574.0, width=479.0, height=70.0)

        self.button_login = Button(
            self,
            image=self.button_image_login,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame01"),
            relief="flat"
        )
        self.button_login.place(x=1176.0, y=681.0, width=63.0, height=24.0)

    # ====== HÀM GỌI OTP ======
    def on_click_send_otp(self):
        email = self.entry_email.get().strip()
        if not email:
            Qmess.popup_15(parent=self,
                        title="Warning",
                        subtitle="Please enter your email first")
            return

        # Chặn double-click trong lúc gửi
        self.button_sendOTP.config(state="disabled")
        try:
            ok, msg = send_otp_if_email_exists(email)
            if ok:
                Qmess.popup_16(parent=self,
                        title="Success",
                        subtitle="The OTP is valid for 10 minutes.\nPlease continue to enter it quickly.")
                if self.controller:
                    self.controller.show_frame("Frame05", email=email)
            else:
                Qmess.popup_17(parent=self,
                        title="Email Not Found",
                        subtitle="This email does not exist in our system. Please check and try again.")
        except Exception as e:
            Qmess.popup_17(parent=self,
                        title="Email Not Found",
                        subtitle="This email does not exist in our system. Please check and try again.")
        finally:
            self.button_sendOTP.config(state="normal")

        # Bind Enter để gọi lại handler này (giống Frame01)
        try:
            self.entry_email.bind("<Return>", lambda _e: self.on_click_send_otp())
            self.bind("<Return>", lambda _e: self.on_click_send_otp())
        except Exception:
            pass
