from pathlib import Path
from tkinter import Tk, Canvas, Entry, Button, PhotoImage

class Frame02:
    def __init__(self, parent):
        self.parent = parent

        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame02")
        self.ASSETS_PATH = ASSETS_PATH

        def relative_to_assets(path: str) -> Path:
            return ASSETS_PATH / Path(path)
        self.relative_to_assets = relative_to_assets

        # Tạo canvas
        self.canvas = Canvas(
            parent,
            bg="#FFFFFF",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # --- Load các ảnh ---
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(360.0, 512.0, image=self.image_image_1)

        self.image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(360.0, 512.0, image=self.image_image_2)

        self.image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.image_3 = self.canvas.create_image(1081.0, 513.0, image=self.image_image_3)

        self.image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
        self.image_4 = self.canvas.create_image(720.0, 512.0, image=self.image_image_4)

        # --- Texts ---
        self.canvas.create_text(915.0, 928.0, anchor="nw",
                                text="Already have an account ?", fill="#000000",
                                font=("Crimson Pro SemiBold", 23 * -1))
        self.canvas.create_text(915.0, 813.0, anchor="nw",
                                text="I agree to Terms & Privacy Policy", fill="#000000",
                                font=("Crimson Pro SemiBold", 23 * -1))
        self.canvas.create_text(874.0, 117.0, anchor="nw",
                                text="Create Account", fill="#000000",
                                font=("Young Serif", 55 * -1))

        # --- Entry password ---
        self.entry_image_password = PhotoImage(file=relative_to_assets("entry_password.png"))
        self.entry_bg_1 = self.canvas.create_image(1083.5, 609.5, image=self.entry_image_password)
        self.entry_password = Entry(
            parent, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_password.place(x=859.0, y=575.0, width=449.0, height=67.0)

        # --- Entry password confirm ---
        self.entry_image_password_confirm = PhotoImage(file=relative_to_assets("entry_password_confirm.png"))
        self.entry_bg_2 = self.canvas.create_image(1083.5, 748.5, image=self.entry_image_password_confirm)
        self.entry_password_confirm = Entry(
            parent, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_password_confirm.place(x=859.0, y=714.0, width=449.0, height=67.0)

        # --- Entry username ---
        self.entry_image_username = PhotoImage(file=relative_to_assets("entry_username.png"))
        self.entry_bg_3 = self.canvas.create_image(1083.5, 338.5, image=self.entry_image_username)
        self.entry_username = Entry(
            parent, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_username.place(x=859.0, y=304.0, width=449.0, height=67.0)

        # --- Entry email ---
        self.entry_image_email = PhotoImage(file=relative_to_assets("entry_email.png"))
        self.entry_bg_4 = self.canvas.create_image(1083.5, 472.5, image=self.entry_image_email)
        self.entry_email = Entry(
            parent, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_email.place(x=859.0, y=438.0, width=449.0, height=67.0)

        # --- Button register ---
        self.button_register_img = PhotoImage(file=relative_to_assets("button_Register.png"))
        self.button_register = Button(parent, image=self.button_register_img,
                                      borderwidth=0, highlightthickness=0,
                                      command=lambda: print("Register button clicked"), relief="flat")
        self.button_register.image = self.button_register_img
        self.button_register.place(x=839.0, y=848.0, width=473.0, height=68.0)

        # --- Button eyes ---
        self.button_eyes_img = PhotoImage(file=relative_to_assets("button_eyes_1.png"))
        self.button_eyes_1 = Button(parent, image=self.button_eyes_img,
                                    borderwidth=0, highlightthickness=0,
                                    command=lambda: print("Eyes button clicked"), relief="flat")
        self.button_eyes_1.image = self.button_eyes_img
        self.button_eyes_1.place(x=1270.0, y=597.0, width=30.0, height=21.0)

        self.button_eyes_2_img = PhotoImage(file=relative_to_assets("button_eyes_2.png"))
        self.button_eyes_2 = Button(parent, image=self.button_eyes_2_img,
                                    borderwidth=0, highlightthickness=0,
                                    command=lambda: print("Eyes 2 button clicked"), relief="flat")
        self.button_eyes_2.image = self.button_eyes_2_img
        self.button_eyes_2.place(x=1270.0, y=738.0, width=30.0, height=21.0)

        # --- Button login ---
        self.button_login_img = PhotoImage(file=relative_to_assets("button_login.png"))
        self.button_login = Button(parent, image=self.button_login_img,
                                   borderwidth=0, highlightthickness=0,
                                   command=lambda: print("Login button clicked"), relief="flat")
        self.button_login.image = self.button_login_img
        self.button_login.place(x=1170.0, y=933.0, width=70.0, height=22.0)
