# ui_frame01.py
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame01")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / path

class Frame01(tk.Frame):
    def __init__(self, parent, controller=None):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.password_hidden = True  # trạng thái ẩn/hiện mật khẩu

        # --- Canvas chính ---
        self.canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # --- Load hình ảnh và hiển thị ---
        self.images = {}  # lưu reference tránh bị GC
        img_files = [
            ("image_1.png", 360, 512),
            ("image_2.png", 1073, 631),
            ("image_3.png", 720, 512),
            ("image_4.png", 359, 512),
            ("image_5.png", 1061, 144),
            ("image_6.png", 863, 471),
            ("image_7.png", 863, 482),
            ("image_8.png", 864, 595),
            ("image_9.png", 864, 586),
            ("image_10.png", 867, 595),
            ("image_11.png", 864.01, 595),
            ("image_12.png", 861.02, 595),
            ("image_13.png", 1020, 810),
            ("image_14.png", 1100, 810)
        ]
        for filename, x, y in img_files:
            img = PhotoImage(file=relative_to_assets(filename))
            self.images[filename] = img
            self.canvas.create_image(x, y, image=img)

        # --- Text ---
        self.canvas.create_text(
            921.0,
            851.0,
            anchor="nw",
            text="Don’t have an account?",
            fill="#000000",
            font=("Crimson Pro SemiBold", 23 * -1)
        )

        self.canvas.create_text(
            957.0,
            792.0,
            anchor="nw",
            text="Or",
            fill="#000000",
            font=("Crimson Pro SemiBold", 23 * -1)
        )

        self.canvas.create_text(
            883.0,
            456.0,
            anchor="nw",
            text="Username",
            fill="#000000",
            font=("Crimson Pro SemiBold", 28 * -1)
        )

        self.canvas.create_text(
            969.0,
            345.0,
            anchor="nw",
            text="Login",
            fill="#000000",
            font=("Young Serif", 64 * -1)
        )

        self.canvas.create_text(
            883.0,
            570.0,
            anchor="nw",
            text="Password",
            fill="#000000",
            font=("Crimson Pro SemiBold", 28 * -1)
        )

        self.canvas.create_text(
            80.0,
            347.0,
            anchor="nw",
            text="Welcome back to",
            fill="#000000",
            font=("Young Serif", 65 * -1)
        )

        self.canvas.create_text(
            206.0,
            454.0,
            anchor="nw",
            text="ChuLiBi",
            fill="#000000",
            font=("Young Serif", 70 * -1)
        )

        self.canvas.create_text(
            33.0,
            595.0,
            anchor="nw",
            text="Your AI data buddy for smarter insights",
            fill="#000000",
            font=("Crimson Pro SemiBold", 40 * -1)
        )

        self.canvas.create_text(
            49.0,
            31.0,
            anchor="nw",
            text="ChuLiBi",
            fill="#FDE5F4",
            font=("Rubik Burned Regular", 35 * -1)
        )

        # --- Entry Username ---
        self.dn_username_image = PhotoImage(file=relative_to_assets("entry_username.png"))
        self.dn_username_bg_image = self.canvas.create_image(1065.0, 524.0, image=self.dn_username_image)
        self.dn_username = Entry(
            self,
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1)
        )
        self.dn_username.place(x=855.0, y=494.0, width=420.0, height=58.0)

        # --- Entry Password ---
        self.dn_password_image = PhotoImage(file=relative_to_assets("entry_password.png"))
        self.dn_password_bg_image = self.canvas.create_image(1065.0, 644.0, image=self.dn_password_image)
        self.dn_password = Entry(
            self,
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1),
            show="*"
        )
        self.dn_password.place(x=855.0, y=614.0, width=420.0, height=58.0)

        # --- Button hiện/ẩn mật khẩu ---
        self.dn_eye_image = PhotoImage(file=relative_to_assets("button_eyes.png"))
        self.dn_eye = Button(
            self,
            image=self.dn_eye_image,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            command=self.toggle_password_visibility
        )
        self.dn_eye.place(x=1243.0, y=635.0, width=30.0, height=21.0)

        # --- Button Quên mật khẩu ---
        self.dn_forget_image = PhotoImage(file=relative_to_assets("button_forget_password.png"))
        self.dn_forget = Button(
            self,
            image=self.dn_forget_image,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            command=lambda: self.controller.show_frame("Frame04") if self.controller else None
        )
        self.dn_forget.place(x=1149.0, y=680.0, width=141.0, height=20.0)

        # --- Button Đăng nhập ---
        self.dn_submit_image = PhotoImage(file=relative_to_assets("button_login.png"))
        self.dn_submit = Button(
            self,
            image=self.dn_submit_image,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.login_action(),
            relief="flat"
        )
        self.dn_submit.place(x=848.0, y=714.0, width=450.0, height=61.0)

        # --- Button Đăng ký ---
        self.dn_signin_image = PhotoImage(file=relative_to_assets("button_register.png"))
        self.dn_signin = Button(
            self,
            image=self.dn_signin_image,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame02") if self.controller else print("Go to Register"),
            relief="flat"
        )
        self.dn_signin.place(x=1151.0, y=855.0, width=92.0, height=24.0)

        # Bind Enter cho tiện (không ảnh hưởng layout)
        self.dn_username.bind("<Return>", lambda e: self.login_action())
        self.dn_password.bind("<Return>", lambda e: self.login_action())

    # ----------------------------------------------------------------
    def toggle_password_visibility(self):
        """Chuyển đổi giữa ẩn và hiện mật khẩu"""
        if self.password_hidden:
            self.dn_password.config(show="")
            self.password_hidden = False
        else:
            self.dn_password.config(show="*")
            self.password_hidden = True

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    def on_show(self):
        """Được gọi mỗi khi Frame01 hiển thị"""
        print("Frame01 hiển thị lại (reload data nếu cần)")


