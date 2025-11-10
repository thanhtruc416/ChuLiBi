# -*- coding: utf-8 -*-
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox
import re
from typing import Optional
from QMess.Qmess_calling import Qmess
from Function.app_controller import AppController

# === NEW: gọi file chức năng riêng cho màn ex ===
# API: send_otp_if_email_not_exists(email) -> (ok: bool, msg: str)
try:
    from Function.Frame02_ex_SendOTP import send_otp_if_email_not_exists
except Exception:
    send_otp_if_email_not_exists = None  # cho phép demo nếu chưa có module


class Frame02_ex(tk.Frame):
    """
    Step 1 — Enter email to receive OTP (embed vào app đa-frame)
    - Ký hiệu & chữ ký đồng điệu với các frame khác: __init__(parent, controller=None)
    - Không tạo Tk root; dùng chính parent làm master
    - on_otp_sent: nếu cần chuyển màn, hãy gọi qua controller trong Main
    """

    def __init__(self, parent, controller: AppController):
        super().__init__(parent)
        self.controller = controller
        self.lower()

        # ---- Paths
        self.output_path = Path(__file__).parent
        # chỉnh tên thư mục assets của bạn tại đây
        self.assets_path = (self.output_path / "assets").resolve()
        if not self.assets_path.exists():
            raise FileNotFoundError(f"Assets folder not found: {self.assets_path}")

        # giữ tham chiếu để tránh GC ảnh
        self._img_refs: dict[str, PhotoImage] = {}

        # ---- Canvas (chiều rộng/cao theo layout gốc)
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

        # ---- Build UI
        self._build_ui()

    # ----------------------
    # Helpers
    # ----------------------
    def _asset(self, name: str) -> str:
        return str(self.assets_path / name)

    def _img(self, key: str, filename: str) -> PhotoImage:
        if key not in self._img_refs:
            self._img_refs[key] = PhotoImage(file=self._asset(filename))
        return self._img_refs[key]

    @staticmethod
    def _valid_email(email: str) -> bool:
        pat = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pat, email or ""))

    # ----------------------
    # UI composition
    # ----------------------
    def _build_ui(self):
        # Background / panels
        self.canvas.create_image(360.0, 512.0, image=self._img("image_1", "image_1.png"))
        self.canvas.create_image(360.0, 512.0, image=self._img("image_2", "image_2.png"))
        self.canvas.create_image(1079.0, 512.0, image=self._img("image_3", "image_3.png"))

        self.canvas.create_text(
            910.0, 670.0, anchor="nw",
            text="Already have an account ?",
            fill="#000000",
            font=("Crimson Pro Bold", -23)
        )

        self.canvas.create_image(720.0, 512.0, image=self._img("image_4", "image_4.png"))

        self.canvas.create_text(
            980.0, 273.0, anchor="nw",
            text="Create",
            fill="#000000",
            font=("Young Serif", -55)
        )

        self.canvas.create_text(
            277.0, 885.0, anchor="nw",
            text="ChuLiBi",
            fill="#FDE5F4",
            font=("Rubik Burned Regular", -35)
        )

        self.canvas.create_text(
            60.0, 637.0, anchor="nw",
            text="we’ll send you a verification code",
            fill="#FFFFFF",
            font=("Crimson Pro SemiBold", -40)
        )

        self.canvas.create_text(
            202.0, 586.0, anchor="nw",
            text="Enter your email ",
            fill="#FFFFFF",
            font=("Crimson Pro SemiBold", -40)
        )

        self.canvas.create_text(
            910.0, 428.0, anchor="nw",
            text="Email",
            fill="#000000",
            font=("Crimson Pro Bold", -28)
        )

        # Entry background
        self.canvas.create_image(1079.0, 492.0, image=self._img("entry_1", "entry_1.png"))

        # Entry widget
        self.entry_email = Entry(self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0, font=("Crimson Pro Regular", -26))
        self.entry_email.place(x=869.0, y=462.0, width=420.0, height=58.0)

        # Email icon / decoration
        self.canvas.create_image(882.0, 441.0, image=self._img("image_5", "image_5.png"))
        self.canvas.create_image(882.0, 444.0, image=self._img("image_6", "image_6.png"))

        # Send OTP button
        self.button_sendOTP = Button(
            self,
            image=self._img("button_sendOTP", "button_sendOTP.png"),
            borderwidth=0,
            highlightthickness=0,
            command=self.on_send_otp,
            relief="flat",
            cursor="hand2"
        )
        self.button_sendOTP.place(x=842.0, y=574.0, width=479.0, height=70.0)

        # Left side image + title
        self.canvas.create_image(357.0, 322.0, image=self._img("image_7", "image_7.png"))

        self.canvas.create_text(
            76.0, 462.0, anchor="nw",
            text="Join ChuLiBi today",
            fill="#FFFFFF",
            font=("Young Serif", -60)
        )

        # Login button (text-like)
        self.button_login = Button(
            self,
            image=self._img("button_login", "button_login.png"),
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame01"),
            relief="flat",
            cursor="hand2"
        )
        self.button_login.place(x=1184.0, y=670.0, width=57.0, height=23.0)

    # ----------------------
    # Callbacks
    # ----------------------
    def on_send_otp(self):
        email = (self.entry_email.get() or "").strip()
        if not email:
            Qmess.popup_06(parent=self, title="Mising Email",
                        subtitle="Please enter your email first")
            return
        if not self._valid_email(email):
            Qmess.popup_06(parent=self, title="Invalid Email",
                        subtitle="Please enter a valid email address.")
            return

        # Module chức năng riêng
        if send_otp_if_email_not_exists is None:
            # fallback demo
            messagebox.showinfo("OTP (demo)", "send_otp_if_email_not_exists() not found. Demo only.")
            print(f"[DEV] (demo) OTP to {email}: 123456")
            # lưu email & chuyển màn
            if self.controller:
                if not hasattr(self.controller, "current_user") or self.controller.current_user is None:
                    self.controller.current_user = {}
                self.controller.current_user["pending_email"] = email
                # thêm dòng dưới để Frame02 đọc được theo cách cũ
                self.controller.pending_email = email
                try:
                    self.controller.show_frame("Frame02")
                except Exception as e:
                    print(f"show_frame('Frame02') failed: {e}")

            return

        # Gọi hàm gửi OTP thật
        try:
            ok, msg = send_otp_if_email_not_exists(email)
        except Exception as e:
            Qmess.popup_15(parent=self, title="System Error",
                        subtitle=f"Send OTP failed: {e}")
            return

        if not ok:
            Qmess.popup_15(parent=self, title="Failed to send OTP",
                        subtitle=msg or "Please try again.")
            return

        # Thành công
        Qmess.popup_07(parent=self, title="OTP Sent",
                        subtitle="The OTP has been sent successfully!\nPlease check your inbox.")

        if self.controller:
            if not hasattr(self.controller, "current_user") or self.controller.current_user is None:
                self.controller.current_user = {}
            self.controller.current_user["pending_email"] = email
            self.controller.pending_email = email  # để Frame02 đọc theo cách cũ

            try:
                self.controller.show_frame("Frame02")
            except Exception as e:
                print(f"show_frame('Frame02') failed: {e}")
        return

    def on_login(self):
        # nếu có controller.show_frame("Frame01") thì gọi ở đây
        if self.controller and hasattr(self.controller, "show_frame"):
            try:
                self.controller.show_frame("Frame01")
            except Exception:
                pass
        else:
            print("[login] clicked")

    # ----------------------
    # Hook khi frame được show
    # ----------------------
    def on_show(self):
        self.entry_email.delete(0, tk.END)
        self.entry_email.focus()


