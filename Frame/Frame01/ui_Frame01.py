# ui_Frame01.py
import sys
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox
_project_root = Path(__file__).resolve().parents[2]  # Frame/Frame01 -> Frame -> project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Function.Frame01_auth import login_with_password
from QMess.Qmess_calling import Qmess
# ========== Đường dẫn asset ==========
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
            self, bg="#FFFFFF", height=1024, width=1440,
            bd=0, highlightthickness=0, relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # --- Load hình ảnh nền (nếu thiếu thì bỏ qua) ---
        self.images = {}  # giữ reference tránh GC
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
            ("image_14.png", 1100, 810),
        ]
        for filename, x, y in img_files:
            try:
                img = PhotoImage(file=relative_to_assets(filename))
                self.images[filename] = img
                self.canvas.create_image(x, y, image=img)
            except Exception:
                pass

        # --- Text ---
        self.canvas.create_text(921.0, 851.0, anchor="nw",
                                text="Don’t have an account?",
                                fill="#000000",
                                font=("Crimson Pro SemiBold", -23))
        self.canvas.create_text(957.0, 792.0, anchor="nw",
                                text="Or", fill="#000000",
                                font=("Crimson Pro SemiBold", -23))
        self.canvas.create_text(883.0, 456.0, anchor="nw",
                                text="Username", fill="#000000",
                                font=("Crimson Pro SemiBold", -28))
        self.canvas.create_text(969.0, 345.0, anchor="nw",
                                text="Login", fill="#000000",
                                font=("Young Serif", -64))
        self.canvas.create_text(883.0, 570.0, anchor="nw",
                                text="Password", fill="#000000",
                                font=("Crimson Pro SemiBold", -28))
        self.canvas.create_text(80.0, 347.0, anchor="nw",
                                text="Welcome back to", fill="#000000",
                                font=("Young Serif", -65))
        self.canvas.create_text(206.0, 454.0, anchor="nw",
                                text="ChuLiBi", fill="#000000",
                                font=("Young Serif", -70))
        self.canvas.create_text(33.0, 595.0, anchor="nw",
                                text="Your AI data buddy for smarter insights",
                                fill="#000000",
                                font=("Crimson Pro SemiBold", -40))
        self.canvas.create_text(49.0, 31.0, anchor="nw",
                                text="ChuLiBi", fill="#FDE5F4",
                                font=("Rubik Burned Regular", -35))

        # --- Entry Username ---
        try:
            self.dn_username_image = PhotoImage(file=relative_to_assets("entry_username.png"))
            self.canvas.create_image(1065.0, 524.0, image=self.dn_username_image)
        except Exception:
            pass

        self.dn_username = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", -26)
        )
        self.dn_username.place(x=855.0, y=494.0, width=420.0, height=58.0)

        # --- Entry Password ---
        try:
            self.dn_password_image = PhotoImage(file=relative_to_assets("entry_password.png"))
            self.canvas.create_image(1065.0, 644.0, image=self.dn_password_image)
        except Exception:
            pass
        self.dn_password = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", -26)
        )
        self.dn_password.place(x=855.0, y=614.0, width=420.0, height=58.0)
        # Bind Enter để submit nhanh
        self.dn_username.bind("<Return>", lambda e: self.login_action())
        self.dn_password.bind("<Return>", lambda e: self.login_action())
        self.dn_password = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", -26),
            show="*"
        )
        self.dn_password.place(x=855.0, y=614.0, width=420.0, height=58.0)
        # --- Button hiện/ẩn mật khẩu (auto-resize icon) ---
        self.eye_imgs = self._load_eye_icons()

        self.dn_eye = Button(
            self,
            image=self.eye_imgs["hide"],  # bắt đầu: đang ẩn
            borderwidth=0, highlightthickness=0,
            relief="flat", cursor="hand2",
            command=self.toggle_password_visibility
        )
        # Không set width/height để button fit theo icon
        # (hoặc dùng anchor="center" nếu muốn căn tâm)
        self.dn_eye.place(x=1243.0, y=635.0)

        # Hover (dùng lại ảnh thường vì bạn chỉ có 2 ảnh)
        self.dn_eye.bind("<Enter>", self._eye_enter)
        self.dn_eye.bind("<Leave>", self._eye_leave)

        # --- Button Quên mật khẩu ---
        try:
            self.dn_forget_image = PhotoImage(file=relative_to_assets("button_forget_password.png"))
        except Exception:
            self.dn_forget_image = None

        self.dn_forget = Button(
            self,
            image=self.dn_forget_image,
            borderwidth=0, highlightthickness=0, relief="flat",
            command=(lambda: self.controller.show_frame("Frame04")) if self.controller else None
        )
        self.dn_forget.place(x=1149.0, y=680.0, width=141.0, height=20.0)

        # --- Button Đăng nhập ---
        try:
            self.dn_submit_image = PhotoImage(file=relative_to_assets("button_login.png"))
        except Exception:
            self.dn_submit_image = None

        self.dn_submit = Button(
            self,
            image=self.dn_submit_image,
            borderwidth=0, highlightthickness=0, relief="flat",
            command=self.login_action
        )
        self.dn_submit.place(x=848.0, y=714.0, width=450.0, height=61.0)

        # --- Button Đăng ký ---
        try:
            self.dn_signin_image = PhotoImage(file=relative_to_assets("button_register.png"))
        except Exception:
            self.dn_signin_image = None

        self.dn_signin = Button(
            self,
            image=self.dn_signin_image,
            borderwidth=0, highlightthickness=0, relief="flat",
            command=(lambda: self.controller.show_frame("Frame02_ex")) if self.controller else (lambda: print("Go to Register"))
        )
        self.dn_signin.place(x=1151.0, y=855.0, width=92.0, height=24.0)

    # ===================== ICON HELPERS =====================
    def _load_eye_icons(self):
        """
        Nạp 2 icon eye_hide/eye_show. Nếu ảnh quá lớn sẽ tự scale về chiều cao ICON_TARGET_H.
        Nếu Pillow có sẵn -> resize mượt; nếu không -> fallback PhotoImage.subsample.
        """
        ICON_TARGET_H = 24  # chiều cao mong muốn (px)

        def load_and_scale(path: Path, target_h: int):
            # Ưu tiên Pillow nếu có
            try:
                from PIL import Image, ImageTk
                im = Image.open(path).convert("RGBA")
                w, h = im.size
                if h != target_h:
                    new_w = max(1, int(round(w * (target_h / float(h)))))
                    im = im.resize((new_w, target_h), Image.LANCZOS)
                return ImageTk.PhotoImage(im)
            except Exception:
                # Fallback: dùng PhotoImage + subsample integer
                img = PhotoImage(file=path)
                h = img.height()
                if h > target_h:
                    factor = max(1, round(h / target_h))
                    try:
                        img = img.subsample(factor, factor)
                    except Exception:
                        pass
                return img

        hide_path = relative_to_assets("eye_hide.png")
        show_path = relative_to_assets("eye_show.png")
        if not hide_path.exists():
            raise RuntimeError("Thiếu ảnh eye_hide.png trong assets_Frame01/")
        if not show_path.exists():
            raise RuntimeError("Thiếu ảnh eye_show.png trong assets_Frame01/")

        icons = {
            "hide": load_and_scale(hide_path, ICON_TARGET_H),
            "show": load_and_scale(show_path, ICON_TARGET_H),
        }
        # Hover dùng lại ảnh thường (bạn chỉ có 2 ảnh)
        icons["hide_hover"] = icons["hide"]
        icons["show_hover"] = icons["show"]
        return icons

    # ===================== TOGGLE & HOVER =====================
    def toggle_password_visibility(self):
        """Chuyển đổi giữa ẩn và hiện mật khẩu + đổi icon."""
        self.password_hidden = not self.password_hidden
        if self.password_hidden:
            self.dn_password.config(show="*")
            self.dn_eye.config(image=self.eye_imgs["hide"])
        else:
            self.dn_password.config(show="")
            self.dn_eye.config(image=self.eye_imgs["show"])

    def _eye_enter(self, _event):
        """Hover vào nút mắt."""
        if self.password_hidden:
            self.dn_eye.config(image=self.eye_imgs["hide_hover"])
        else:
            self.dn_eye.config(image=self.eye_imgs["show_hover"])

    def _eye_leave(self, _event):
        """Rời chuột khỏi nút mắt."""
        if self.password_hidden:
            self.dn_eye.config(image=self.eye_imgs["hide"])
        else:
            self.dn_eye.config(image=self.eye_imgs["show"])

    # ===================== LOGIN FLOW =====================
    def login_action(self):
        """Xử lý đăng nhập"""
        username = self.dn_username.get().strip()
        password = self.dn_password.get().strip()

        if not username or not password:
            Qmess.popup_01(parent=self, title="Missing data",
                        subtitle="Please enter both username and password!")
            return

        try:
            ok, data = login_with_password(username, password)

            if ok:
                # data là dict {"id","email","username"}
                user_data = data

                # lưu vào controller (nếu có)
                if self.controller:
                    if not hasattr(self.controller, "current_user"):
                        self.controller.current_user = {}
                    self.controller.current_user = user_data

                # clear password field
                self.dn_password.delete(0, tk.END)

                # vì API login trả về chưa có full_name/business_name/role
                # => xem như chưa hoàn tất hồ sơ -> chuyển về Frame03 để bổ sung
                profile_complete = (
                        user_data.get("full_name")
                        and user_data.get("business_name")
                        and user_data.get("role")
                )

                if not profile_complete:
                    Qmess.popup_15(parent=self, title="Complete Your Profile",
                        subtitle=f"Welcome, {user_data['username']}! Please complete your profile to continue.")
                    if self.controller:
                        try:
                            self.controller.show_frame("Frame03")
                        except KeyError:
                            pass
                else:
                    Qmess.popup_03(parent=self, title="Login Successful",
                        subtitle=f"Welcome back, {data['full_name']}!")
                    if self.controller:
                        try:
                            self.controller.show_frame("Frame06")
                        except KeyError:
                            pass
            else:
                # data lúc này là thông báo lỗi (string)
                Qmess.popup_02(parent=self, title="Invalid Login",
                        subtitle="Please check your username or password.")
                self.dn_password.delete(0, tk.END)
                self.dn_password.focus()

        except ImportError as e:
            Qmess.popup_02(parent=self, title="System Error",
                        subtitle=f"Authentication module not found: {str(e)}")
        except Exception as e:
            Qmess.popup_02(parent=self, title="System Error",subtitle=f"An unexpected error occurred: {str(e)}")
            print(f"Login error: {e}")

    # ===================== LIFECYCLE =====================
    def on_show(self):
        """Được gọi mỗi khi Frame01 hiển thị"""
        # Clear form
        self.dn_username.delete(0, tk.END)
        self.dn_password.delete(0, tk.END)
        # Reset password visibility + icon
        self.dn_password.config(show="*")
        self.password_hidden = True
        if hasattr(self, "dn_eye"):
            self.dn_eye.config(image=self.eye_imgs["hide"])
        # Focus
        self.dn_username.focus()

