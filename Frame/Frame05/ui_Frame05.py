# frame05.py
from pathlib import Path
from tkinter import Frame, Canvas, Entry, Button, PhotoImage, messagebox, StringVar, NORMAL, DISABLED
from QMess.Qmess_calling import Qmess
import tkinter as tk
# ===== functions: lấy username theo email + verify OTP & reset mật khẩu =====
from Function.Frame05_ResetPassword import get_username_by_email, reset_password_with_otp

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame05")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame05(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#FFFFFF")
        self.lower()

        # sẽ nhận từ frame trước qua on_show(email=...)
        self.email = None

        # ---- bind biến để đọc/ghi dễ hơn ----
        self.username_var = StringVar()
        self.newpw_var = StringVar()
        self.confirmpw_var = StringVar()
        self.otp_var = StringVar()

        # ---- trạng thái hiển thị password (để đồng bộ icon) ----
        self.pw1_visible = False
        self.pw2_visible = False

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

        # Ảnh cho các nút
        self.button_image_resetpassword = PhotoImage(file=relative_to_assets("button_resetpassword.png"))

        # 2 ảnh mắt: hide/show (đặt thuộc tính self để tránh GC)
        self.eye_hide_img = PhotoImage(file=relative_to_assets("eye_hide.png"))
        self.eye_show_img = PhotoImage(file=relative_to_assets("eye_show.png"))

        # --- Text ---
        canvas.create_text(865.0, 115.0, anchor="nw", text="Reset Password", fill="#000000",
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
        # Ảnh nền bo tròn cho ô Username
        self.image_username_bg = PhotoImage(file=relative_to_assets("image_username.png"))
        self.username_bg_item = canvas.create_image(1083.5, 341.0, image=self.image_username_bg)

        # Label hiển thị username nằm đè lên ảnh (chữ thay đổi được)
        self.username_text = tk.Label(
            self,
            text="",
            bg="#D9D9D9",
            fg="#000716",
            font=("Crimson Pro Regular", 26),
            anchor="w"
        )
        self.username_text.place(x=859.0, y=306.5, width=449.0, height=67.0)

        self.entry_image_newpassword = PhotoImage(file=relative_to_assets("entry_newpassword.png"))
        canvas.create_image(1083.5, 488.5, image=self.entry_image_newpassword)
        self.entry_newpassword = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1), textvariable=self.newpw_var, show="*"
        )
        self.entry_newpassword.place(x=859.0, y=454.0, width=449.0, height=67.0)

        self.entry_image_confirm_password = PhotoImage(file=relative_to_assets("entry_confirm_password.png"))
        canvas.create_image(1083.5, 627.5, image=self.entry_image_confirm_password)
        self.entry_confirm_newpassword = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1), textvariable=self.confirmpw_var, show="*"
        )
        self.entry_confirm_newpassword.place(x=859.0, y=593.0, width=449.0, height=67.0)

        self.entry_image_OTP = PhotoImage(file=relative_to_assets("entry_OTP.png"))
        canvas.create_image(1083.5, 750.5, image=self.entry_image_OTP)
        self.entry_OTP = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1), textvariable=self.otp_var
        )
        self.entry_OTP.place(x=859.0, y=716.0, width=449.0, height=67.0)

        # --- Buttons ---
        # Nút mắt cho "New Password"
        self.button_eyes_password = Button(
            self, image=self.eye_hide_img, borderwidth=0,
            highlightthickness=0, command=self.toggle_pw1, relief="flat", cursor="hand2"
        )
        self.button_eyes_password.place(x=1270.0, y=476.0, width=30.0, height=21.0)

        # Nút mắt cho "Confirm password"
        self.button_eyes = Button(
            self, image=self.eye_hide_img, borderwidth=0, highlightthickness=0,
            command=self.toggle_pw2, relief="flat", cursor="hand2"
        )
        self.button_eyes.place(x=1270.0, y=617.0, width=30.0, height=21.0)

        # Hover bindings (chỉ đổi ảnh khi rê chuột; rời chuột thì trả về theo trạng thái)
        self.button_eyes_password.bind("<Enter>", lambda e: self.button_eyes_password.config(image=self.eye_show_img))
        self.button_eyes_password.bind("<Leave>", lambda e: self._sync_eye_icons())

        self.button_eyes.bind("<Enter>", lambda e: self.button_eyes.config(image=self.eye_show_img))
        self.button_eyes.bind("<Leave>", lambda e: self._sync_eye_icons())

        self.button_resetpassword = Button(
            self, image=self.button_image_resetpassword, borderwidth=0,
            highlightthickness=0, command=self.on_reset_password, relief="flat", cursor="hand2"
        )
        self.button_resetpassword.place(x=844.0, y=817.0, width=484.0, height=67.0)

        # Đồng bộ icon lần đầu
        self._sync_eye_icons()

    # ========= lifecycle: nhận email từ frame trước & fill username =========
    def on_show(self, email=None):
        self.email = (email or "").strip().lower()
        self.newpw_var.set(""); self.confirmpw_var.set(""); self.otp_var.set("")
        self.username_var.set("")
        self.pw1_visible = False
        self.pw2_visible = False
        self.entry_newpassword.config(show="*")
        self.entry_confirm_newpassword.config(show="*")
        self._sync_eye_icons()

        if not self.email:
            messagebox.showwarning("Cảnh báo", "Thiếu email để đặt lại mật khẩu")
            return

        ok, res = get_username_by_email(self.email)
        if ok:
            self.username_text.config(text=res)
        else:
            messagebox.showerror("Lỗi", res)

    # ========= actions =========
    def on_reset_password(self):
        new_pw = self.newpw_var.get()
        confirm_pw = self.confirmpw_var.get()
        otp = self.otp_var.get().strip()

        if not otp:
            Qmess.popup_18(parent=self,
                        title="Warning",
                        subtitle="Please enter the OTP.")
            return
        if not new_pw or not confirm_pw:
            Qmess.popup_19(parent=self,
                        title="Warning",
                        subtitle="Please enter your new password.")
            return
        if new_pw != confirm_pw:
            Qmess.popup_20(parent=self,
                        title="Warning",
                        subtitle="The passwords do not match.")
            return
        if len(new_pw) < 8:
            Qmess.popup_21(parent=self,
                        title="Warning",
                        subtitle="The password must be at least 8 characters long.")
            return

        self.button_resetpassword.config(state=DISABLED)
        try:
            ok, msg = reset_password_with_otp(self.email, otp, new_pw)
            if ok:
                Qmess.popup_22(parent=self,
                            title="Success",subtitle=msg)
                if self.controller:
                    self.controller.show_frame("Frame01")  # quay về màn đăng nhập
            else:
                Qmess.popup_29(parent=self,
                        title="Error",
                        subtitle=f"Unable to reset password")
        except Exception as e:
            Qmess.popup_29(parent=self,
                        title="Error",
                        subtitle=f"Unable to reset password: {e}")
        finally:
            self.button_resetpassword.config(state=NORMAL)

    # ========= helpers: show/hide password + đồng bộ icon =========
    def toggle_pw1(self):
        # Đảo trạng thái
        self.pw1_visible = not self.pw1_visible
        self.entry_newpassword.config(show="" if self.pw1_visible else "*")
        self._sync_eye_icons()

    def toggle_pw2(self):
        # Đảo trạng thái
        self.pw2_visible = not self.pw2_visible
        self.entry_confirm_newpassword.config(show="" if self.pw2_visible else "*")
        self._sync_eye_icons()

    def _sync_eye_icons(self):
        """
        Đồng bộ icon theo trạng thái hiển thị của 2 ô password:
        - Visible -> eye_show
        - Hidden  -> eye_hide
        (Hàm này cũng được gọi khi <Leave> để trả icon về đúng trạng thái)
        """
        self.button_eyes_password.config(image=self.eye_show_img if self.pw1_visible else self.eye_hide_img)
        self.button_eyes.config(image=self.eye_show_img if self.pw2_visible else self.eye_hide_img)

        # Bind Enter để gọi lại handler (giống Frame01)
        try:
            self.entry_newpassword.bind("<Return>", lambda _e: self.on_reset_password())
            self.entry_confirm_newpassword.bind("<Return>", lambda _e: self.on_reset_password())
            self.entry_OTP.bind("<Return>", lambda _e: self.on_reset_password())
            self.bind("<Return>", lambda _e: self.on_reset_password())
        except Exception:
            pass


# ==========================
# CHẠY ĐỘC LẬP FRAME05
# ==========================
if __name__ == "__main__":
    import tkinter as tk

    # Mock (giả lập) 2 hàm trong Frame05_ResetPassword để test độc lập
    def get_username_by_email(email):
        if email == "thanhtruc.yee@gmail.com":
            return True, "thanhtruc"
        elif email == "truc@gmail.com":
            return True, "Cao Trúc"
        else:
            return False, "Email không tồn tại"

    def reset_password_with_otp(email, otp, new_pw):
        if otp == "123456":
            return True, f"Đặt lại mật khẩu cho {email} thành công!"
        else:
            return False, "OTP không hợp lệ"

    # Mock controller (để không lỗi show_frame)
    class DummyController:
        def show_frame(self, name):
            print(f"Điều hướng tới: {name}")

    # Tạo cửa sổ Tkinter
    root = tk.Tk()
    root.title("Test Frame05 - Reset Password")
    root.geometry("1440x1024")
    root.configure(bg="#FFFFFF")

    # Khởi tạo frame
    frame = Frame05(root, DummyController())
    frame.pack(fill="both", expand=True)

    # Giả lập gọi từ frame trước (email người dùng nhập)
    frame.on_show(email="thanhtruc.yee@gmail.com")

    root.mainloop()
