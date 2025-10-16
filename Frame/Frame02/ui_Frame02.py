from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, messagebox
import re

# --- Đường dẫn chung ngoài class ---
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_frame02")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Frame02(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # trạng thái hiển thị mật khẩu
        self.password_hidden = True
        self.confirm_password_hidden = True
        self.terms_accepted = False

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

        # --- Images nền ---
        self.image_1_img = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(360.0, 512.0, image=self.image_1_img)

        self.image_2_img = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(360.0, 512.0, image=self.image_2_img)

        self.image_3_img = PhotoImage(file=relative_to_assets("image_3.png"))
        self.image_3 = self.canvas.create_image(1081.0, 513.0, image=self.image_3_img)

        self.image_4_img = PhotoImage(file=relative_to_assets("image_4.png"))
        self.image_4 = self.canvas.create_image(720.0, 512.0, image=self.image_4_img)

        self.image_5_img = PhotoImage(file=relative_to_assets("image_5.png"))
        self.image_5 = self.canvas.create_image(360.0, 340.0, image=self.image_5_img)

        self.image_6_img = PhotoImage(file=relative_to_assets("image_6.png"))
        self.image_6 = self.canvas.create_image(877.0, 556.0, image=self.image_6_img)

        self.image_7_img = PhotoImage(file=relative_to_assets("image_7.png"))
        self.image_7 = self.canvas.create_image(876.0, 694.0, image=self.image_7_img)

        self.image_8_img = PhotoImage(file=relative_to_assets("image_8.png"))
        self.image_8 = self.canvas.create_image(878.0, 543.0, image=self.image_8_img)

        self.image_9_img = PhotoImage(file=relative_to_assets("image_9.png"))
        self.image_9 = self.canvas.create_image(877.0, 681.0, image=self.image_9_img)

        self.image_10_img = PhotoImage(file=relative_to_assets("image_10.png"))
        self.image_10 = self.canvas.create_image(875.0, 287.0, image=self.image_10_img)

        self.image_11_img = PhotoImage(file=relative_to_assets("image_11.png"))
        self.image_11 = self.canvas.create_image(875.0, 270.0, image=self.image_11_img)

        self.image_12_img = PhotoImage(file=relative_to_assets("image_12.png"))
        self.image_12 = self.canvas.create_image(876.0, 414.0, image=self.image_12_img)

        self.image_13_img = PhotoImage(file=relative_to_assets("image_13.png"))
        self.image_13 = self.canvas.create_image(876.0, 417.0, image=self.image_13_img)

        # --- Texts ---
        texts = [
            (915.0, 928.0, "Already have an account ?", "Crimson Pro SemiBold", 23),
            (915.0, 813.0, "I agree to Terms & Privacy Policy", "Crimson Pro SemiBold", 23),
            (875.0, 131.0, "Create Account", "Young Serif", 55),
            (906.0, 539.0, "Password", "Crimson Pro SemiBold", 28),
            (900.0, 680.0, "Confirm password", "Crimson Pro SemiBold", 28),
            (903.0, 271.0, "Username", "Crimson Pro SemiBold", 28),
            (904.0, 401.0, "Email", "Crimson Pro SemiBold", 28),
            (49.0, 31.0, "ChuLiBi", "Rubik Burned Regular", 35, "#FDE5F4"),
            (76.0, 476.0, "Join ChuLiBi today", "Young Serif", 60, "#FFFFFF"),
            (181.0, 598.0, "Let’s start building AI", "Crimson Pro SemiBold", 40, "#FFFFFF"),
            (139.0, 645.0, "powered insights together", "Crimson Pro SemiBold", 40, "#FFFFFF")
        ]
        for t in texts:
            x, y, text, font_name, size = t[:5]
            fill = t[5] if len(t) > 5 else "#000000"
            self.canvas.create_text(x, y, anchor="nw", text=text, fill=fill,
                                    font=(font_name, size * -1))

        # --- Entry ---
        self.entry_image_password = PhotoImage(file=relative_to_assets("entry_password.png"))
        self.canvas.create_image(1083.5, 609.5, image=self.entry_image_password)
        self.entry_password = Entry(self, bd=0, bg="#FFFFFF", fg="#000716",
                                    highlightthickness=0, font=("Crimson Pro Regular", 26 * -1),
                                    show="*")
        self.entry_password.place(x=859.0, y=575.0, width=449.0, height=67.0)

        self.entry_image_password_confirm = PhotoImage(file=relative_to_assets("entry_password_confirm.png"))
        self.canvas.create_image(1083.5, 748.5, image=self.entry_image_password_confirm)
        self.entry_password_confirm = Entry(self, bd=0, bg="#FFFFFF", fg="#000716",
                                            highlightthickness=0, font=("Crimson Pro Regular", 26 * -1),
                                            show="*")
        self.entry_password_confirm.place(x=859.0, y=714.0, width=449.0, height=67.0)

        self.entry_image_username = PhotoImage(file=relative_to_assets("entry_username.png"))
        self.canvas.create_image(1083.5, 338.5, image=self.entry_image_username)
        self.entry_username = Entry(self, bd=0, bg="#FFFFFF", fg="#000716",
                                    highlightthickness=0, font=("Crimson Pro Regular", 26 * -1))
        self.entry_username.place(x=859.0, y=304.0, width=449.0, height=67.0)

        self.entry_image_email = PhotoImage(file=relative_to_assets("entry_email.png"))
        self.canvas.create_image(1083.5, 472.5, image=self.entry_image_email)
        self.entry_email = Entry(self, bd=0, bg="#FFFFFF", fg="#000716",
                                 highlightthickness=0, font=("Crimson Pro Regular", 26 * -1))
        self.entry_email.place(x=859.0, y=438.0, width=449.0, height=67.0)

        # --- Buttons: Register / Login ---
        self.button_register_img = PhotoImage(file=relative_to_assets("button_register.png"))
        self.button_register = Button(self, image=self.button_register_img,
                                      borderwidth=0, highlightthickness=0,
                                      command=self.register_action, relief="flat", cursor="hand2")
        self.button_register.place(x=839.0, y=848.0, width=473.0, height=68.0)

        self.button_login_img = PhotoImage(file=relative_to_assets("button_login.png"))
        self.button_login = Button(self, image=self.button_login_img,
                                   borderwidth=0, highlightthickness=0,
                                   command=lambda: self.controller.show_frame("Frame01") if self.controller else None,
                                   relief="flat", cursor="hand2")
        self.button_login.place(x=1169.0, y=933.0, width=67.0, height=21.0)

        # --- Checkbox Terms ---
        self.button_3_img = PhotoImage(file=relative_to_assets("button_3.png"))
        self.button_2 = Button(self, image=self.button_3_img,
                               borderwidth=0, highlightthickness=0,
                               command=self.toggle_terms, relief="flat", cursor="hand2")
        self.button_2.place(x=887.0, y=810.0, width=21.5, height=23.0)

        # --- Eye Icons (mới): dùng eye_hide / eye_show + hover ---
        self.eye_hide_img = PhotoImage(file=relative_to_assets("eye_hide.png"))
        self.eye_show_img = PhotoImage(file=relative_to_assets("eye_show.png"))

        # Eye cho Password
        self.button_eye_pw = Button(self, image=self.eye_hide_img,
                                    borderwidth=0, highlightthickness=0,
                                    command=self.toggle_password_visibility,
                                    relief="flat", cursor="hand2")
        self.button_eye_pw.place(x=1270.0, y=597.0, width=30.0, height=21.0)
        self.button_eye_pw.bind("<Enter>", lambda e: self.button_eye_pw.config(image=self.eye_show_img))
        self.button_eye_pw.bind("<Leave>", lambda e: self._sync_eye_icons())

        # Eye cho Confirm Password
        self.button_eye_cpw = Button(self, image=self.eye_hide_img,
                                     borderwidth=0, highlightthickness=0,
                                     command=self.toggle_confirm_password_visibility,
                                     relief="flat", cursor="hand2")
        self.button_eye_cpw.place(x=1270.0, y=738.0, width=30.0, height=21.0)
        self.button_eye_cpw.bind("<Enter>", lambda e: self.button_eye_cpw.config(image=self.eye_show_img))
        self.button_eye_cpw.bind("<Leave>", lambda e: self._sync_eye_icons())

        # Đồng bộ icon ban đầu
        self._sync_eye_icons()

    # ----------------------------------------------------------------
    def _sync_eye_icons(self):
        """Đồng bộ icon theo trạng thái show/hide của 2 ô mật khẩu."""
        self.button_eye_pw.config(image=self.eye_hide_img if self.password_hidden else self.eye_show_img)
        self.button_eye_cpw.config(image=self.eye_hide_img if self.confirm_password_hidden else self.eye_show_img)

    def toggle_password_visibility(self):
        """Toggle password visibility + sync icon"""
        self.password_hidden = not self.password_hidden
        self.entry_password.config(show="*" if self.password_hidden else "")
        self._sync_eye_icons()

    def toggle_confirm_password_visibility(self):
        """Toggle confirm password visibility + sync icon"""
        self.confirm_password_hidden = not self.confirm_password_hidden
        self.entry_password_confirm.config(show="*" if self.confirm_password_hidden else "")
        self._sync_eye_icons()

    def toggle_terms(self):
        """Toggle terms and conditions acceptance"""
        self.terms_accepted = not self.terms_accepted
        # đổi icon checkbox
        self.button_2_img = PhotoImage(file=relative_to_assets("button_2.png"))
        self.button_2.config(image=self.button_2_img if self.terms_accepted else self.button_3_img)
        if self.terms_accepted:
            print("Terms accepted")
        else:
            print("Terms not accepted")

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def register_action(self):
        """Handle user registration"""
        username = self.entry_username.get().strip()
        email = self.entry_email.get().strip()
        password = self.entry_password.get().strip()
        confirm_password = self.entry_password_confirm.get().strip()

        # Check if terms are accepted
        if not self.terms_accepted:
            messagebox.showwarning(
                "Terms & Conditions",
                "Please accept the Terms & Privacy Policy to continue"
            )
            return

        # Validate email format
        if email and not self.validate_email(email):
            messagebox.showerror(
                "Invalid Email",
                "Please enter a valid email address"
            )
            return

        try:
            from Function.auth import AuthService

            # Attempt registration
            result = AuthService.register_user(username, email, password, confirm_password)

            if result["success"]:
                messagebox.showinfo(
                    "Registration Successful",
                    "Account created! Please complete your profile."
                )

                if self.controller:
                    if not hasattr(self.controller, 'current_user'):
                        self.controller.current_user = {}
                    self.controller.current_user = result['user_data']
                    self.controller.current_user['profile_completed'] = 0

                self.clear_form()

                if self.controller:
                    try:
                        self.controller.show_frame("Frame03")
                    except KeyError:
                        self.controller.show_frame("Frame01")

            else:
                messagebox.showerror(
                    "Registration Failed",
                    result["message"]
                )

        except ImportError as e:
            messagebox.showerror(
                "System Error",
                f"Authentication module not found: {str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "System Error",
                f"An unexpected error occurred: {str(e)}"
            )
            print(f"Registration error: {e}")

    def clear_form(self):
        """Clear all form fields"""
        self.entry_username.delete(0, tk.END)
        self.entry_email.delete(0, tk.END)
        self.entry_password.delete(0, tk.END)
        self.entry_password_confirm.delete(0, tk.END)
        self.terms_accepted = False
        self.password_hidden = True
        self.confirm_password_hidden = True
        self.entry_password.config(show="*")
        self.entry_password_confirm.config(show="*")
        self._sync_eye_icons()

    def on_show(self):
        """Called when Frame02 is displayed"""
        print("Frame02 displayed")
        self.clear_form()
        self.entry_username.focus()
