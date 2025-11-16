# frame03.py
from pathlib import Path
from tkinter import Frame, Canvas, Entry, Button, PhotoImage, messagebox
import tkinter as tk
from QMess.Qmess_calling import Qmess
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame03")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Frame03(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.lower()
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
        # --- thay entry_your_role bằng dropdown ---
        self.entry_image_yourrole = PhotoImage(file=relative_to_assets("entry_full_name.png"))
        canvas.create_image(1081.5, 755.5, image=self.entry_image_yourrole)

        self.entry_your_role = Entry(
            self,
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            font=("Crimson Pro Regular", 26 * -1)
        )
        self.entry_your_role.place(x=857.0, y=721.0, width=449.0, height=67.0)
        self.role_var = ""  # lưu giá trị đã chọn
        self.role_button_img = PhotoImage(file=relative_to_assets("button_26.png"))
        self.role_button = Button(
            self,
            image=self.role_button_img,
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg="#FFFFFF",
            activebackground="#FFFFFF",
            command=lambda: self.show_dropdown(self.entry_your_role, "Role")
        )
        self.role_button.place(x=1280, y=738)

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
                                      command=self.save_profile, relief="flat")
        self.button_continue.place(x=843.0, y=822.0, width=482.0, height=69.0)

    def save_profile(self):
        """Save user profile information"""
        full_name = self.entry_full_name.get().strip()
        business_name = self.entry_business_name.get().strip()
        role = getattr(self, "role_var", "").strip()
        if not role:
            Qmess.popup_13(parent=self, title="Incomplete Profile",
                           subtitle="Please select your role")
            return

        # Validate inputs
        if not full_name or not business_name or not role:
            Qmess.popup_13(parent=self, title="Incomplete Profile",
                        subtitle="Please fill in all fields to complete your profile")
            return
        # Get current user from controller
        if not self.controller or not hasattr(self.controller, 'current_user'):
            Qmess.popup_29(parent=self, title="Error",
                        subtitle="No user session found. Please login again.")
            if self.controller:
                self.controller.show_frame("Frame01")
            return
        user_id = self.controller.current_user.get('id')
        if not user_id:
            Qmess.popup_29(parent=self, title="Error",
                        subtitle="No user session found. Please login again.")
            self.controller.show_frame("Frame01")
            return
        try:
            from Function.Frame03_Profile import AuthService

            # Update profile
            result = AuthService.update_user_profile(user_id, full_name, business_name, role)
            if result["success"]:
                # Update current user data
                self.controller.current_user['full_name'] = full_name
                self.controller.current_user['business_name'] = business_name
                self.controller.current_user['role'] = role

                Qmess.popup_14(parent=self, title="Profile Completed",
                        subtitle=f"Welcome, {full_name}! Your profile has been completed successfully! Please login again.")

                if self.controller:
                    self.controller.current_user = {}
                    try:
                        self.controller.show_frame("Frame01")
                    except KeyError:
                        print("Login frame (Frame01) not found")
            else:
                messagebox.showerror(
                    "Update Failed",
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
            print(f"Profile save error: {e}")

    def load_user_data(self):
        """Load existing user data if available"""
        if self.controller and hasattr(self.controller, 'current_user'):
            user = self.controller.current_user

            # Pre-fill fields if data exists
            if user.get('full_name'):
                self.entry_full_name.delete(0, tk.END)
                self.entry_full_name.insert(0, user['full_name'])

            if user.get('business_name'):
                self.entry_business_name.delete(0, tk.END)
                self.entry_business_name.insert(0, user['business_name'])

            self.role_var = user.get('role', "")
            self.entry_your_role.delete(0, tk.END)
            role = user.get('role') or ""
            self.entry_your_role.insert(0, str(role))

    def clear_form(self):
        """Clear all form fields"""
        self.entry_full_name.delete(0, tk.END)
        self.entry_business_name.delete(0, tk.END)
        self.entry_your_role.delete(0, tk.END)

    def on_show(self):
        """Called when Frame03 is displayed"""
        print("Frame03 displayed - Complete Profile")
        # Load existing data if any
        self.load_user_data()
        # Focus on first field
        self.entry_full_name.focus()

    def show_dropdown(self, entry_widget, key):
        """Hiển thị dropdown ChuLiBi ngay dưới entry"""
        # Giá trị dropdown theo key
        dropdown_values = {
            "Role": ["User", "Admin"]
        }
        values = dropdown_values.get(key, [])
        if not values:
            return

        # Tạo popup
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)  # xóa viền window
        popup.config(bg="#FFFFFF")

        # Vị trí popup ngay dưới entry
        x = self.winfo_rootx() + entry_widget.winfo_x()
        y = self.winfo_rooty() + entry_widget.winfo_y() + entry_widget.winfo_height()
        popup.geometry(f"170x{len(values) * 30}+{x}+{y}")

        # Hàm chọn giá trị
        def on_select(val):
            self.role_var = val
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, str(val))
            popup.destroy()

        # Tạo label cho mỗi lựa chọn
        for val in values:
            lbl = tk.Label(
                popup,
                text=val,
                bg="#FFFFFF",
                fg="#745fa3",
                font=("Crimson Pro SemiBold", 12),
                anchor="w",
                padx=12,
                pady=3
            )
            lbl.pack(fill="x")

            # Chọn giá trị khi click
            lbl.bind("<Button-1>", lambda e, v=val: on_select(v))
            # Hover effect
            lbl.bind("<Enter>", lambda e, l=lbl: l.config(bg="#EDE6F9", fg="#2E1E5B"))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(bg="#FFFFFF", fg="#745fa3"))

        # Tự động ẩn khi mất focus
        popup.focus_force()
        popup.bind("<FocusOut>", lambda e: popup.destroy())
if __name__ == "__main__":
    import tkinter as tk

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("ChuLiBi - Complete Profile")
            self.geometry("1440x1024")
            self.current_user = {"id": 1, "full_name": "", "business_name": "", "role": ""}

            # Tạo frame03
            self.frame03 = Frame03(parent=self, controller=self)
            self.frame03.place(x=0, y=0, relwidth=1, relheight=1)

            # Hiển thị frame03
            self.frame03.lift()
            if hasattr(self.frame03, "on_show"):
                self.frame03.on_show()

    # Khởi chạy app
    app = App()
    app.mainloop()


