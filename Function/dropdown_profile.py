import tkinter as tk

class DropdownMenu:
    def __init__(self, parent, controller=None, x_offset=1270, y_offset=72, width=149):

        self.parent = parent
        self.controller = controller or getattr(parent, "controller", None)
        self.dropdown_window = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width

    def show(self):
        # Nếu dropdown đang mở thì đóng lại
        if self.dropdown_window and self.dropdown_window.winfo_exists():
            self.close()
            return

        # Cửa sổ nổi
        self.dropdown_window = tk.Toplevel(self.parent)
        self.dropdown_window.overrideredirect(True)
        self.dropdown_window.configure(bg="#FFFFFF")
        self.dropdown_window.attributes("-topmost", True)

        # Vị trí hiển thị
        x = self.parent.winfo_rootx() + self.x_offset
        y = self.parent.winfo_rooty() + self.y_offset
        self.dropdown_window.geometry(f"{self.width}x82+{x}+{y}")

        # Frame bao
        container = tk.Frame(
            self.dropdown_window,
            bg="#FFFFFF",
            highlightbackground="#CFC6DC",
            highlightthickness=1,
            bd=0,
        )
        container.pack(fill="both", expand=True)

        # Danh sách menu (font_size phải là int)
        buttons = [
            ("Account Info", self.open_account_info, 11, (7, 6)),
            ("Log Out", self.logout, 11, (7, 6)),
        ]

        for i, (text, cmd, font_size, pady) in enumerate(buttons):
            btn = tk.Label(
                container,
                text=text,
                bg="#FFFFFF",
                fg="#5A3372",
                font=("Crimson Pro", font_size, "bold"),
                anchor="w",
                padx=15,
                pady=0,
                cursor="hand2",
            )
            btn.pack(fill="x", pady=pady)

            # Hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#E9E2F2"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#FFFFFF"))
            btn.bind("<Button-1>", lambda e, c=cmd: c())

            # Separator
            if i < len(buttons) - 1:
                sep = tk.Frame(container, bg="#DDD", height=1)
                sep.pack(fill="x")

        # Tự đóng khi mất focus
        self.dropdown_window.bind("<FocusOut>", lambda e: self.close())
        self.dropdown_window.focus_force()

    def close(self):
        if self.dropdown_window and self.dropdown_window.winfo_exists():
            self.dropdown_window.destroy()
            self.dropdown_window = None

    def open_account_info(self):
        """Mở trang thông tin tài khoản."""
        self.close()
        if not self.controller:
            print("Không có controller — cần truyền controller khi khởi tạo DropdownMenu.")
            return

        # import ở đây để tránh circular import
        from Function import Frame12_ChangeInformation

        user_data = self.controller.get_current_user()
        if not user_data:
            print("Không tìm thấy thông tin người dùng hiện tại.")
            return

        self.controller.show_frame("Frame12", user_data=user_data)
        print(f"Đã mở Account Info cho user: {user_data.get('username', '')}")

    def logout(self):
        self.close()
        if not self.controller:
            print("Không có controller — cần truyền controller khi khởi tạo DropdownMenu.")
            return

        try:
            self.controller.clear_current_user()
            self.controller.show_frame("Frame01")
            print("Đăng xuất thành công.")
        except Exception as e:
            print("Lỗi khi logout:", e)
