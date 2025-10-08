from tkinter import Toplevel, Button


class DropdownMenu:
    def __init__(self, parent, x_offset=1275, y_offset=65, width=150):
        """
        parent: frame hoặc window chứa dropdown
        x_offset, y_offset: vị trí so với parent
        width: chiều rộng của dropdown
        """
        self.parent = parent
        self.dropdown_window = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width

    def show(self):
        # Nếu dropdown đang mở thì đóng
        if self.dropdown_window and self.dropdown_window.winfo_exists():
            self.close()
            return

        # Tạo cửa sổ Toplevel
        self.dropdown_window = Toplevel(self.parent)
        self.dropdown_window.overrideredirect(True)
        self.dropdown_window.configure(bg="#FFFFFF")

        # Tính vị trí
        x = self.parent.winfo_rootx() + self.x_offset
        y = self.parent.winfo_rooty() + self.y_offset

        # Chiều cao đủ cho 3 nút, tự động dựa vào số nút * 35px + padding
        height = 3 * 40 + 15  # Increased height and padding for even spacing
        self.dropdown_window.geometry(f"{self.width}x{height}+{x}+{y}")

        # --- Danh sách nút và callback ---
        buttons = [
            ("Data Management", lambda: print("Data Management clicked")),
            ("Account Info", lambda: print("Account Info clicked")),
            ("Log Out", lambda: print("Logged out"))
        ]

        # Tạo các nút, padding đều nhau
        for text, cmd in buttons:
            btn = Button(
                self.dropdown_window,
                text=text,
                bg="#FFFFFF",
                fg="#5A3372",
                font=("Crimson Pro", 10, "bold"),
                anchor="w",
                relief="flat",
                command=cmd,
                height=1,
                width=self.width // 6  # Further increased width to ensure full text display
            )
            btn.pack(fill="x", pady=5, padx=15)

        # Tự động đóng khi mất focus
        self.dropdown_window.bind("<FocusOut>", lambda e: self.close())
        self.dropdown_window.focus_force()

    def close(self):
        if self.dropdown_window and self.dropdown_window.winfo_exists():
            self.dropdown_window.destroy()
            self.dropdown_window = None