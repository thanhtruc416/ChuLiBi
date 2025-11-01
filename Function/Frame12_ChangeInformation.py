import tkinter as tk
from tkinter import messagebox
import re
from Frame.Frame12.ui_Frame12 import Frame12  # import UI layout
from Function.user_repository import update_user_info


class Frame12_ChangeInformation(Frame12):
    """Frame Change Information — hiển thị thông tin tài khoản và cho phép đổi mật khẩu."""

    def __init__(self, parent, controller=None):
        super().__init__(parent, controller)
        self.controller = controller
        self.user_info = {}

        # Chỉnh lại style readonly (Username, Gmail)
        self._make_readonly(self.entry_Username)
        self._make_readonly(self.entry_Gmail)

        # Gắn command cho nút Save
        self.button_save.config(command=self._on_save_clicked)

    # -----------------------------
    # Utility: set readonly style (đẹp hơn, không nền trắng)
    # -----------------------------
    def _make_readonly(self, entry: tk.Entry):
        """Giữ nguyên style UI nhưng chặn sửa nội dung."""
        current_fg = entry.cget("fg")
        entry.configure(fg="#6f6f6f", state="normal")  # Giữ style, chỉ đổi màu chữ

        # Ngăn nhập liệu
        def block_all(event): return "break"
        entry.bind("<Key>", block_all)
        entry.bind("<Button-1>", lambda e: "break")
        entry.bind("<FocusIn>", lambda e: self.focus())
        entry.bind("<FocusOut>", lambda e: entry.configure(fg="#6f6f6f"))

    # -----------------------------
    # Khi frame hiển thị
    # -----------------------------
    def on_show(self, user_data=None):
        """Hiển thị thông tin user khi frame được bật."""
        if not user_data and self.controller:
            user_data = self.controller.get_current_user()

        if user_data:
            self.user_info = user_data

        info = self.user_info or {}

        def set_entry(entry, value, readonly=False):
            entry.configure(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, value)
            if readonly:
                self._make_readonly(entry)

        # --- Gán dữ liệu ---
        set_entry(self.entry_FullName, info.get("full_name", info.get("FullName", "")))
        set_entry(self.entry_Business_Name, info.get("business_name", info.get("BusinessName", "")))
        set_entry(self.entry_Your_Role, info.get("role", info.get("YourRole", "")))

        # Clear ô Password (mỗi lần mở lại frame)
        self.entry_Password.configure(state="normal")
        self.entry_Password.delete(0, tk.END)
        self.entry_Password.focus_set()

        print(f"[Frame12] Hiển thị thông tin user: {info.get('username', '')}")

    # -----------------------------
    # Khi nhấn Save → ghi xuống DB
    # -----------------------------
    def _on_save_clicked(self):
        """Khi nhấn nút Save → cập nhật xuống MySQL"""
        full_name = self.entry_FullName.get().strip()
        business_name = self.entry_Business_Name.get().strip()
        role = self.entry_Your_Role.get().strip()
        password = self.entry_Password.get().strip()



        if not full_name or not business_name or not role:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng điền đầy đủ các thông tin cần thiết.")
            return

        try:
            update_user_info( full_name, business_name, role,  password if password else None)
            messagebox.showinfo("Thành công", "Thông tin người dùng đã được lưu thành công!")
            self.entry_Password.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu thông tin:\n{e}")
