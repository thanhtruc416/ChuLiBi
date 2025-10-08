from tkinter import Tk, Button
from dropdown_profile import DropdownMenu  # import module bạn vừa tạo

# --- Tạo window chính ---
root = Tk()
root.title("Test Dropdown")
root.geometry("400x300")

# --- Tạo nút mở dropdown ---
btn_open = Button(root, text="Open Dropdown")
btn_open.pack(pady=50)

# --- Tạo dropdown ---
dropdown = DropdownMenu(root, x_offset=150, y_offset=50)

# Gắn hàm show dropdown cho nút
btn_open.config(command=dropdown.show)

# --- Chạy ứng dụng ---
root.mainloop()
