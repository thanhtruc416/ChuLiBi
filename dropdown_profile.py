def show_dropdown():
    # Nếu dropdown đã mở thì đóng trước
    for widget in window.winfo_children():
        if isinstance(widget, Toplevel):
            widget.destroy()

    # === Tạo popup dropdown ===
    dropdown = Toplevel(window)
    dropdown.overrideredirect(True)  # Bỏ viền cửa sổ
    dropdown.configure(bg="#FFFFFF")
    dropdown.attributes('-topmost', True)
    # Lấy vị trí button_2
    x = button_Profile.winfo_rootx()
    y = button_Profile.winfo_rooty() + button_Profile.winfo_height()

    dropdown.geometry(f"150x80+{x-100}+{y+5}")  # Điều chỉnh vị trí

    dropdown.update_idletasks()
    dropdown.lift()

    frame = Frame(dropdown, bg="#FFFFFF", highlightthickness=0)
    frame.pack(fill="both", expand=True, padx=5, pady=5)
    # Nút Account Info
    account_btn = Button(
        frame,
        text="Account Info",
        bg="#FFFFFF",
        fg="#5A3372",
        font=("Crimson Pro", 12, "bold"),
        anchor="w",
        relief="flat",
        command=lambda: print("Account Info clicked")
    )
    account_btn.pack(fill="x", pady=(5, 0), padx=10)

    # Nút Log out
    logout_btn = Button(
        frame,
        text="Log Out",
        bg="#FFFFFF",
        fg="#5A3372",
        font=("Crimson Pro", 12, "bold"),
        anchor="w",
        relief="flat",
        command=lambda: print("Logged out")
    )
    logout_btn.pack(fill="x", pady=(5, 5), padx=10)

    # Đóng dropdown khi click ra ngoài
    window.bind("<Button-1>", lambda e: close_dropdown(e, dropdown))

def close_dropdown(event, popup):
    # Nếu click không nằm trong popup
    if popup and not (popup.winfo_x() <= event.x_root <= popup.winfo_x() + popup.winfo_width() and
                      popup.winfo_y() <= event.y_root <= popup.winfo_y() + popup.winfo_height()):
        popup.destroy()
# Gán sự kiện click vào icon người dùng
button_Profile.configure(command=show_dropdown)
