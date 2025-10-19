# popup_manager.py
from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import Toplevel, Canvas, PhotoImage, Button

# ---- Core builder (giống mẫu tr_Qmess_thongtin: tạo Toplevel, canh giữa, vẽ Canvas) ----
def _build_popup(parent: tk.Misc | None,
                 assets_dir: Path,
                 title: str,
                 subtitle: str = "",
                 image_filename: str = "image_success.png",
                 ok_text: str | None = None,
                 on_ok=None) -> Toplevel:
    win = Toplevel(parent)
    win.configure(bg="#FFFFFF")
    win.resizable(False, False)
    # Kích thước mặc định của các ui_popup_* (480x329)
    W, H = 480, 329
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = (sw // 2) - (W // 2)
    y = (sh // 2) - (H // 2)
    win.geometry(f"{W}x{H}+{x}+{y}")

    canvas = Canvas(win, bg="#FFFFFF", width=W, height=H, bd=0,
                    highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)

    # Giữ reference ảnh trên window để tránh GC
    img = PhotoImage(file=assets_dir / image_filename)
    btn_img = PhotoImage(file=assets_dir / "button_okay.png")
    setattr(win, "_pm_image", img)
    setattr(win, "_pm_btn", btn_img)

    # Ảnh minh hoạ
    canvas.create_image(W // 2, 92, image=img)

    # Title & subtitle (tọa độ bám theo các ui_popup_*)
    canvas.create_text(40, 159, anchor="nw", text=title,
                       fill="#706093", font=("Young Serif", 24))
    if subtitle:
        canvas.create_text(80 if len(subtitle) < 32 else 95, 204 if "username" in subtitle else 206,
                           anchor="nw", text=subtitle, fill="#B992B9",
                           font=("Crimson Pro Regular", 17))

    # Nút OK
    def _ok():
        try:
            if callable(on_ok):
                on_ok()
        finally:
            win.destroy()

    Button(win, image=btn_img, borderwidth=0, highlightthickness=0,
           command=_ok, relief="flat").place(x=95, y=244, width=289, height=61)

    # Đóng khi mất focus (giống dropdown) – tuỳ chọn
    win.bind("<FocusOut>", lambda e: win.destroy())
    win.after(10, win.deiconify)
    return win


# ---- Bảng mapping: mã_popup -> (thư mục ui_popup_XX, title, subtitle, ảnh) ----
# Gắn đúng tên thư mục ui_popup_* của bạn nếu khác.
POPUPS: dict[str, dict] = {
    # ĐƯỜNG DẪN: Path(__file__).parent / "ui_popup_XX" / "assets"
    # 01
    "import_success": {
        "assets": "ui_popup_01",  # "Data imported successfully" (image_success)
        "title": "Data imported successfully",
        "subtitle": "Start your journey now!",
        "image": "image_success.png",
    },
    # 02
    "changes_saved": {
        "assets": "ui_popup_02",
        "title": "Done",
        "subtitle": "All changes have been saved!",
        "image": "image_success.png",
    },
    # 03
    "invalid_file": {
        "assets": "ui_popup_03",
        "title": "Invalid file format",
        "subtitle": "Please import again!",
        "image": "image_cancel.png",
    },
    # 04
    "invalid_credentials": {
        "assets": "ui_popup_04",
        "title": "Invalid username or password",
        "subtitle": "Please enter correct username or password!",
        "image": "image_cancel.png",
    },
    # 05
    "missing_login_fields": {
        "assets": "ui_popup_05",
        "title": "Missing data",
        "subtitle": "Please enter both username and password!",
        "image": "image_alert.png",
    },
    # 06
    "otp_sent": {
        "assets": "ui_popup_06",
        "title": "Sent!",
        "subtitle": "OTP has been sent successfully",
        "image": "image_success.png",
    },
    # 07
    "login_success": {
        "assets": "ui_popup_07",
        "title": "Signed in successfully!",
        "subtitle": "Welcome!",
        "image": "image_success.png",
    },
    # 08
    "import_missing_required": {
        "assets": "ui_popup_08",
        "title": "Imported data is missing required",
        "subtitle": "Please import again!",
        "image": "image_alert.png",
    },
    # 09
    "reset_ok": {
        "assets": "ui_popup_09",
        "title": "Password reset successfully",
        "subtitle": "You can login again now!",
        "image": "image_success.png",
    },
    # 10
    "missing_generic": {
        "assets": "ui_popup_10",
        "title": "Missing data",
        "subtitle": "Please fill in all required field!",
        "image": "image_alert.png",
    },
    # 11
    "server_unreachable": {
        "assets": "ui_popup_11",
        "title": "Cannot connect to the server",
        "subtitle": "Please try again later!",
        "image": "image_alert.png",
    },
    # 12
    "email_not_found": {
        "assets": "ui_popup_12",
        "title": "Email not found or incorrect",
        "subtitle": "Please fill correct email!",
        "image": "image_profile.png",
    },
    # 13
    "password_mismatch": {
        "assets": "ui_popup_13",
        "title": "New password and confirmation do not match!",
        "subtitle": "",
        "image": "image_alert.png",
    },
    # 14
    "registration_ok": {
        "assets": "ui_popup_14",
        "title": "Registration successfully!",
        "subtitle": "Your profile has been completed",
        "image": "image_success.png",
    },
    # 15
    "wrong_otp": {
        "assets": "ui_popup_15",
        "title": "Wrong OTP!",
        "subtitle": "Please fill again",
        "image": "image_cancel.png",
    },
}


def show_popup(kind: str,
               parent: tk.Misc | None = None,
               on_ok=None) -> Toplevel:
    """
    Mở pop-up theo mã 'kind'.
    - parent: Window/Frame cha (khuyên truyền vào để pop-up nổi trên app)
    - on_ok: callback sau khi bấm OK (tùy chọn)
    Trả về: Toplevel của pop-up.
    """
    cfg = POPUPS.get(kind)
    if not cfg:
        # fallback: hiển thị generic success nếu mã không tồn tại
        assets_dir = Path(__file__).parent / "ui_popup_01" / "assets"
        return _build_popup(parent, assets_dir,
                            title="Done",
                            subtitle="Operation completed",
                            image_filename="image_success.png",
                            on_ok=on_ok)

    assets_dir = Path(__file__).parent / cfg["assets"] / "assets"
    return _build_popup(
        parent=parent,
        assets_dir=assets_dir,
        title=cfg["title"],
        subtitle=cfg.get("subtitle", ""),
        image_filename=cfg.get("image", "image_success.png"),
        on_ok=on_ok,
    )
