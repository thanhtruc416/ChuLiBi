"""
Qmess_calling.py

Mục tiêu: Gọi được 29 QMess (01..29) bạn đã làm sẵn (thư mục `ui_popup_XX/assets`).

Cách hoạt động:
- Xây dựng Toplevel (không dùng `Tk()` độc lập) để nhúng vào app chính.
- Tự động tìm ảnh trong các đường dẫn quen thuộc của mỗi popup n (1..29):
    1) ./ui_popup_XX/assets/
    2) ./assets/ui_popup_XX/
    3) ./assets/popup_XX/
    4) ./assets/ (dùng chung) – ưu tiên đặt tên image_n*.png / button_n*.png
- Nếu thiếu ảnh, sẽ fallback sang layout chữ đơn giản.

API chính:
- Qmess.show(code_or_index, parent=None, on_ok=None, title=None, subtitle=None)
  - code_or_index: có thể là chuỗi mã hoặc số (1..29)
  - title/subtitle: cho phép override nội dung chữ; nếu không truyền, sẽ dùng gợi ý sẵn hoặc để trống
- Helper: Qmess.popup_01(...) .. Qmess.popup_29(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Iterable, Tuple, Union
from pathlib import Path
import tkinter as tk
from tkinter import Toplevel, Canvas, Button, PhotoImage

ROOT_DIR = Path(__file__).resolve().parent

# Phân loại icon mong muốn theo chỉ số popup (dựa vào kịch bản)
_SUCCESS_INDEXES = {3, 5, 7, 14, 16, 22, 28}
_WARNING_INDEXES = {15, 18, 19, 20, 21, 23, 24, 25, 29}
_ERROR_INDEXES = {
    1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 17, 26, 27
}


# ------------------------ Asset discovery ------------------------
def _find_assets_for(n: int) -> Tuple[Optional[Path], int]:
    candidates = [
        ROOT_DIR / f"ui_popup_{n:02d}" / "assets",
        ROOT_DIR / "assets" / f"ui_popup_{n:02d}",
        ROOT_DIR / "assets" / f"popup_{n:02d}",
        ROOT_DIR / "assets",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            if any(p.glob("image_*.png")) or any(p.glob("button_*.png")):
                return p, n
    return None, n


def _first_match(folder: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pat in patterns:
        files = sorted(folder.glob(pat))
        if files:
            return files[0]
    return None


# ------------------------ Core window ------------------------
@dataclass
class PopupSpec:
    title: str = ""
    subtitle: str = ""
    width: int = 480
    height: int = 329
    ok_text: str = "Okay"
    assets_dir: Optional[Path] = None
    asset_index: int = 0


class PopupWindow:
    def __init__(self, parent: Optional[tk.Misc], spec: PopupSpec,
                 on_ok: Optional[Callable[[], None]] = None):
        self.parent = parent
        self.spec = spec
        self.on_ok = on_ok

        self.top = Toplevel(parent)
        self.top.withdraw()
        self.top.configure(bg="#FFFFFF")
        self.top.resizable(False, False)

        self._img_icon: Optional[PhotoImage] = None
        self._img_btn: Optional[PhotoImage] = None

        self._build_ui()

    def _center(self):
        self.top.update_idletasks()
        sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        x = (sw - self.spec.width) // 2
        y = (sh - self.spec.height) // 2
        self.top.geometry(f"{self.spec.width}x{self.spec.height}+{x}+{y}")

    def _load_icon(self) -> Optional[PhotoImage]:
        d = self.spec.assets_dir
        if not d:
            return None
        n = self.spec.asset_index
        preferred = [f"image_{n}.png", f"image_{n}_*.png"]

        # Ưu tiên icon theo ngữ nghĩa nếu không có ảnh theo số
        if n in _SUCCESS_INDEXES:
            semantic_first = ["image_success.png", "icon_success.png"]
        elif n in _WARNING_INDEXES:
            semantic_first = ["image_warning.png", "icon_warning.png", "image_alert.png"]
        else:  # lỗi
            semantic_first = ["image_cancel.png", "icon_error.png", "image_alert.png"]

        generic = [
            *semantic_first,
            "image_*.png", "icon_*.png", "icon.png",
            # fallback tên cụ thể cũ
            "image_cancel.png",
        ]
        path = _first_match(d, preferred + generic)
        if path:
            try:
                return PhotoImage(file=str(path))
            except Exception:
                return None
        return None

    def _load_button_bg(self) -> Optional[PhotoImage]:
        d = self.spec.assets_dir
        if not d:
            return None
        n = self.spec.asset_index
        preferred = [f"button_{n}.png", f"button_{n}_*.png"]
        generic = ["button_okay.png", "button_Save.png", "button_*.png"]
        path = _first_match(d, preferred + generic)
        if path:
            try:
                return PhotoImage(file=str(path))
            except Exception:
                return None
        return None

    def _build_ui(self):
        w, h = self.spec.width, self.spec.height
        c = Canvas(self.top, bg="#FFFFFF", height=h, width=w,
                   bd=0, highlightthickness=0, relief="ridge")
        c.place(x=0, y=0)

        self._img_icon = self._load_icon()
        if self._img_icon:
            c.create_image(w / 2, 90, image=self._img_icon)

        c.create_text(
            w / 2, 150,
            anchor="n",
            text=self.spec.title or "",
            fill="#706093",
            font=("Young Serif", 18),
            width=w - 40,
            justify="center"
        )
        c.create_text(
            w / 2, 195,
            anchor="n",
            text=self.spec.subtitle or "",
            fill="#B992B9",
            font=("Crimson Pro", 14),
            width=w - 80,
            justify="center"
        )

        self._img_btn = self._load_button_bg()
        if self._img_btn:
            btn = Button(self.top, image=self._img_btn,
                         compound="center", fg="#FFFFFF",
                         font=("Crimson Pro", 14, "bold"), borderwidth=0,
                         highlightthickness=0, relief="flat",
                         command=self._ok_and_close)
            bw, bh = 289, 61
            x = (w - bw) // 2
            y = h - 91
            btn.place(x=x, y=y, width=bw, height=bh)
        else:
            btn = Button(self.top, text=self.spec.ok_text,
                         bg="#FFFFFF", fg="#5A3372",
                         font=("Crimson Pro", 12, "bold"),
                         relief="flat", command=self._ok_and_close)
            btn.place(x=w // 2 - 70, y=h - 85, width=140, height=44)

        self._center()
        self.top.deiconify()
        self.top.transient(self.parent)
        try:
            self.top.lift()
            self.top.attributes('-topmost', True)
            # hạ cờ topmost về False sau khi đã nổi lên trên cùng
            self.top.after(200, lambda: self.top.attributes('-topmost', False))
        except Exception:
            pass
        self.top.grab_set()
        try:
            self.top.focus_force()
        except Exception:
            pass

    def _ok_and_close(self):
        try:
            if callable(self.on_ok):
                self.on_ok()
        finally:
            self.top.destroy()


# ------------------------ Public API ------------------------
class Qmess:
    """Bộ gọi QMess 1..29 dựa theo assets đã export từ Tkinter Designer.

    Cách dùng nhanh:
        Qmess.show(4, parent=self)                      # gọi popup 04
        Qmess.show("LOGIN_INVALID", parent=self)        # nếu có mapping sẵn
        Qmess.popup_17(parent=self, title="Warning")    # helper theo số, override title
    """

    _REGISTRY: Dict[str, PopupSpec] = {}

    # Một số mã tên quen dùng (có thể chỉnh theo nhu cầu). Map → index (1..29)
    _NAMED_TO_INDEX: Dict[str, int] = {
        # Frame 01 – Login / Logout
        "LOGIN_ERROR": 1,
        "LOGIN_FAILED": 2,
        "LOGIN_SUCCESSFUL": 3,

        # Frame 02 – Registration
        "REGISTRATION_FAILED_MISSING_ENTRY": 4,
        "REGISTRATION_SUCCESSFUL": 5,
        "INVALID_EMAIL": 6,
        "OTP_SENT": 7,
        "TERMS_AND_CONDITIONS": 8,
        "REGISTRATION_FAILED_USERNAME_EXISTS": 9,
        "REGISTRATION_FAILED_PASSWORD_REQUIREMENT": 10,
        "REGISTRATION_FAILED_CONFIRM_MISMATCH": 11,
        "REGISTRATION_FAILED_OTP_WRONG": 12,

        # Frame 03 – Profile
        "INCOMPLETE_PROFILE": 13,
        "PROFILE_COMPLETED": 14,

        # Frame 04 – Forgot Password / OTP
        "WARNING_EMAIL_REQUIRED": 15,
        "SUCCESS_OTP_SENT": 16,
        "EMAIL_NOT_FOUND": 17,

        # Frame 05 – Reset Password
        "WARNING_ENTER_OTP": 18,
        "WARNING_ENTER_NEW_PASSWORD": 19,
        "WARNING_PASSWORDS_DO_NOT_MATCH": 20,
        "WARNING_PASSWORD_TOO_SHORT": 21,
        "SUCCESS_PASSWORD_RESET": 22,

        # Frame 09/10/… – Data / CustomerID
        "WARNING_CUSTOMERID_NOT_EXIST": 23,

        # Frame 11 – Data Management
        "WARNING_DATA_INCOMPLETE": 24,

        # Frame 12 – Password Change
        "CHANGE_FAILED_PASSWORD_REQUIREMENTS": 25,

        # Frame 13 – Data Upload / Management
        "FILE_ERROR_INVALID_FORMAT": 26,
        "DATA_MISSING_REQUIRED_FIELDS": 27,
        "UPLOAD_SUCCESS": 28,

        # Frame 14 – Delivery (theo file 29)
        "WARNING_DELIVERY_DATA_MISSING": 29,
    }

    # Gợi ý nội dung mặc định (map 1..29 theo tài liệu kịch bản)
    _DEFAULT_TEXTS: Dict[int, Tuple[str, str]] = {
        # Frame 01 – Login / Logout
        1: ("Login Error", "Please enter both username and password"),
        2: ("Login Failed", "Unknown error"),
        3: ("Login Successful", "Welcome back, {name}!"),

        # Frame 02 – Registration
        4: ("Registration Failed", "Please fill the missing entry"),
        5: ("Registration Successful", "Account created! Please complete your profile."),
        6: ("Invalid email", "Please enter a valid email address"),
        7: ("OTP Sent", "We sent a verification code to your email."),
        8: ("Terms & Conditions", "Please accept the Terms & Privacy Policy to continue"),
        9: ("Registration Failed", "Please choose another username"),
        10: ("Registration Failed", "Password must have 8 characters."),
        11: ("Registration Failed", "Confirm Password must be same as Password"),
        12: ("Registration Failed", "OTP is wrong"),

        # Frame 03 – Profile / Update Info
        13: ("Incomplete Profile", "Please fill in all fields to complete your profile"),
        14: ("Profile Completed", "Welcome, {full_name}! Your profile has been completed successfully! Please login again."),

        # Frame 04 – Forgot Password / OTP
        15: ("Warning", "Please enter your email first"),
        16: ("Success", "The OTP is valid for 10 minutes. Please continue to enter it quickly."),
        17: ("Email Not Found", "This email does not exist in our system. Please check and try again."),

        # Frame 05 – Reset Password
        18: ("Warning", "Please enter the OTP"),
        19: ("Warning", "Please enter your new password"),
        20: ("Warning", "The passwords do not match"),
        21: ("Warning", "The password must be at least 8 characters long"),
        22: ("Success", "Password reset successfully"),

        # Frame 09 – Data / Customer ID
        23: ("Warning", "CustomerID does not exist"),

        # Frame 11 – Data Management
        24: ("Warning", "Please fill all the following information below"),

        # Frame 12 – Password Change
        25: ("Change Failed", "Password must have 8 characters."),

        # Frame 13 – Data Upload / Management
        26: ("File Error", "Invalid file format"),
        27: ("Data Missing", "Please fill all required fields"),
        28: ("Upload Success", "Data has been uploaded successfully"),

        # Frame 14 – Delivery
        29: ("Error", "Something went wrong! Please try again"),
    }

    @staticmethod
    def _make_for_index(n: int,
                        title: Optional[str] = None,
                        subtitle: Optional[str] = None,
                        ok_text: str = "Okay") -> PopupSpec:
        d, idx = _find_assets_for(n)
        if title is None or subtitle is None:
            t, s = Qmess._DEFAULT_TEXTS.get(n, ("", ""))
            title = t if title is None else title
            subtitle = s if subtitle is None else subtitle
        return PopupSpec(title=title or "", subtitle=subtitle or "",
                         ok_text=ok_text, assets_dir=d, asset_index=idx)

    @staticmethod
    def show(code_or_index: Union[str, int],
             parent: Optional[tk.Misc] = None,
             on_ok: Optional[Callable[[], None]] = None,
             title: Optional[str] = None,
             subtitle: Optional[str] = None,
             ok_text: str = "Okay") -> Toplevel:
        # Ưu tiên: nếu là số 1..29 → dùng theo index
        idx: Optional[int] = None
        if isinstance(code_or_index, int):
            idx = code_or_index
        else:
            code = (code_or_index or "").strip().upper()
            if code.isdigit():
                try:
                    idx = int(code)
                except Exception:
                    idx = None
            if idx is None:
                idx = Qmess._NAMED_TO_INDEX.get(code)

        if idx is not None and 1 <= idx <= 29:
            spec = Qmess._make_for_index(idx, title=title, subtitle=subtitle, ok_text=ok_text)
        else:
            # Fallback – popup chữ trơn
            spec = PopupSpec(title=title or (str(code_or_index) if code_or_index else "Message"),
                             subtitle=subtitle or "",
                             ok_text=ok_text)
        return PopupWindow(parent, spec, on_ok=on_ok).top

    # 29 helpers theo số
    popup_01 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(1, parent, on_ok, title, subtitle, ok_text))
    popup_02 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(2, parent, on_ok, title, subtitle, ok_text))
    popup_03 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(3, parent, on_ok, title, subtitle, ok_text))
    popup_04 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(4, parent, on_ok, title, subtitle, ok_text))
    popup_05 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(5, parent, on_ok, title, subtitle, ok_text))
    popup_06 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(6, parent, on_ok, title, subtitle, ok_text))
    popup_07 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(7, parent, on_ok, title, subtitle, ok_text))
    popup_08 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(8, parent, on_ok, title, subtitle, ok_text))
    popup_09 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(9, parent, on_ok, title, subtitle, ok_text))
    popup_10 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(10, parent, on_ok, title, subtitle, ok_text))
    popup_11 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(11, parent, on_ok, title, subtitle, ok_text))
    popup_12 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(12, parent, on_ok, title, subtitle, ok_text))
    popup_13 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(13, parent, on_ok, title, subtitle, ok_text))
    popup_14 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(14, parent, on_ok, title, subtitle, ok_text))
    popup_15 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(15, parent, on_ok, title, subtitle, ok_text))
    popup_16 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(16, parent, on_ok, title, subtitle, ok_text))
    popup_17 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(17, parent, on_ok, title, subtitle, ok_text))
    popup_18 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(18, parent, on_ok, title, subtitle, ok_text))
    popup_19 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(19, parent, on_ok, title, subtitle, ok_text))
    popup_20 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(20, parent, on_ok, title, subtitle, ok_text))
    popup_21 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(21, parent, on_ok, title, subtitle, ok_text))
    popup_22 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(22, parent, on_ok, title, subtitle, ok_text))
    popup_23 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(23, parent, on_ok, title, subtitle, ok_text))
    popup_24 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(24, parent, on_ok, title, subtitle, ok_text))
    popup_25 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(25, parent, on_ok, title, subtitle, ok_text))
    popup_26 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(26, parent, on_ok, title, subtitle, ok_text))
    popup_27 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(27, parent, on_ok, title, subtitle, ok_text))
    popup_28 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(28, parent, on_ok, title, subtitle, ok_text))
    popup_29 = staticmethod(lambda parent=None, on_ok=None, title=None, subtitle=None, ok_text="Okay": Qmess.show(29, parent, on_ok, title, subtitle, ok_text))
