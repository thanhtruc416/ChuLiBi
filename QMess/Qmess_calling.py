# Qmess_calling.py
"""
Unified popup caller for Tkinter app.

- Provides Qmess.show(<CODE>) and helper methods (e.g., Qmess.login_invalid())
- Auto-loads PNG assets (icon, gradient button) from several possible layouts.
- Falls back to text-only layout if assets are missing.

Directory strategies (checked in order) for each popup N (1..15):
  1) ./ui_popup_XX/assets/
  2) ./assets/ui_popup_XX/
  3) ./assets/popup_XX/
  4) ./assets/            # one shared folder for all popups

If using a shared folder (4), prefer files named image_N*.png / button_N*.png.
Otherwise any image_*.png / button_*.png will be used.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Iterable, Tuple
from pathlib import Path
import tkinter as tk
from tkinter import Toplevel, Canvas, Button, PhotoImage

ROOT_DIR = Path(__file__).resolve().parent  # where this file lives


# ------------------------ Asset discovery ------------------------
def _find_assets_for(n: int) -> Tuple[Optional[Path], int]:
    """
    Return (assets_dir, index) for popup n.

    assets_dir may be None (text-only). `index` is forwarded so loaders can try
    image_{index}.png / button_{index}.png when using a shared assets folder.
    """
    candidates = [
        ROOT_DIR / f"ui_popup_{n:02d}" / "assets",
        ROOT_DIR / "assets" / f"ui_popup_{n:02d}",
        ROOT_DIR / "assets" / f"popup_{n:02d}",
        ROOT_DIR / "assets",  # shared
    ]

    for p in candidates:
        if p.exists():
            # If there is at least icon or button, accept
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
    title: str
    subtitle: str
    width: int = 480
    height: int = 329
    ok_text: str = "Okay"
    assets_dir: Optional[Path] = None
    asset_index: int = 0  # popup number for naming in shared assets


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

        # Keep references to images
        self._img_icon: Optional[PhotoImage] = None
        self._img_btn: Optional[PhotoImage] = None

        self._build_ui()

    # -------------- geometry helpers --------------
    def _center(self):
        self.top.update_idletasks()
        sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        x = (sw - self.spec.width) // 2
        y = (sh - self.spec.height) // 2
        self.top.geometry(f"{self.spec.width}x{self.spec.height}+{x}+{y}")

    # -------------- asset loading --------------
    def _load_icon(self) -> Optional[PhotoImage]:
        d = self.spec.assets_dir
        if not d:
            return None
        n = self.spec.asset_index
        # Prefer exact-numbered names when shared folder is used
        preferred = [f"image_{n}.png", f"image_{n}_*.png"]
        generic = ["image_1.png", "image_*.png", "icon_*.png", "icon.png"]
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
        generic = ["button_1.png", "button_*.png"]
        path = _first_match(d, preferred + generic)
        if path:
            try:
                return PhotoImage(file=str(path))
            except Exception:
                return None
        return None

    # -------------- UI --------------
    def _build_ui(self):
        w, h = self.spec.width, self.spec.height
        c = Canvas(self.top, bg="#FFFFFF", height=h, width=w,
                   bd=0, highlightthickness=0, relief="ridge")
        c.place(x=0, y=0)

        # Try assets
        self._img_icon = self._load_icon()
        if self._img_icon:
            # center icon horizontally near top
            c.create_image(w / 2, 90, image=self._img_icon)

        # Title + Subtitle (colors per design)
        # Use pixel-like sizes close to Tkinter Designer look
        c.create_text(40, 150, anchor="nw",
                      text=self.spec.title, fill="#706093",
                      font=("Young Serif", 24))
        c.create_text(40, 190, anchor="nw",
                      text=self.spec.subtitle, fill="#B992B9",
                      font=("Crimson Pro", 14))

        # Button
        self._img_btn = self._load_button_bg()
        if self._img_btn:
            btn = Button(self.top, image=self._img_btn, text=self.spec.ok_text,
                         compound="center", fg="#FFFFFF",
                         font=("Crimson Pro", 14, "bold"), borderwidth=0,
                         highlightthickness=0, relief="flat",
                         command=self._ok_and_close)
            # try to mimic exported sizes (around 289x61)
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
        self.top.grab_set()

    def _ok_and_close(self):
        try:
            if callable(self.on_ok):
                self.on_ok()
        finally:
            self.top.destroy()


# ------------------------ Public API ------------------------
class Qmess:
    """
    Usage:
        from Qmess_calling import Qmess
        Qmess.show("LOGIN_INVALID", parent=self)
        # or helpers:
        Qmess.login_invalid(self)
    """

    # Registry filled below
    _REGISTRY: Dict[str, PopupSpec] = {}

    @staticmethod
    def _make(title: str, subtitle: str, n: int) -> PopupSpec:
        d, idx = _find_assets_for(n)
        return PopupSpec(title=title, subtitle=subtitle,
                         assets_dir=d, asset_index=idx)

    # Build registry
    _REGISTRY = {
        "IMPORT_SUCCESS":          _make.__func__("Data imported successfully", "Start your journey now!", 1),
        "SAVE_DONE":               _make.__func__("Done", "All changes have been saved!", 2),
        "IMPORT_BAD_FORMAT":       _make.__func__("Invalid file format", "Please import again!", 3),
        "LOGIN_INVALID":           _make.__func__("Invalid username or password", "Please enter correct username or password!", 4),
        "LOGIN_MISSING":           _make.__func__("Missing data", "Please enter both username and password!", 5),
        "OTP_SENT":                _make.__func__("Sent!", "OTP has been sent successfully", 6),
        "LOGIN_SUCCESS":           _make.__func__("Signed in successfully!", "Welcome!", 7),
        "IMPORT_MISSING_REQUIRED": _make.__func__("Imported data is missing required", "Please import again!", 8),
        "RESET_PASSWORD_SUCCESS":  _make.__func__("Password reset successfully", "You can login again now!", 9),
        "PROFILE_MISSING_REQUIRED":_make.__func__("Missing data", "Please fill in all required field!", 10),
        "CANNOT_CONNECT_SERVER":   _make.__func__("Cannot connect to the server", "Please try again later!", 11),
        "EMAIL_NOT_FOUND":         _make.__func__("Email not found or incorrect", "Please fill correct email!", 12),
        "PASSWORD_MISMATCH":       _make.__func__("New password and confirmation do not match!", "", 13),
        "REGISTER_SUCCESS":        _make.__func__("Registration successfully!", "Your profile has been completed", 14),
        "WRONG_OTP":               _make.__func__("Wrong OTP!", "Please fill again", 15),
    }

    @staticmethod
    def show(code: str, parent: Optional[tk.Misc] = None,
             on_ok: Optional[Callable[[], None]] = None) -> Toplevel:
        spec = Qmess._REGISTRY.get((code or "").upper())
        if not spec:
            spec = PopupSpec(title=code or "Message", subtitle="")
        return PopupWindow(parent, spec, on_ok=on_ok).top

    # Convenience helpers
    import_success              = staticmethod(lambda parent=None, on_ok=None: Qmess.show("IMPORT_SUCCESS", parent, on_ok))
    save_done                   = staticmethod(lambda parent=None, on_ok=None: Qmess.show("SAVE_DONE", parent, on_ok))
    import_bad_format           = staticmethod(lambda parent=None, on_ok=None: Qmess.show("IMPORT_BAD_FORMAT", parent, on_ok))
    login_invalid               = staticmethod(lambda parent=None, on_ok=None: Qmess.show("LOGIN_INVALID", parent, on_ok))
    login_missing               = staticmethod(lambda parent=None, on_ok=None: Qmess.show("LOGIN_MISSING", parent, on_ok))
    otp_sent                    = staticmethod(lambda parent=None, on_ok=None: Qmess.show("OTP_SENT", parent, on_ok))
    login_success               = staticmethod(lambda parent=None, on_ok=None: Qmess.show("LOGIN_SUCCESS", parent, on_ok))
    import_missing_required     = staticmethod(lambda parent=None, on_ok=None: Qmess.show("IMPORT_MISSING_REQUIRED", parent, on_ok))
    reset_password_success      = staticmethod(lambda parent=None, on_ok=None: Qmess.show("RESET_PASSWORD_SUCCESS", parent, on_ok))
    profile_missing_required    = staticmethod(lambda parent=None, on_ok=None: Qmess.show("PROFILE_MISSING_REQUIRED", parent, on_ok))
    cannot_connect_server       = staticmethod(lambda parent=None, on_ok=None: Qmess.show("CANNOT_CONNECT_SERVER", parent, on_ok))
    email_not_found             = staticmethod(lambda parent=None, on_ok=None: Qmess.show("EMAIL_NOT_FOUND", parent, on_ok))
    password_mismatch           = staticmethod(lambda parent=None, on_ok=None: Qmess.show("PASSWORD_MISMATCH", parent, on_ok))
    register_success            = staticmethod(lambda parent=None, on_ok=None: Qmess.show("REGISTER_SUCCESS", parent, on_ok))
    wrong_otp                   = staticmethod(lambda parent=None, on_ok=None: Qmess.show("WRONG_OTP", parent, on_ok))
