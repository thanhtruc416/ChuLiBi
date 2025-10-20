# test_qmess_calling.py
"""
Manual test runner for Qmess_calling.py

Run:
    python test_qmess_calling.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

try:
    from Qmess_calling import Qmess
except Exception as e:
    raise SystemExit("Cannot import Qmess_calling.py. Put this file in the same folder.\n"
                     f"Original error: {e}")


POPUPS = [
    ("01 IMPORT_SUCCESS", "IMPORT_SUCCESS"),
    ("02 SAVE_DONE", "SAVE_DONE"),
    ("03 IMPORT_BAD_FORMAT", "IMPORT_BAD_FORMAT"),
    ("04 LOGIN_INVALID", "LOGIN_INVALID"),
    ("05 LOGIN_MISSING", "LOGIN_MISSING"),
    ("06 OTP_SENT", "OTP_SENT"),
    ("07 LOGIN_SUCCESS", "LOGIN_SUCCESS"),
    ("08 IMPORT_MISSING_REQUIRED", "IMPORT_MISSING_REQUIRED"),
    ("09 RESET_PASSWORD_SUCCESS", "RESET_PASSWORD_SUCCESS"),
    ("10 PROFILE_MISSING_REQUIRED", "PROFILE_MISSING_REQUIRED"),
    ("11 CANNOT_CONNECT_SERVER", "CANNOT_CONNECT_SERVER"),
    ("12 EMAIL_NOT_FOUND", "EMAIL_NOT_FOUND"),
    ("13 PASSWORD_MISMATCH", "PASSWORD_MISMATCH"),
    ("14 REGISTER_SUCCESS", "REGISTER_SUCCESS"),
    ("15 WRONG_OTP", "WRONG_OTP"),
]


class TestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Qmess Calling - Test Runner")
        self.geometry("740x560")
        self.configure(bg="#faf7ff")

        header = tk.Label(self, text="Qmess_calling – Manual Test",
                          font=("Segoe UI", 16, "bold"), bg="#faf7ff", fg="#4a2f66")
        header.pack(pady=(14, 6))

        sel_frame = tk.Frame(self, bg="#faf7ff")
        sel_frame.pack(pady=(0, 10))

        tk.Label(sel_frame, text="Choose a popup:", bg="#faf7ff").grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.combo = ttk.Combobox(sel_frame, state="readonly", width=35,
                                  values=[name for name, code in POPUPS])
        self.combo.grid(row=0, column=1, padx=(0, 8))
        self.combo.current(0)

        ttk.Button(sel_frame, text="Show", command=self.show_selected).grid(row=0, column=2, padx=(4, 0))

        grid = tk.LabelFrame(self, text="All Popups", bg="#faf7ff", fg="#4a2f66")
        grid.pack(fill="both", expand=True, padx=14, pady=8)
        self._build_grid(grid)

        bottom = tk.Frame(self, bg="#faf7ff")
        bottom.pack(fill="x", padx=14, pady=(0, 12))

        ttk.Button(bottom, text="Show All (sequential)",
                   command=self.show_all_seq).pack(side="left")

        ttk.Button(bottom, text="About",
                   command=lambda: messagebox.showinfo("About",
                       "This window helps manually trigger all Qmess popups.\n"
                       "Ensure Qmess_calling.py is in the same folder.\n"
                       "Place PNGs under ./assets or ./ui_popup_XX/assets etc."))\
                   .pack(side="right")

    def _build_grid(self, parent: tk.Frame):
        cols = 3
        for idx, (label, code) in enumerate(POPUPS):
            r = idx // cols
            c = idx % cols
            ttk.Button(parent, text=label, width=28,
                       command=lambda code=code: self._show(code))\
                .grid(row=r, column=c, padx=8, pady=8, sticky="ew")
        for c in range(cols):
            parent.grid_columnconfigure(c, weight=1)

    def _show(self, code: str):
        def after_ok():
            print(f"[OK] {code}")
        Qmess.show(code, parent=self, on_ok=after_ok)

    def show_selected(self):
        sel = self.combo.get()
        for name, code in POPUPS:
            if name == sel:
                self._show(code)
                return

    def show_all_seq(self):
        delay = 300  # ms
        for i, (_, code) in enumerate(POPUPS):
            self.after(i * delay, lambda c=code: self._show(c))


if __name__ == "__main__":
    app = TestApp()
    app.mainloop()
