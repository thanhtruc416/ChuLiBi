# ui_Frame08.py — FULL sidebar buttons + content scroll bên phải

import os

# Suppress multiprocessing warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['LOKY_MAX_CPU_COUNT'] = '2'  # Limit parallel workers

from pathlib import Path
from tkinter import Frame, Canvas, Button, PhotoImage, Scrollbar
import tkinter as tk
import sys

# Import build_content from the same directory
try:
    from .ui_content_Frame08 import build_content
except ImportError:
    from ui_content_Frame08 import build_content

# Import dropdown if needed (similar to Frame06)
try:
    from Function.dropdown_profile import DropdownMenu
except ImportError:
    DropdownMenu = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame08")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Frame08(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._imgs = {}
        self.configure(bg="#FFFFFF")

        # Constants
        APP_W, APP_H = 1440, 1024
        RIGHT_X = 340
        RIGHT_Y = 65
        BAR_W = 20

        # Main Canvas
        canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=1088,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        canvas.place(x=0, y=0)

        # Helper function to load and cache images
        def _safe_img(name):
            try:
                if name not in self._imgs:
                    self._imgs[name] = PhotoImage(file=relative_to_assets(name))
                return self._imgs[name]
            except Exception as e:
                print(f"[Frame08] Could not load image {name}: {e}")
                return None

        def _add_img(img, x, y):
            if img is not None:
                canvas.create_image(x, y, image=img)

        # --- Background images ---
        image_image_1 = _safe_img("image_1.png")
        _add_img(image_image_1, 723.0, 544.0)

        image_image_2 = _safe_img("image_sidebar_bg.png")
        _add_img(image_image_2, 168.0, 512.0)

        # Logo text
        canvas.create_text(
            92.0, 942.0, anchor="nw",
            text="ChuLiBi", fill="#FDE5F4",
            font=("Rubik Burned Regular", 35 * -1)
        )

        # Logo icon
        image_image_3 = _safe_img("image_3.png")
        _add_img(image_image_3, 169.0, 103.0)

        # --- Sidebar buttons ---
        self._make_button("button_dashboard.png", cmd=lambda: self.controller.show_frame("Frame06"), x=0.0, y=198.0,
                          w=335.0, h=97.0)
        self._make_button("button_customer.png", cmd=lambda: self.controller.show_frame("Frame07"), x=0.0, y=289.0,
                          w=335.0, h=90.0)
        self._make_button("button_churn_1.png", cmd=lambda: print("Churn"), x=0.0, y=378.0, w=335.0, h=92.0)
        self._make_button("button_delivery.png", cmd=lambda: self.controller.show_frame("Frame09"), x=0.0, y=464.0,
                          w=335.0, h=89.0)
        self._make_button("button_recommend.png", cmd=lambda: self.controller.show_frame("Frame10"), x=0.0, y=544.0,
                          w=335.0, h=98.0)
        self._make_button("button_report.png", cmd=lambda: print("Report"), x=0.0, y=634.0, w=335.0, h=96.0)

        # Decorative icon
        image_image_5 = _safe_img("image_5.png")
        _add_img(image_image_5, 14.0, 426.0)

        # --- Top bar ---
        image_image_4 = _safe_img("image_4.png")
        _add_img(image_image_4, 885.0, 31.0)

        canvas.create_text(
            348.0, 2.0, anchor="nw",
            text="   Churn", fill="#000000",
            font=("Young Serif", 40 * -1)
        )

        # Profile button with dropdown
        if DropdownMenu:
            self.button_Profile_image = _safe_img("button_Profile.png")
            self.dropdown = DropdownMenu(self)
            self.button_Profile = Button(
                self,
                image=self.button_Profile_image,
                borderwidth=0,
                highlightthickness=0,
                command=self.dropdown.show,
                relief="flat"
            )
            self.button_Profile.place(x=1371.0, y=10.0, width=46.0, height=40.0)
        else:
            self._make_button(
                "button_Profile.png",
                cmd=lambda: print("Profile"),
                x=1371.0, y=10.0, w=46.0, h=40.0
            )

        self._make_button(
            "button_Notification.png",
            cmd=lambda: print("Notification"),
            x=1306.0, y=10.0, w=46.0, h=43.0
        )

        # ================== Content + Scrollbar ==================
        # Right content frame
        right_content = Frame(self, bg="#FFFFFF")
        right_content.place(x=RIGHT_X, y=RIGHT_Y)

        # Build content canvas
        RIGHT_W = APP_W - RIGHT_X - BAR_W
        RIGHT_H = APP_H - RIGHT_Y
        canvas_content = build_content(right_content, RIGHT_W, RIGHT_H)

        # Scrollbar gutter
        right_gutter = Frame(self, bg="#FFFFFF")
        right_gutter.place(x=APP_W - BAR_W, y=RIGHT_Y, width=BAR_W, height=RIGHT_H)

        # Vertical scrollbar
        vbar = Scrollbar(
            right_gutter,
            orient="vertical",
            command=canvas_content.yview,
            width=BAR_W,
            bd=0,
            highlightthickness=0
        )
        vbar.pack(fill="y", side="left")
        canvas_content.configure(yscrollcommand=vbar.set)

        # Sync sizes function
        def _sync_sizes(_=None):
            app_w = self.winfo_width() or APP_W
            app_h = self.winfo_height() or APP_H

            w = max(100, app_w - RIGHT_X - BAR_W)
            h = max(100, app_h - RIGHT_Y)

            right_content.configure(width=w, height=h)
            canvas_content.configure(width=w, height=h)

            # Position scrollbar at the right edge
            right_gutter.place_configure(x=app_w - BAR_W, y=RIGHT_Y, width=BAR_W, height=h)

            # Update scroll region
            bbox = canvas_content.bbox("all")
            if not bbox:
                bbox = (0, 0, w, h)
            x0, y0, x1, y1 = bbox
            canvas_content.configure(scrollregion=(0, 0, max(w, x1 - x0), max(h, y1 - y0)))

        self.after(60, _sync_sizes)
        self.bind("<Configure>", _sync_sizes)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            step = -1 if delta > 0 else 1
            canvas_content.yview_scroll(step, "units")

        # Bind mouse wheel events
        canvas_content.bind_all("<MouseWheel>", _on_mousewheel)
        canvas_content.bind_all("<Button-4>", lambda e: canvas_content.yview_scroll(-1, "units"))
        canvas_content.bind_all("<Button-5>", lambda e: canvas_content.yview_scroll(1, "units"))

    def _make_button(self, filename, cmd, x, y, w, h):
        """Helper method to create buttons with images"""
        if filename not in self._imgs:
            try:
                self._imgs[filename] = PhotoImage(file=relative_to_assets(filename))
            except Exception as e:
                print(f"[Frame08] Could not load button image {filename}: {e}")
                return None

        btn = Button(
            self,
            image=self._imgs[filename],
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            command=cmd
        )
        btn.place(x=x, y=y, width=w, height=h)
        return btn


# -------------------------
# Standalone preview runner
# -------------------------
if __name__ == "__main__":
    # Ensure project imports work even when running this file directly
    try:
        from Function.dropdown_profile import DropdownMenu  # validate import
    except ModuleNotFoundError:
        ROOT_LOCAL = Path(__file__).parent.resolve()
        if str(ROOT_LOCAL) not in sys.path:
            sys.path.insert(0, str(ROOT_LOCAL))
        from Function.dropdown_profile import DropdownMenu


    class _DummyController:
        def show_frame(self, *args, **kwargs):
            print(f"Navigate to: {args}")


    root = tk.Tk()
    root.title("Churn - Frame08")
    root.geometry("1440x1024")
    root.configure(bg="#FFFFFF")
    app = Frame08(root, _DummyController())
    app.pack(fill="both", expand=True)
    root.resizable(False, False)
    root.mainloop()

