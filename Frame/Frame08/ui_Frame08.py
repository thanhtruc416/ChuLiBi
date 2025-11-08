from pathlib import Path
import sys
import tkinter as tk
from tkinter import Canvas, Button, PhotoImage

from Function.dropdown_profile import DropdownMenu

# =========================
# Asset resolver (auto find)
# =========================
OUTPUT_PATH = Path(__file__).parent
_ASSET_DIRS = [
    OUTPUT_PATH / "assets_Frame08_2",
    OUTPUT_PATH / "assets_Frame08",
    OUTPUT_PATH / "assets_frame0",
    OUTPUT_PATH / "assets",
    OUTPUT_PATH,
]
def _resolve_asset(path: str) -> Path:
    name = Path(path).name
    for d in _ASSET_DIRS:
        p = d / name
        if p.exists():
            return p
    low = name.lower()
    for d in _ASSET_DIRS:
        try:
            for f in d.iterdir():
                if f.name.lower() == low:
                    return f
        except Exception:
            pass
    return _ASSET_DIRS[0] / name

def relative_to_assets(path: str) -> Path:
    return _resolve_asset(path)

# =========================
# Import ui_content_Frame08
# =========================
PROJECT_ROOT = OUTPUT_PATH.parent.parent  # .../ChuLiBi
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from ui_content_Frame08 import build_content as build_content_frame08
except Exception:
    try:
        from Frame.Frame08.ui_content_Frame08 import build_content as build_content_frame08
    except Exception as e:
        print("Warning: cannot import ui_content_Frame08.build_content ->", e)
        build_content_frame08 = None


class Frame08(tk.Frame):
    """
    Khung chính của màn Churn (Frame08):
    - Giữ nguyên layout từ Tkinter Designer (ảnh/nút/coords).
    - Nhúng content động (charts + table) ở khu vực bên phải, có scrollbar dọc.
    """
    def __init__(self, parent, controller=None):
        super().__init__(parent, bg="#E2E2E2")
        self.controller = controller
        self.lower()

        # === MAIN CANVAS (giữ layout gốc) ===
        self.canvas = Canvas(self, bg="#E2E2E2", height=1024, width=1440,
                             bd=0, highlightthickness=0, relief="ridge")
        self.canvas.place(x=0, y=0)

        # === Images / giữ reference tránh GC ===
        self._img_refs = []
        def _img(fname: str):
            try:
                im = PhotoImage(file=relative_to_assets(fname))
                self._img_refs.append(im)
                return im
            except Exception as e:
                print(f"[asset] Could not load {fname}: {e}")
                return None

        # ======= BACKGROUND IMAGES =======
        img = _img("image_1.png")
        if img: self.canvas.create_image(720.0, 512.0, image=img)

        img = _img("image_2.png")
        if img: self.canvas.create_image(168.0, 514.0, image=img)

        self.canvas.create_text(81.0, 914.0, anchor="nw",
                                text="ChuLiBi", fill="#FDE5F4",
                                font=("Rubik Burned Regular", 35 * -1))

        img = _img("image_3.png")
        if img: self.canvas.create_image( 162.0, 101.0, image=img)

        img = _img("image_4.png")
        if img: self.canvas.create_image(887.0, 31.0, image=img)

        self.canvas.create_text(348.0, 2.0, anchor="nw",
                                text="   Churn", fill="#000000",
                                font=("Young Serif", 40 * -1))

        img = _img("button_Notification.png")
        if img: self.canvas.create_image(1323.0, 31.0, image=img)

        # ======= BUTTONS (ảnh mới theo label) =======
        # --- Dropdown/Profile button ---
        self.button_Profile_image = PhotoImage(file=relative_to_assets("button_Profile.png"))
        self.dropdown = DropdownMenu(self)
        self.button_Profile = Button(
            self,
            image=self.button_Profile_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.dropdown.show,
            relief="flat"
        )
        self.button_Profile.place(x=1359.0, y=6.0, width=50.0, height=45.0)
        # --- Sidebar buttons --

        self.button_image_1 = PhotoImage(
            file=relative_to_assets("button_EL.png"))
        button_EL = Button(
            self,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame09_EL"),
            relief="flat"
        )
        button_EL.place(
            x=0.0,
            y=466.0,
            width=338.0,
            height=81.0
        )

        self.button_image_2 = PhotoImage(
            file=relative_to_assets("button_Recommendation.png"))
        button_Recommendation = Button(
            self,
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame10"),
            relief="flat"
        )
        button_Recommendation.place(
            x=0.0,
            y=547.0,
            width=338.0,
            height=81.0
        )

        self.button_image_3 = PhotoImage(
            file=relative_to_assets("button_PredictCustomer.png"))
        button_PredictCustomer = Button(
            self,
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame11"),
            relief="flat"
        )
        button_PredictCustomer.place(
            x=0.0,
            y=628.0,
            width=338.0,
            height=81.0
        )

        self.button_image_4 = PhotoImage(
            file=relative_to_assets("button_Dashboard.png"))
        button_Dashboard = Button(
            self,
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame06"),
            relief="flat"
        )
        button_Dashboard.place(
            x=0.0,
            y=223.0,
            width=338.0,
            height=81.0
        )

        self.button_image_5 = PhotoImage(
            file=relative_to_assets("button_Churn.png"))
        button_Churn = Button(
            self,
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame08"),
            relief="flat"
        )
        button_Churn.place(
            x=0.0,
            y=385.0,
            width=338.0,
            height=81.0
        )

        self.button_image_6 = PhotoImage(
            file=relative_to_assets("button_CustomerAnalysis.png"))
        button_CustomerAnalysis = Button(
            self,
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Frame07"),
            relief="flat"
        )
        button_CustomerAnalysis.place(
            x=0.0,
            y=304.0,
            width=338.0,
            height=81.0
        )
        # ======= CONTENT HOST (embedded ui_content_Frame08 + SCROLLBAR) =======
        self.CONTENT_X, self.CONTENT_Y = 370, 70
        self.CONTENT_W, self.CONTENT_H = 1040, 900

        self.content_host = tk.Frame(self, bg="#E2E2E2", highlightthickness=0, bd=0)
        self.canvas.create_window(self.CONTENT_X, self.CONTENT_Y,
                                  window=self.content_host, anchor="nw",
                                  width=self.CONTENT_W, height=self.CONTENT_H)

        # wrapper + content_area
        self.wrapper = tk.Frame(self.content_host, bg="#E2E2E2")
        self.wrapper.pack(fill="both", expand=True, padx=(0, 20))

        self.content_area = tk.Frame(self.wrapper, bg="#E2E2E2")
        self.content_area.pack(fill="both", expand=True)

        # scrollbar dọc — đặt "in_" = content_host để dán vào mép phải khung content
        self.vbar = tk.Scrollbar(
            self.content_host, orient="vertical",
            takefocus=0, highlightthickness=0, bd=0, relief="flat"
        )
        self.vbar.place(in_=self.content_host, relx=1.0, rely=0.0,
                        relheight=1.0, x=0, anchor="ne")

        self._content_canvas = None
        self._mount_content()   # build & wire scroll
        self._bind_mousewheel() # wheel support

    # ---------- mount nội dung từ ui_content_Frame08 ----------
    def _mount_content(self):
        for w in self.content_area.winfo_children():
            w.destroy()

        if build_content_frame08 is None:
            tk.Label(self.content_area, text="(ui_content_Frame08 not available)",
                     bg="#E2E2E2", fg="#3a2a68", font=("Crimson Pro", 14)).pack(expand=True, fill="both")
            return

        try:
            # build_content trả về Canvas có scrollregion
            self._content_canvas = build_content_frame08(
                self.content_area,
                width=self.CONTENT_W - 20,
                height=self.CONTENT_H - 20
            )
        except Exception as e:
            print("Error mounting ui_content_Frame08:", e)
            tk.Label(self.content_area, text=f"Error loading content: {e}",
                     bg="#E2E2E2", fg="#B00020", font=("Crimson Pro", 12)).pack(expand=True, fill="both")
            self._content_canvas = None
            return

        if isinstance(self._content_canvas, Canvas):
            self._content_canvas.configure(yscrollcommand=self.vbar.set)
            self.vbar.configure(command=self._content_canvas.yview)

            def _sync_region(_=None):
                try:
                    self._content_canvas.configure(scrollregion=self._content_canvas.bbox("all"))
                except Exception:
                    pass
            self._content_canvas.bind("<Configure>", _sync_region)
            self.after(120, _sync_region)

    def _bind_mousewheel(self):
        # Bind wheel cho Windows/macOS + Linux
        target = self.content_area
        def _on_wheel(event):
            if not self._content_canvas:
                return
            delta = 0
            if event.num == 4:   # Linux up
                delta = 120
            elif event.num == 5: # Linux down
                delta = -120
            elif hasattr(event, "delta"):
                delta = event.delta
            if delta:
                self._content_canvas.yview_scroll(-1 if delta > 0 else 1, "units")
        target.bind_all("<MouseWheel>", _on_wheel)   # Win/mac
        target.bind_all("<Button-4>", _on_wheel)     # Linux up
        target.bind_all("<Button-5>", _on_wheel)     # Linux down


# =========================
# Run standalone
# =========================
def run_this_frame_only():
    root = tk.Tk()
    root.title("Frame08 — Churn")
    root.geometry("1440x1024")
    root.configure(bg="#E2E2E2")
    root.resizable(False, False)

    app = Frame08(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    run_this_frame_only()
