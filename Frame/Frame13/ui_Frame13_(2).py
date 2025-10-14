# gui_content_embed.py
# Chuyển gui.py thành module build_content(parent, width, height)
# Giữ nguyên assets_Frame07/relative_to_assets như file gốc gui.py

from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets")  # y như gui.py

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def build_content(parent: tk.Widget, width: int, height: int) -> Canvas:
    """
    Dựng toàn bộ 'Data List' vào trong parent.
    Trả về Canvas để host có thể gắn scrollbar.
    """
    canvas = Canvas(
        parent,
        bg="#D4C5D2",     # nền cùng tông như gui.py
        width=width,
        height=height,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.pack(fill="both", expand=True)

    # Giữ reference ảnh để tránh bị GC
    _img_refs = []
    def _img(name: str) -> PhotoImage:
        im = PhotoImage(file=relative_to_assets(name))
        _img_refs.append(im)
        return im
    canvas._img_refs = _img_refs

    # ------- (giữ nguyên toạ độ, fonts, text) -------
    image_image_1 = _img("image_1_1.png")
    canvas.create_image(514.0, 348.0, image=image_image_1)

    canvas.create_text(34.0, 27.0, anchor="nw",
        text="Data List", fill="#706093", font=("YoungSerif Regular", 26 * -1))

    canvas.create_text(529.0, 80.0, anchor="nw",
        text="Search:", fill="#B992B9", font=("CrimsonPro Regular", 20 * -1))

    entry_image_1 = _img("entry_1_1.png")
    canvas.create_image(779.5, 91.5, image=entry_image_1)
    entry_1 = Entry(canvas, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(612.0, 71.0, window=entry_1, anchor="nw", width=335.0, height=39.0)

    canvas.create_text(393.0, 221.0, anchor="nw",
        text="No data availavle in table...", fill="#B992B9", font=("CrimsonPro Regular", 20 * -1))

    image_image_2 = _img("image_2_1.png"); canvas.create_image(149.0, 165.0, image=image_image_2)
    image_image_3 = _img("image_3_1.png"); canvas.create_image(295.0, 165.0, image=image_image_3)
    image_image_4 = _img("image_4_1.png"); canvas.create_image(441.0, 165.0, image=image_image_4)
    image_image_5 = _img("image_5_1.png"); canvas.create_image(587.0, 165.0, image=image_image_5)
    image_image_6 = _img("image_6_1.png"); canvas.create_image(733.0, 165.0, image=image_image_6)
    image_image_7 = _img("image_7_1.png"); canvas.create_image(879.0, 165.0, image=image_image_7)

    canvas.create_rectangle(42.0, 658.0000001466271, 984.9999881741605, 662.0,
                            fill="#E0E0E0", outline="")

    canvas.create_text(110.0, 154.0, anchor="nw", text="Variable 1", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    canvas.create_text(256.0, 154.0, anchor="nw", text="Variable 2", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    canvas.create_text(402.0, 154.0, anchor="nw", text="Variable 3", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    canvas.create_text(548.0, 154.0, anchor="nw", text="Variable 4", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    canvas.create_text(694.0, 154.0, anchor="nw", text="Variable 5", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    canvas.create_text(840.0, 154.0, anchor="nw", text="Variable 6", fill="#FFFFFF",
                       font=("YoungSerif Regular", 16 * -1))
    # -----------------------------------------------------------------------

    # Luôn cập nhật scrollregion theo nội dung thực tế
    def _sync_scrollregion(_=None):
        bbox = canvas.bbox("all") or (0, 0, canvas.winfo_width(), canvas.winfo_height())
        canvas.configure(scrollregion=bbox)
    canvas.bind("<Configure>", _sync_scrollregion)
    canvas.after(0, _sync_scrollregion)

    return canvas
