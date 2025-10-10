# host_test_riu.py
# Dùng layout/ảnh y như test_riu.py làm nền, rồi chèn frame 'Data List' (gui) + scrollbar bên phải.
# KHÔNG sửa đường dẫn assets của test_riu: ASSETS_PATH = Path("assets")
import sys
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Scrollbar, Frame

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from ui_Frame13_2 import build_content
ASSETS_PATH = Path("assets")  # y như test_riu.py
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

ASSETS_PATH = Path("assets")  # y như test_riu.py
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# ========== APP ==========
window = Tk()
#fit với màn hình laptop( nếu ko cần thì xóa 5 dòng tiếp theo)
window.title("Data Management")
try:
    window.state("zoomed")          # Windows/macOS: fit full screen
except Exception:
    window.attributes("-zoomed", True)  # fallback một số Linux

window.resizable(True, True)
window.geometry("1440x1024")
window.configure(bg="#D4C5D2")

# ---- NỀN (giữ nguyên từ test_riu.py) ----
canvas = Canvas(window, bg="#D4C5D2", height=1024, width=1440, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

canvas.create_text(98.0, 927.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4", font=("Rubik Burned Regular", 35 * -1))

image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
canvas.create_image(889.0, 194.0, image=image_image_1)

canvas.create_text(399.0, 122.0, anchor="nw", text="Import Data", fill="#706093", font=("Young Serif", 26 * -1))
canvas.create_text(399.0, 162.0, anchor="nw", text="Choose Data File (CSV/JSON)", fill="#B992B9", font=("Crimson Pro", 20 * -1))
canvas.create_text(1051.0, 162.0, anchor="nw", text="Or you can see the Reference File: ", fill="#B992B9", font=("Crimson Pro", 20 * -1))

image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
canvas.create_image(168.0, 513.0, image=image_image_3)

canvas.create_text(98.0, 928.0, anchor="nw", text="ChuLiBi", fill="#FDE5F4", font=("Rubik Burned Regular", 35 * -1))

image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
canvas.create_image(162.0, 102.0, image=image_image_4)

# Sidebar buttons (giữ nguyên)
button_image_CustomerAnalysis = PhotoImage(file=relative_to_assets("button_CustomerAnalysis.png"))
Button(image=button_image_CustomerAnalysis, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_CustomerAnalysis clicked"), relief="flat"
).place(x=0.0, y=303.0, width=337.0, height=77.0)

button_image_Recommendation = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
Button(image=button_image_Recommendation, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Recommendation clicked"), relief="flat"
).place(x=0.0, y=469.0, width=336.0, height=82.0)

button_image_Churn = PhotoImage(file=relative_to_assets("button_Churn.png"))
Button(image=button_image_Churn, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Churn clicked"), relief="flat"
).place(x=0.0, y=382.0, width=336.0, height=86.0)

button_image_Delivery = PhotoImage(file=relative_to_assets("button_Delivery.png"))
Button(image=button_image_Delivery, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Delivery clicked"), relief="flat"
).place(x=0.0, y=553.0, width=337.0, height=90.0)

button_image_Dashboard = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
Button(image=button_image_Dashboard, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Dashboard clicked"), relief="flat"
).place(x=0.0, y=222.0, width=337.0, height=79.0)

image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
canvas.create_image(887.0, 42.0, image=image_image_5)

canvas.create_text(381.0, 14.0, anchor="nw", text="Data Management", fill="#000000", font=("Young Serif", 40 * -1))

button_image_Profile = PhotoImage(file=relative_to_assets("button_Profile.png"))
Button(image=button_image_Profile, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Profile clicked"), relief="flat"
).place(x=1360.0, y=17.0, width=44.0, height=45.0)

image_image_6 = PhotoImage(file=relative_to_assets("image_6.png"))
canvas.create_image(1321.0, 42.0, image=image_image_6)

canvas.create_text(765.0, 50.0, anchor="nw", text="View and import customer or order data",
                   fill="#FFFFFF", font=("Crimson Pro", 18 * -1))

image_image_7 = PhotoImage(file=relative_to_assets("image_7.png"))
canvas.create_image(573.0, 221.0, image=image_image_7)

canvas.create_text(577.0, 210.0, anchor="nw", text="No file chosen", fill="#B992B9", font=("Crimson Pro", 20 * -1))

button_image_ChooseFile = PhotoImage(file=relative_to_assets("button_ChooseFile.png"))
Button(image=button_image_ChooseFile, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_ChooseFile clicked"), relief="flat"
).place(x=408.0, y=207.0, width=153.0, height=28.0)

button_image_ImportData = PhotoImage(file=relative_to_assets("button_ImportData.png"))
Button(image=button_image_ImportData, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_ImportData clicked"), relief="flat"
).place(x=776.0, y=193.0, width=219.0, height=57.0)

canvas.create_rectangle(416.0, 949.0000001466271, 1358.9999881741605, 953.0, fill="#E0E0E0", outline="")

button_image_ReferenceFile = PhotoImage(file=relative_to_assets("button_ReferenceFile.png"))
Button(image=button_image_ReferenceFile, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_ReferenceFile clicked"), relief="flat"
).place(x=1090.0, y=192.0, width=236.0, height=58.0)

button_image_Report = PhotoImage(file=relative_to_assets("button_Report.png"))
Button(image=button_image_Report, borderwidth=0, highlightthickness=0,
       command=lambda: print("button_Report clicked"), relief="flat"
).place(x=0.0, y=643.0, width=338.0, height=88.0)

# ---------- KHUNG 'DATA LIST' + SCROLLBAR SÁT PHẢI ----------
APP_W, APP_H = 1440, 1024
# Toạ độ neo theo ảnh nền trong screenshot: dưới phần "Import Data"
RIGHT_X = 375   # canh lề trái panel nội dung
RIGHT_Y = 288   # nằm dưới dải "Import Data"
BAR_W  = 20     # độ rộng thanh cuộn ngoài cùng bên phải

right_content = Frame(window, bg="#D4C5D2")
right_content.place(x=RIGHT_X, y=RIGHT_Y)

RIGHT_W = APP_W - RIGHT_X - BAR_W
RIGHT_H = APP_H - RIGHT_Y - 20
canvas_content = build_content(right_content, RIGHT_W, RIGHT_H)

right_gutter = Frame(window, bg="#D4C5D2")
right_gutter.place(x=APP_W - BAR_W, y=RIGHT_Y, width=BAR_W, height=RIGHT_H)

#scroll dọc
vbar = Scrollbar(right_gutter, orient="vertical", command=canvas_content.yview,
                 width=BAR_W, bd=0, highlightthickness=0)
vbar.pack(fill="y", side="left")
canvas_content.configure(yscrollcommand=vbar.set)
#scroll ngang:
HBAR_H = 18  # chiều cao thanh cuộn ngang

bottom_gutter = Frame(window, bg="#D4C5D2")
bottom_gutter.place(x=RIGHT_X, y=RIGHT_Y + RIGHT_H - HBAR_H, width=RIGHT_W, height=HBAR_H)

hbar = Scrollbar(bottom_gutter, orient="horizontal",
                 command=canvas_content.xview, width=HBAR_H, bd=0, highlightthickness=0)
hbar.pack(fill="x", side="top")

canvas_content.configure(xscrollcommand=hbar.set)
#===============================
def _sync_scrollregion():
    bbox = canvas_content.bbox("all") or (0, 0, RIGHT_W, RIGHT_H)
    x0, y0, x1, y1 = bbox
    canvas_content.configure(scrollregion=(0, 0, max(RIGHT_W, x1 - x0), max(RIGHT_H, y1 - y0)))

def _sync_sizes(_=None):
    app_w = window.winfo_width() or APP_W
    app_h = window.winfo_height() or APP_H
    w = max(100, app_w - RIGHT_X - BAR_W)
    h = max(100, app_h - RIGHT_Y)
    content_h = max(120, h - HBAR_H)
    right_content.configure(width=w, height=h)
    canvas_content.configure(width=w, height=h)
    right_gutter.place_configure(x=app_w - BAR_W, y=RIGHT_Y, width=BAR_W, height=h)
    window.after_idle(_sync_scrollregion)

window.bind("<Configure>", _sync_sizes)
window.after_idle(_sync_sizes)

# Cuộn chuột cross-platform (chỉ cuộn khi chuột ở vùng content)
_scroll_active = {"on": False}
canvas_content.bind("<Enter>", lambda e: _scroll_active.__setitem__("on", True))
canvas_content.bind("<Leave>", lambda e: _scroll_active.__setitem__("on", False))

def _yview_units(steps: int):
    canvas_content.yview_scroll(steps, "units")

def _on_mousewheel(event):
    if not _scroll_active["on"]:
        return
    d = event.delta
    if d == 0: return
    steps = -int(d/120) if abs(d) >= 120 else (-1 if d > 0 else 1)
    _yview_units(steps)

window.bind_all("<MouseWheel>", _on_mousewheel)           # Win/macOS
window.bind_all("<Button-4>",  lambda e: _scroll_active["on"] and _yview_units(-1))  # Linux
window.bind_all("<Button-5>",  lambda e: _scroll_active["on"] and _yview_units(1))   # Linux
# ------------------------------------------------------------
# ==== [THÊM Ở ĐÂY] Shift + lăn chuột => cuộn ngang ====
def _on_shift_mousewheel(event):
    if not _scroll_active["on"]:
        return
    d = event.delta
    if d == 0:
        return
    steps = -int(d/120) if abs(d) >= 120 else (-1 if d > 0 else 1)
    canvas_content.xview_scroll(steps, "units")

window.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

window.resizable(False, False)
window.mainloop()

