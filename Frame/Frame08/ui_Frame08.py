# ui_Frame08.py — FULL sidebar buttons + content scroll bên phải

from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Scrollbar, Frame
from ui_content_Frame08 import build_content

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame08")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# ================== APP ==================
window = Tk()
window.geometry("1440x1024")
window.configure(bg="#FFFFFF")

# NỀN/ASSETS TỪ TKINTER DESIGNER (sidebar, topbar…)
canvas = Canvas(window, bg="#FFFFFF", height=1088, width=1440,
                bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

# --- nền trái/phải (giữ như file gốc) ---
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
canvas.create_image(723.0, 544.0, image=image_image_1)

image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
canvas.create_image(173.0, 544.0, image=image_image_2)

# logo chữ dưới cùng
canvas.create_text(92.0, 942.0, anchor="nw",
                   text="ChuLiBi", fill="#FDE5F4",
                   font=("Rubik Burned Regular", 35 * -1))

# logo ong
image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
canvas.create_image(169.0, 103.0, image=image_image_3)

# ---- BUTTONS TRÊN SIDEBAR (đầy đủ) ----
btn_customer_img = PhotoImage(file=relative_to_assets("button_Customer_analysis.png"))
Button(image=btn_customer_img, bd=0, highlightthickness=0,
       command=lambda: print("Customer analysis"), relief="flat"
).place(x=6.0, y=304.0, width=339.0, height=78.0)

btn_reco_img = PhotoImage(file=relative_to_assets("button_Recommendation.png"))
Button(image=btn_reco_img, bd=0, highlightthickness=0,
       command=lambda: print("Recommendation"), relief="flat"
).place(x=6.0, y=470.0, width=338.0, height=83.0)

btn_delivery_img = PhotoImage(file=relative_to_assets("button_Delivery.png"))
Button(image=btn_delivery_img, bd=0, highlightthickness=0,
       command=lambda: print("Delivery"), relief="flat"
).place(x=6.0, y=554.0, width=338.0, height=91.0)

btn_report_img = PhotoImage(file=relative_to_assets("button_Report.png"))
Button(image=btn_report_img, bd=0, highlightthickness=0,
       command=lambda: print("Report"), relief="flat"
).place(x=7.0, y=648.0, width=337.0, height=88.0)

btn_dashboard_img = PhotoImage(file=relative_to_assets("button_Dashboard.png"))
Button(image=btn_dashboard_img, bd=0, highlightthickness=0,
       command=lambda: print("Dashboard"), relief="flat"
).place(x=2.0, y=194.0, width=341.0, height=89.0)

btn_churn_img = PhotoImage(file=relative_to_assets("button_Churn.png"))
Button(image=btn_churn_img, bd=0, highlightthickness=0,
       command=lambda: print("Churn"), relief="flat"
).place(x=8.0, y=385.5, width=336.0, height=80.0)

# icon trang trí nhỏ (nếu cần)
image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
canvas.create_image(14.0, 426.0, image=image_image_5)

# ---- TOP BAR ----
image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
canvas.create_image(892.0, 31.0, image=image_image_4)

canvas.create_text(348.0, 2.0, anchor="nw",
                   text="   Churn", fill="#000000",
                   font=("Young Serif", 40 * -1))

btn_profile_img = PhotoImage(file=relative_to_assets("button_Profile.png"))
Button(image=btn_profile_img, bd=0, highlightthickness=0,
       command=lambda: print("Profile"), relief="flat"
).place(x=1371.0, y=10.0, width=46.0, height=40.0)

btn_noti_img = PhotoImage(file=relative_to_assets("button_Notification.png"))
Button(image=btn_noti_img, bd=0, highlightthickness=0,
       command=lambda: print("Notification"), relief="flat"
).place(x=1306.0, y=10.0, width=46.0, height=43.0)

# ================== NỘI DUNG + SCROLLBAR SÁT PHẢI ==================
APP_W, APP_H = 1440, 1024

# Lề trái vùng nội dung (đứng sau sidebar). TĂNG số này nếu còn lấn.
RIGHT_X = 340     # <— chỉnh 290~310 tùy file nền/ảnh sidebar của bạn
RIGHT_Y = 65
BAR_W   = 20      # độ rộng thanh cuộn ngoài cùng bên phải

# Khung chứa nội dung
right_content = Frame(window, bg="#FFFFFF")
right_content.place(x=RIGHT_X, y=RIGHT_Y)

# Vẽ content vào khung (Canvas trả về để gắn scrollbar)
RIGHT_W = APP_W - RIGHT_X - BAR_W
RIGHT_H = APP_H - RIGHT_Y
canvas_content = build_content(right_content, RIGHT_W, RIGHT_H)

# Gutter (rãnh) sát mép phải cửa sổ cho scrollbar
right_gutter = Frame(window, bg="#FFFFFF")
right_gutter.place(x=APP_W - BAR_W, y=RIGHT_Y, width=BAR_W, height=RIGHT_H)

# Scrollbar dọc
vbar = Scrollbar(right_gutter, orient="vertical",
                 command=canvas_content.yview,
                 width=BAR_W, bd=0, highlightthickness=0)
vbar.pack(fill="y", side="left")
canvas_content.configure(yscrollcommand=vbar.set)

def _sync_sizes(_=None):
    app_w = window.winfo_width() or APP_W
    app_h = window.winfo_height() or APP_H

    w = max(100, app_w - RIGHT_X - BAR_W)
    h = max(100, app_h - RIGHT_Y)

    right_content.configure(width=w, height=h)
    canvas_content.configure(width=w, height=h)

    # đặt scrollbar SÁT mép phải cửa sổ
    right_gutter.place_configure(x=app_w - BAR_W, y=RIGHT_Y, width=BAR_W, height=h)

    # scrollregion theo nội dung thực tế
    bbox = canvas_content.bbox("all")
    if not bbox:
        bbox = (0, 0, w, h)
    x0, y0, x1, y1 = bbox
    canvas_content.configure(scrollregion=(0, 0, max(w, x1 - x0), max(h, y1 - y0)))

window.after(60, _sync_sizes)
window.bind("<Configure>", _sync_sizes)

# Cuộn chuột
def _on_mousewheel(event):
    delta = event.delta
    if delta == 0:
        return
    step = -1 if delta > 0 else 1
    canvas_content.yview_scroll(step, "units")

# Win/macOS
canvas_content.bind_all("<MouseWheel>", _on_mousewheel)
# Linux
canvas_content.bind_all("<Button-4>", lambda e: canvas_content.yview_scroll(-1, "units"))
canvas_content.bind_all("<Button-5>", lambda e: canvas_content.yview_scroll(1, "units"))

window.resizable(False, False)
window.mainloop()
