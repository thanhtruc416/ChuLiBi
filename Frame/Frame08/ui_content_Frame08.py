# ui_content_Frame08.py
# Chuyển file Tkinter-Designer thành module build_content(parent)
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, PhotoImage, Entry, Button

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets_Frame08_2")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def build_content(parent: tk.Widget, width: int, height: int) -> Canvas:
    """
    Dựng toàn bộ nội dung vào trong 'parent' (khối bên phải).
    Trả về Canvas để khung chính gắn scrollbar.
    """
    canvas = Canvas(
        parent,
        bg="#D4C5D2",  # nền đúng tông khối nội dung
        width=width,
        height=height,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )
    canvas.pack(fill="both", expand=True)

    # Giữ reference ảnh để tránh GC
    _img_refs = []
    def _img(name: str) -> PhotoImage:
        im = PhotoImage(file=relative_to_assets(name))
        _img_refs.append(im)
        return im
    canvas._img_refs = _img_refs

    # ======= Toàn bộ block item gốc (giữ nguyên toạ độ) =======
    image_BGPredict = _img("image_BGPredict.png")
    canvas.create_image(555.0, 1731.0, image=image_BGPredict)

    btn_img_predict = _img("button_Predict.png")
    btn_predict = Button(parent, image=btn_img_predict, bd=0, highlightthickness=0,
                         relief="flat", command=lambda: print("Predict clicked"))
    canvas.create_window(859.0, 1618.0, window=btn_predict, anchor="nw",
                         width=118.0, height=48.0)

    canvas.create_text(634.0, 1591.0, anchor="nw", text="Cluster",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    canvas.create_text(66.0, 1590.0, anchor="nw", text="Age",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))

    # Age
    img_age = _img("entry_Age.png")
    canvas.create_image(156, 1639.7264, image=img_age)
    entry_age = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(75, 1622.5, window=entry_age, anchor="nw",
                         width=164.5294, height=32.1118)

    canvas.create_text(699.0, 1672.0, anchor="nw", text="85%",
                       fill="#706093", font=("Crimson Pro SemiBold", 96 * -1))
    canvas.create_text(693.0, 1781.0, anchor="nw", text="High risk of late delivery",
                       fill="#6B6B7A", font=("Crimson Pro", 18 * -1))

    # Cluster
    img_cluster = _img("entry_Cluster.png")
    canvas.create_image(724.0, 1640.0, image=img_cluster)
    entry_cluster = Entry(parent, bd=0, bg="#D4C5D2", fg="#000716", highlightthickness=0)
    canvas.create_window(642.0, 1625.0, window=entry_cluster, anchor="nw",
                         width=164.0, height=32.0)

    # Gender
    canvas.create_text(69.0, 1680.0, anchor="nw", text="Gender",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    img_gender = _img("entry_Gender.png")
    canvas.create_image(159.0, 1730.0, image=img_gender)
    entry_gender = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(77.0, 1713.0, window=entry_gender, anchor="nw",
                         width=164.5, height=33)

    # Order Value
    canvas.create_text(69.0, 1763.0, anchor="nw", text="Order Value",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    img_ov = _img("entry_OrderValue.png")
    canvas.create_image(159.0, 1813.0, image=img_ov)
    entry_ov = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(77.0, 1796.0, window=entry_ov, anchor="nw",
                         width=164.5, height=32.0)

    # Occupation
    canvas.create_text(342.0, 1590.0, anchor="nw", text="Occupation",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    img_occ = _img("entry_Occupation.png")
    canvas.create_image(432.0, 1640.0, image=img_occ)
    entry_occ = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(350.0, 1625.0, window=entry_occ, anchor="nw",
                         width=164.5, height=32.0)

    # Delivery
    canvas.create_text(342.0, 1680.0, anchor="nw", text="Delivery",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    img_del = _img("entry_Delivery.png")
    canvas.create_image(432.0, 1730.0, image=img_del)
    entry_del = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(350.0, 1714.0, window=entry_del, anchor="nw",
                         width=164.5, height=32.0)

    # Preference
    canvas.create_text(342.0, 1763.0, anchor="nw", text="Preference",
                       fill="#483969", font=("Crimson Pro SemiBold", 18 * -1))
    img_pref = _img("entry_Preference.png")
    canvas.create_image(432.0, 1813.0, image=img_pref)
    entry_pref = Entry(parent, bd=0, bg="#FFFFFF", fg="#000716", highlightthickness=0)
    canvas.create_window(350.0, 1796.0, window=entry_pref, anchor="nw",
                         width=164.5, height=32.0)

    # Các block phía trên
    img_shap = _img("image_SHAP.png")
    canvas.create_image(807.0, 1275.0, image=img_shap)
    canvas.create_text(592.0, 1047.0, anchor="nw", text="SHAP Summary Plot",
                       fill="#000000", font=("Young Serif", 20 * -1))
    img_feat = _img("image_FeatureImportant.png")
    canvas.create_image(280.0, 1275.0, image=img_feat)
    canvas.create_text(55.0, 1047.0, anchor="nw", text="Feature Important",
                       fill="#000000", font=("Young Serif", 20 * -1))
    img_table = _img("image_Table.png")
    canvas.create_image(538.0, 764.0, image=img_table)
    canvas.create_text(48.0, 960.0, anchor="nw", text="Detail Analysis",
                       fill="#706093", font=("Young Serif", 32 * -1))
    canvas.create_text(69.0, 630.0, anchor="nw", text="Model Evaluation Metrics Table",
                       fill="#000000", font=("Young Serif", 20 * -1))
    img_reason = _img("image_ReasonsChart.png")
    canvas.create_image(880.0, 380.0, image=img_reason)
    canvas.create_text(740.0, 216.0, anchor="nw", text="Reasons Chart",
                       fill="#000000", font=("Young Serif", 20 * -1))
    img_rate = _img("image_ChurnRate.png")
    canvas.create_image(354.0, 380.0, image=img_rate)
    canvas.create_text(59.0, 213.0, anchor="nw", text="Churn Rate by Customer Segment",
                       fill="#000000", font=("Young Serif", 20 * -1))
    img_best = _img("image_LogisticRegression.png")
    canvas.create_image(880.0, 104.0, image=img_best)
    canvas.create_text(790.0, 79.0, anchor="nw", text="    Logistic\n  Regression",
                       fill="#706093", font=("Young Serif", 32 * -1))
    canvas.create_text(830.0, 47.0, anchor="nw", text="Best Model",
                       fill="#000000", font=("Young Serif", 18 * -1))
    img_clusters = _img("image_Clusters.png")
    canvas.create_image(525.0, 101.0, image=img_clusters)
    canvas.create_text(484.0, 138.0, anchor="nw", text="Clusters",
                       fill="#000000", font=("Young Serif", 18 * -1))
    canvas.create_text(509.0, 94.0, anchor="nw", text="4",
                       fill="#706093", font=("Kodchasan Regular", 45 * -1))
    img_icon_clusters = _img("image_iconClusters.png")
    canvas.create_image(526.0, 56.628395080566406, image=img_icon_clusters)
    img_avg = _img("image_AvgChurn.png")
    canvas.create_image(179.0, 101.0, image=img_avg)
    canvas.create_text(130.0, 145.0, anchor="nw", text="Avg Churn",
                       fill="#000000", font=("Young Serif", 18 * -1))
    canvas.create_text(150.0, 128.0, anchor="nw", text="220/499",
                       fill="#979797", font=("Kodchasan Regular", 11 * -1))
    canvas.create_text(140.0, 88.0, anchor="nw", text="44%",
                       fill="#706093", font=("Kodchasan Regular", 40 * -1))
    img_icon_avg = _img("image_iconAvgChurn.png")
    canvas.create_image(169.0, 58.0, image=img_icon_avg)
    # ======= Hết block gốc =======

    # Luôn set scrollregion theo nội dung
    def _sync_scrollregion(_=None):
        bbox = canvas.bbox("all")
        if bbox is None:
            bbox = (0, 0, canvas.winfo_width(), canvas.winfo_height())
        canvas.configure(scrollregion=bbox)

    canvas.bind("<Configure>", _sync_scrollregion)
    canvas.after(0, _sync_scrollregion)

    return canvas
