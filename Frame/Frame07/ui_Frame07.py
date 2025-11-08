from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Button, PhotoImage
import tkinter.font as tkfont
import pandas as pd

import matplotlib.font_manager as fm
from matplotlib import rcParams

font_path = Path(__file__).resolve().parents[2] / "Font" / "Crimson_Pro" / "static" / "CrimsonPro-Regular.ttf"

if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    rcParams['font.family'] = 'Crimson Pro'
else:
    print("[WARNING] Font không tồn tại tại:", font_path)

try:
    from Function.dropdown_profile import DropdownMenu
except Exception:
    DropdownMenu = None  # fallback nếu chưa có

# ==== Chart helpers (y như bạn đang dùng) ====
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Function.Frame07_Cluster import (
    load_scaled_dataset, attach_group_pca, select_X,
    kmeans_labels, counts_by_cluster,
    figure_elbow_silhouette, figure_cluster_distribution, figure_pca_scatter,
    CLUSTER_FEATURES
)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets_Frame07")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class Frame07(tk.Frame):
    """
    Khung UI Customer Segmentation dạng class.
    - Không đổi layout/UI gốc (toạ độ, ảnh, text giữ nguyên).
    - Bổ sung glue để gắn biểu đồ và cập nhật số liệu.
    """
    def __init__(self, parent, controller=None):
        super().__init__(parent, bg="#D9D9D9")
        self.controller = controller
        self._imgs = {}
        self.dropdown = None
        self.canvas = None
        self.button_Profile = None
        self.lower()

        # build UI phần tĩnh
        self._build_static_ui()

        # dữ liệu & vẽ chart
        self._load_and_mount_charts()

    # ---------------- UI gốc (không đổi layout) ----------------
    def _build_static_ui(self):
        self.canvas = Canvas(
            self,
            bg="#D9D9D9",
            height=1024,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # --- Images & Text theo file gốc ---
        self._img("image_4.png")
        self.canvas.create_image(1130.0, 353.0, image=self._imgs["image_4.png"])

        self._img("image_5.png")
        self.canvas.create_image(1129.0, 788.0, image=self._imgs["image_5.png"])

        self._img("image_1.png")
        self.canvas.create_image(887.0, 31.0, image=self._imgs["image_1.png"])

        # Font cho bullets (KHÔNG NGHIÊNG)
        self.font_feature = tkfont.Font(
            family="Crimson Pro",
            size=-20,
            weight="normal",
            slant="roman"
        )

        self.canvas.create_text(
            337.0, 0.0, anchor="nw",
            text="   Customer Segmentation",
            fill="#000000", font=("Young Serif", 40 * -1)
        )

        self._img("image_2.png")
        self.canvas.create_image(1320.0, 31.0, image=self._imgs["image_2.png"])

        self._img("image_3.png")
        self.canvas.create_image(580.0, 205.0, image=self._imgs["image_3.png"])

        self.canvas.create_text(475.0, 85.0, anchor="nw",
            text="Elbow Method", fill="#706093", font=("Crimson Pro Bold", 35 * -1)
        )
        text_spaced = "   ".join("RESULT")
        self.canvas.create_text(985.0, 75.0, anchor="nw", text=text_spaced,
                        fill="#706093", font=("Young Serif", 40 * -1))

        self.canvas.create_rectangle(824.0, 62.0, 825.0, 353.0, fill="#D9D9D9", outline="")
        self.canvas.create_rectangle(332.0, 352.0, 825.0, 353.0, fill="#D9D9D9", outline="")

        self._img("image_4.png")
        self.canvas.create_image(1131.0, 350, image=self._imgs["image_4.png"])
        self.canvas.create_rectangle(
            970.0, 165.0, 1290.0, 195.0,
            fill="#FFFFFF",
            outline=""
        )
        # Text đè lên image_4
        self.canvas.create_text(
            1130.0, 190.0,
            text="Cluster Distribution Chart",
            fill="#706093",
            font=("Crimson Pro Bold", 30),
            anchor="center"
        )

        # --- Scatter Plot Chart (image_5) ---
        self._img("image_5.png")
        self.canvas.create_image(1130.0, 790, image=self._imgs["image_5.png"])

        self.canvas.create_rectangle(
            970.0, 600.0, 1310.0, 790.0,
            fill="#FFFFFF",
            outline=""
        )

        # Thêm tiêu đề mới
        self.canvas.create_text(
            1130.0, 625.0,
            text="Scatter Plot Chart",
            fill="#706093",
            font=("Crimson Pro Bold", 30),
            anchor="center"
        )

        self._img("image_6.png")
        self.canvas.create_image(588.0, 688.0, image=self._imgs["image_6.png"])

        self.canvas.create_text(450.0, 390.0, anchor="nw",
            text="Characteristics of", fill="#706093", font=("Crimson Pro Bold", 35 * -1)
        )
        self.canvas.create_text(408.0, 424.0, anchor="nw",
            text="Customer Segmentation", fill="#706093", font=("Crimson Pro Bold", 35 * -1)
        )

        # === placeholder cho 3 ô feature — LƯU ID để cập nhật về sau ===
        self.feature_text_ids = {}
        self.feature_text_ids[0] = self.canvas.create_text(
            609.0, 516.0, anchor="nw",
            text="• Feature A\n• Feature B\n• Feature C",
            fill="#000000", font=self.font_feature,
            width=230, justify="left"
        )
        self.feature_text_ids[1] = self.canvas.create_text(
            609.0, 689.0, anchor="nw",
            text="• Feature A\n• Feature B\n• Feature C",
            fill="#000000", font=self.font_feature,
            width=230, justify="left"
        )
        self.feature_text_ids[2] = self.canvas.create_text(
            609.0, 860.0, anchor="nw",
            text="• Feature A\n• Feature B\n• Feature C",
            fill="#000000", font=self.font_feature,
            width=230, justify="left"
        )

        self._img("image_7.png")
        self.canvas.create_image(588.0, 551.0, image=self._imgs["image_7.png"])
        self._img("image_8.png")
        self.canvas.create_image(588.0, 724.0, image=self._imgs["image_8.png"])
        self._img("image_9.png")
        self.canvas.create_image(588.0, 896.0, image=self._imgs["image_9.png"])

        # --- 3 con số mặc định (có lưu ID để cập nhật về sau) ---
        self.card_number_ids = {}
        self.card_number_ids[0] = self.canvas.create_text(
            435.0, 530.0, anchor="nw",
            text="244", fill="#000000", font=("Kodchasan Regular", 40 * -1)
        )
        self.card_number_ids[1] = self.canvas.create_text(
            435.0, 703.0, anchor="nw",
            text="244", fill="#000000", font=("Kodchasan Regular", 40 * -1)
        )
        self.card_number_ids[2] = self.canvas.create_text(
            435.0, 877.0, anchor="nw",
            text="244", fill="#000000", font=("Kodchasan Regular", 40 * -1)
        )

        self.canvas.create_text(421.0, 491.0, anchor="nw",
            text="Cluster 1", fill="#2E126A", font=("Young Serif", 20 * -1)
        )
        self.canvas.create_text(421.0, 665.0, anchor="nw",
            text="Cluster 2", fill="#2E126A", font=("Young Serif", 20 * -1)
        )
        self.canvas.create_text(421.0, 837.0, anchor="nw",
            text="Cluster 3", fill="#2E126A", font=("Young Serif", 20 * -1)
        )

        self.canvas.create_text(120.0, 918.0, anchor="nw",
            text="ChuLiBi", fill="#FDE5F4", font=("Rubik Burned Regular", 35 * -1)
        )
        self._img("image_sidebar_bg.png")
        self.canvas.create_image(168.0, 512.0, image=self._imgs["image_sidebar_bg.png"])
        self.canvas.create_text(98.0, 927.0, anchor="nw",
            text="ChuLiBi", fill="#FDE5F4", font=("Rubik Burned Regular", 35 * -1)
        )
        self._img("image_11.png")
        self.canvas.create_image(162.0, 101.0, image=self._imgs["image_11.png"])

        # --- Sidebar buttons ---
        self.button_image_1 = PhotoImage(
            file=relative_to_assets("button_Churn.png"))
        button_Churn = Button(self,
                              image=self.button_image_1,
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

        self.button_image_2 = PhotoImage(
            file=relative_to_assets("button_EL.png"))
        button_EL = Button(self,
                           image=self.button_image_2,
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

        self.button_image_3 = PhotoImage(
            file=relative_to_assets("button_Recommendation.png"))
        button_Recommendation = Button(self,
                                       image=self.button_image_3,
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

        self.button_image_4 = PhotoImage(
            file=relative_to_assets("button_PredictCustomer.png"))
        button_PredictCustomer = Button(self,
                                        image=self.button_image_4,
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

        self.button_image_5 = PhotoImage(
            file=relative_to_assets("button_Dashboard.png"))
        button_Dashboard = Button(self,
                                  image=self.button_image_5,
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

        self.button_image_6 = PhotoImage(
            file=relative_to_assets("button_CustomerAnalysis.png"))
        button_CustomerAnalysis = Button(self,
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

        # Profile (dropdown)
        self._img("button_Profile.png")
        self.button_Profile = Button(
            self,
            image=self._imgs["button_Profile.png"],
            borderwidth=0, highlightthickness=0,
            relief="flat",
            command=self._on_profile_clicked
        )
        self.button_Profile.place(x=1359.0, y=7.0, width=53.0, height=48.0)

    def _img(self, filename: str):
        if filename not in self._imgs:
            self._imgs[filename] = PhotoImage(file=relative_to_assets(filename))
        return self._imgs[filename]

    def _make_button(self, filename, cmd, x, y, w, h):
        self._img(filename)
        btn = Button(self, image=self._imgs[filename],
                     borderwidth=0, highlightthickness=0,
                     relief="flat", command=cmd)
        btn.place(x=x, y=y, width=w, height=h)
        return btn

    def _on_profile_clicked(self):
        if DropdownMenu is None:
            print("[Frame07] DropdownMenu chưa sẵn sàng.")
            return
        if self.dropdown is None:
            self.dropdown = DropdownMenu(self)  # truyền khung làm parent
        try:
            self.dropdown.show(self.button_Profile)
        except TypeError:
            self.dropdown.show()

    # ---------------- GLUE (gắn chart + số liệu) ----------------
    def _load_and_mount_charts(self):
        DATA_DIR = Path(__file__).resolve().parents[2] / "Dataset" / "Output"
        CSV_PATH = DATA_DIR / "df_scaled_model.csv"

        if not CSV_PATH.exists():
            self.canvas.create_text(
                360, 120, anchor="nw",
                text=f"[LỖI] Không tìm thấy {CSV_PATH}\nHãy kiểm tra lại cây thư mục.",
                fill="#b00020", font=("Crimson Pro", 16 * -1)
            )
            return

        try:
            df_scaled, _ids = load_scaled_dataset(CSV_PATH)
            df_pca = attach_group_pca(df_scaled, random_state=42)
            X = select_X(df_pca)
        except Exception as e:
            self.canvas.create_text(
                360, 120, anchor="nw",
                text=f"[LỖI] Load/chuẩn bị dữ liệu thất bại:\n{e}",
                fill="#b00020", font=("Crimson Pro", 16 * -1)
            )
            return
        for tag in ["image_4.png", "image_5.png", "chart_frame"]:
            try:
                items = self.canvas.find_withtag(tag)
                for i in items:
                    self.canvas.itemconfigure(i, state="hidden")
            except Exception:
                pass

        # Elbow
        fig_elbow = figure_elbow_silhouette(X, k_min=2, k_max=11)
        self._mount(fig_elbow, x=352, y=130, w=450, h=190)

        # kmeans & 2 chart còn lại
        labels = kmeans_labels(X, k=3, random_state=42, n_init=10)

        # --- Cluster Distribution (Pie) ở TRÊN ---
        fig_pie = figure_cluster_distribution(labels, scale=1.3)
        self._mount(fig_pie, x=890, y=210, w=480, h=340)

        # --- PCA Scatter ở DƯỚI ---
        fig_scatter = figure_pca_scatter(X, labels)
        self._mount(fig_scatter, x=850, y=650, w=550, h=340)

        # ----- Top-3 feature theo từng cụm (hiển thị trong 3 ô text) -----
        cols_src = CLUSTER_FEATURES
        use_cols = [c for c in cols_src if c in df_pca.columns]

        Xd = pd.DataFrame(X, columns=use_cols)
        global_mean = Xd.mean()
        Xd["cluster"] = labels
        means_by_cluster = Xd.groupby("cluster")[use_cols].mean()

        top3_per_cluster = {}
        for c in sorted(means_by_cluster.index):
            diffs = (means_by_cluster.loc[c] - global_mean).abs().sort_values(ascending=False)
            top3_per_cluster[c] = list(diffs.index[:3])

        # bullet + xuống dòng, wrap gọn trong card (KHÔNG NGHIÊNG vì dùng self.font_feature)
        fmt = lambda names: "• " + "\n• ".join(names)
        self._set_feature_text_at(0, fmt(top3_per_cluster.get(0, ["—", "—", "—"])))  # Cluster 1
        self._set_feature_text_at(1, fmt(top3_per_cluster.get(1, ["—", "—", "—"])))  # Cluster 2
        self._set_feature_text_at(2, fmt(top3_per_cluster.get(2, ["—", "—", "—"])))  # Cluster 3

        # cập nhật 3 con số card
        c1, c2, c3 = counts_by_cluster(labels, k=3)
        self._set_card_number_at(421.0, 548.0, c1)
        self._set_card_number_at(421.0, 721.0, c2)
        self._set_card_number_at(421.0, 895.0, c3)
        # === Đọc mô tả đặc trưng cụm (định tính) ===
        desc_path = Path(__file__).resolve().parents[
                        2] / "Dataset" / "Output" / "cluster_characteristics_descriptive.csv"
        if desc_path.exists():
            desc_df = pd.read_csv(desc_path)

            show_cols = ["Restaurant Rating", "Family size", "Delivery Rating"]

            for i in range(min(3, len(desc_df))):
                row = desc_df.iloc[i]
                bullets = []

                # --- các đặc trưng định lượng ---
                for col in show_cols:
                    if col not in row:
                        continue
                    val = row[col]
                    if isinstance(val, (int, float)):
                        if "size" in col.lower():
                            val = f"{round(val)} người"
                        else:
                            val = f"{val:.2f}"
                    elif pd.isna(val):
                        val = "-"
                    bullets.append(f"{col}: {val}")

                # --- gộp 3 đặc trưng định tính ---
                convenience = str(row.get("Mức độ coi trọng sự tiện lợi", "")).strip().lower()
                service = str(row.get("Vấn đề dịch vụ", "")).strip().lower()
                offer = str(row.get("Nhạy cảm ưu đãi/đánh giá", "")).strip().lower()

                desc_parts = []
                if "cao" in convenience:
                    desc_parts.append("coi trọng tiện lợi")
                elif "thấp" in convenience:
                    desc_parts.append("ít quan tâm tiện lợi")

                if "cao" in service:
                    desc_parts.append("đánh giá cao dịch vụ")
                elif "thấp" in service:
                    desc_parts.append("không chú trọng dịch vụ")

                if "cao" in offer:
                    desc_parts.append("rất nhạy cảm với ưu đãi/đánh giá")
                elif "thấp" in offer:
                    desc_parts.append("ít bị ảnh hưởng bởi ưu đãi")

                # --- auto rút gọn ---
                if len(desc_parts) > 2:
                    keep = [p for p in desc_parts if "tiện" in p or "ưu đãi" in p]
                    if len(keep) < 2:
                        keep = desc_parts[:2]
                    desc_parts = keep

                desc_text = "• " + ", ".join(desc_parts).capitalize() + "."

                # --- render ---
                txt_id = self.feature_text_ids.get(i)
                if txt_id:
                    # Font thống nhất
                    common_font = tkfont.Font(family="Crimson Pro", size=-20, slant="roman")

                    # --- BULLET LIST ---
                    max_len = max(len(line) for line in bullets)
                    est_width = min(max(260, max_len * 10), 330)

                    self.canvas.itemconfigure(
                        txt_id,
                        text="• " + "\n• ".join(bullets),
                        font=common_font,
                        anchor="nw",
                        justify="left",
                        width=est_width,
                        fill="#000000"
                    )

                    # canh vị trí bullet
                    x0, y0 = self.canvas.coords(txt_id)
                    self.canvas.coords(txt_id, x0 - 60, y0 - 29)

                    # --- MÔ TẢ ---
                    desc_font = tkfont.Font(family="Crimson Pro", size=-19, slant="roman")

                    line_spacing = 22  # giãn cách đều hơn
                    y_desc = y0 - 15 + (len(bullets) * line_spacing) + 2

                    y_desc -= 3  # dịch lên nhẹ cho dòng mô tả gần bullet hơn

                    desc_width = est_width - 20

                    self.canvas.create_text(
                        x0 - 60, y_desc,
                        anchor="nw",
                        text="• " + desc_text.replace("• ", ""),
                        font=desc_font,
                        fill="#000000",  # cùng màu chữ
                        width=desc_width,
                        justify="left"
                    )

        else:
            print("[Frame07] Không tìm thấy file mô tả đặc trưng cụm.")

    def _mount(self, fig, x, y, w, h):
        frm = tk.Frame(self, bg="#D9D9D9", highlightthickness=0, bd=0)
        self.canvas.create_window(x, y, window=frm, anchor="nw", width=w, height=h)
        cv = FigureCanvasTkAgg(fig, master=frm)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True)

    def _set_card_number_at(self, x, y, value):
        """Cập nhật con số card tại đúng vị trí (x, y) nếu có ID đã lưu."""
        ref_positions = [(421.0, 548.0), (421.0, 721.0), (421.0, 895.0)]
        for i, (x_ref, y_ref) in enumerate(ref_positions):
            if abs(x - x_ref) < 5 and abs(y - y_ref) < 5:
                item_id = self.card_number_ids.get(i)
                if item_id:
                    self.canvas.itemconfigure(item_id, text=str(value), fill="#2E126A")
                return

    def _set_feature_text_at(self, idx: int, text: str):
        """Cập nhật text theo index cụm (0/1/2) đã lưu sẵn ID."""
        item_id = self.feature_text_ids.get(idx)
        if not item_id:
            return
        self.canvas.itemconfigure(
            item_id,
            text=str(text),
            fill="#000000",
            font=self.font_feature,  # dùng font KHÔNG NGHIÊNG
            width=230,
            justify="left"
        )

    def on_show(self, **kwargs):
        pass

# ---- Test nhanh độc lập (chạy file này trực tiếp) ----
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Frame07")
    root.geometry("1440x1024")
    root.configure(bg="#D9D9D9")

    f = Frame07(root)
    f.place(x=0, y=0, width=1440, height=1024)

    root.resizable(False, False)
    root.mainloop()
