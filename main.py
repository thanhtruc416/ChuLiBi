# -*- coding: utf-8 -*-
# File: main.py
# Hợp nhất phiên bản Frame02_ex + Frame12_ChangeInformation
# Giữ đầy đủ tất cả Frame (01–12)

import tkinter as tk
from Function.app_controller import AppController
from dotenv import load_dotenv
load_dotenv()

# ==== Import các Frame ====
from Frame.Frame01.ui_Frame01 import Frame01
from Frame.Frame02.ui_Frame02 import Frame02
from Frame.Frame02_ex.ui_Frame02_ex import Frame02_ex
from Frame.Frame03.ui_Frame03 import Frame03
from Frame.Frame04.ui_Frame04 import Frame04
from Frame.Frame05.ui_Frame05 import Frame05
from Frame.Frame06.ui_Frame06 import Frame06
from Frame.Frame07.ui_Frame07 import Frame07
from Frame.Frame08.ui_Frame08 import Frame08
from Function.Frame09_EL import Frame09_EL
from Frame.Frame10.ui_Frame10 import Frame10
from Frame.Frame11.ui_Frame11 import Frame11
from Frame.Frame12.ui_Frame12 import Frame12




class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Khởi tạo controller ---
        self.controller = AppController(self)

        # --- Cấu hình cửa sổ chính ---
        self.geometry("1440x1024")
        self.title("ChuLiBi")
        self.resizable(False, False)

        # --- Căn giữa màn hình ---
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1440) // 2
        y = (screen_height - 1024) // 2
        self.geometry(f"1440x1024+{x}+{y}")

        # --- Container chứa tất cả frame ---
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # --- Khởi tạo và lưu trữ các frame ---
        self.frames = {}
        frame_classes = [
            Frame01,
            Frame02,
            Frame02_ex,
            Frame03,
            Frame04,
            Frame05,
            Frame06,
            Frame07,
            Frame08,
            Frame09_EL,
            Frame10,
            Frame11,
            Frame12
        ]

        # Ẩn toàn bộ frame sau khi khởi tạo
        for F in frame_classes:
            frame = F(parent=container, controller=self.controller)
            self.frames[F.__name__] = frame
            self.controller.register_frame(F.__name__, frame)

        # Hiển thị Frame01 làm trang chủ
        # Bảo đảm Frame01 đã được đăng ký rồi mới show
        if "Frame01" in self.frames:
            self.frames["Frame01"].grid()
            self.controller.show_frame("Frame01")
        else:
            print("[ERROR] Frame01 chưa được register!")


if __name__ == "__main__":
    app = Main()
    app.mainloop()
