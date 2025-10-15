import tkinter as tk
from Function.app_controller import AppController
from dotenv import load_dotenv
load_dotenv()
from Frame.Frame01.ui_Frame01 import Frame01
from Frame.Frame02.ui_Frame02 import Frame02
from Frame.Frame03.ui_Frame03 import Frame03
from Frame.Frame04.ui_Frame04 import Frame04
from Frame.Frame05.ui_Frame05 import Frame05
from Frame.Frame06.ui_Frame06 import Frame06
from Frame.Frame07.ui_Frame07 import Frame07
# Import các frame khác nếu bạn có, ví dụ:
# from Frame.Frame02.ui_Frame02 import Frame02
# from Frame.Frame03.ui_Frame03 import Frame03
# ...

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
            Frame03,
            Frame04,
            Frame05,
            Frame06,
            Frame07
            # Thêm các frame khác ở đây: Frame02, Frame03, ...
        ]

        for F in frame_classes:
            frame = F(parent=container, controller=self.controller)
            self.frames[F.__name__] = frame
            self.controller.register_frame(F.__name__, frame)
            frame.grid(row=0, column=0, sticky="nsew")

        # --- Hiển thị Frame01 làm trang chủ ---
        self.controller.show_frame("Frame01")


if __name__ == "__main__":
    app = Main()
    app.mainloop()
