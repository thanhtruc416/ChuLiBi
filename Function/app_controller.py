# app_controller.py
import tkinter as tk

class AppController:
    """
    Controller quản lý các Frame:
    - register_frame(name, frame): đăng ký và đặt frame vào container
    - show_frame(name, **kwargs): đưa frame lên và gọi on_show(**kwargs) nếu có
    - get_frame(name): lấy frame theo tên
    """
    def __init__(self, root: tk.Tk, container: tk.Frame | None = None):
        self.root = root
        self.container = container or root    # cho phép pass container riêng
        self.frames: dict[str, tk.Frame] = {}
        self.last_kwargs: dict[str, dict] = {}  # lưu tham số lần cuối mỗi frame
        self.current_user = {}
        # bảo đảm container có layout
        if isinstance(self.container, tk.Frame):
            self.container.grid_rowconfigure(0, weight=1)
            self.container.grid_columnconfigure(0, weight=1)

    def register_frame(self, frame_name: str, frame: tk.Frame):
        """Đăng ký frame và đặt vào container để có thể tkraise()."""
        if frame_name in self.frames:
            # thay thế frame cũ nếu trùng tên
            self.frames[frame_name].destroy()
        self.frames[frame_name] = frame

        # nếu container là grid thì grid, còn lại dùng place fill
        try:
            frame.grid(row=0, column=0, sticky="nsew")
        except Exception:
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def get_frame(self, frame_name: str) -> tk.Frame:
        try:
            return self.frames[frame_name]
        except KeyError as e:
            raise KeyError(f"Frame '{frame_name}' chưa được register") from e

    def show_frame(self, frame_name: str, **kwargs):
        """
        Hiển thị frame theo tên.
        - Gọi on_show(**kwargs) nếu frame có
        - Lưu lại kwargs lần cuối của frame
        """
        frame = self.get_frame(frame_name)

        # truyền dữ liệu trước khi hiển thị
        if hasattr(frame, "on_show"):
            frame.on_show(**kwargs)

        # lưu lại tham số gần nhất (hữu ích khi muốn refresh)
        self.last_kwargs[frame_name] = dict(kwargs)

        # đưa frame lên trên
        frame.tkraise()

    # tiện ích: hiển thị lại frame với tham số trước đó
    def refresh_frame(self, frame_name: str):
        frame = self.get_frame(frame_name)
        if hasattr(frame, "on_show"):
            kwargs = self.last_kwargs.get(frame_name, {})
            frame.on_show(**kwargs)
        frame.tkraise()

    # =========================
    # Quản lý người dùng hiện tại
    # =========================
    def get_current_user(self):
        """Trả về thông tin người dùng hiện tại."""
        return getattr(self, "current_user", {})

    def set_current_user(self, user_data: dict):
        """Lưu thông tin người dùng hiện tại."""
        self.current_user = user_data or {}

    def clear_current_user(self):
        """Xóa thông tin người dùng hiện tại."""
        self.current_user = {}
