class AppController:
    def __init__(self, root):
        self.root = root
        self.frames = {}

    def register_frame(self, frame_name, frame):
        """Lưu frame vào dictionary"""
        self.frames[frame_name] = frame

    def show_frame(self, frame_name):
        """Hiển thị frame theo tên"""
        frame = self.frames.get(frame_name)
        if frame:
            frame.tkraise()
            # Gọi on_show nếu frame có
            if hasattr(frame, "on_show"):
                frame.on_show()
