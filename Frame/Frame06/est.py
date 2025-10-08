# main.py
import tkinter as tk
from Frame.Frame06.ui_Frame06 import Frame06

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChuLiBi Dashboard")
        self.geometry("1440x1024")
        self.resizable(False, False)

        # --- Frame container ---
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        frame = Frame06(container, self)
        self.frames["Frame06"] = frame
        frame.pack(fill="both", expand=True)

        self.show_frame("Frame06")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()
