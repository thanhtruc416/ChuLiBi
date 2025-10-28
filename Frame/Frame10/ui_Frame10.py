# ui_Frame10.py — Recommendations Frame (class-based)

from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, Button, PhotoImage, Frame, messagebox
import pandas as pd

# Import table builder
try:
    from .ui_content_Frame10 import build_table
except ImportError:
    from ui_content_Frame10 import build_table

# Import recommendation engine
try:
    from Function.Frame10_Recommend import get_recommendations
except ImportError:
    get_recommendations = None

# Import dropdown for profile button
try:
    from Function.dropdown_profile import DropdownMenu
except ImportError:
    DropdownMenu = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_Frame10")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Frame10(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller
        self._imgs = {}  # Store images to prevent garbage collection
        self.configure(bg="#ECE7EB")

        # Load recommendation data
        self.df_recommendations = None
        self.filtered_data = None
        self._load_recommendations()

        # Main Canvas
        canvas = Canvas(
            self, bg="#ECE7EB", height=1024, width=1440,
            bd=0, highlightthickness=0, relief="ridge"
        )
        canvas.place(x=0, y=0)
        self.canvas = canvas

        # ---- Top bar ----
        self._imgs["image_1"] = PhotoImage(file=relative_to_assets("image_1.png"))
        canvas.create_image(890.0, 42.0, image=self._imgs["image_1"])

        canvas.create_text(
            385.0, 14.0, anchor="nw",
            text="Recommendations",
            fill="#000000", font=("Young Serif", 40 * -1),
        )

        # Profile button with dropdown
        if DropdownMenu:
            self._imgs["button_Profile"] = PhotoImage(file=relative_to_assets("button_Profile.png"))
            self.dropdown = DropdownMenu(self)
            self.button_Profile = Button(
                self,
                image=self._imgs["button_Profile"],
                borderwidth=0, highlightthickness=0,
                command=self.dropdown.show,
                relief="flat"
            )
            self.button_Profile.place(x=1361.18, y=17.03, width=44.18, height=44.69)
        else:
            self._imgs["button_Profile"] = PhotoImage(file=relative_to_assets("button_Profile.png"))
            self.button_Profile = Button(
                self,
                image=self._imgs["button_Profile"],
                borderwidth=0, highlightthickness=0,
                command=lambda: print("Profile"),
                relief="flat"
            )
            self.button_Profile.place(x=1361.18, y=17.03, width=44.18, height=44.69)

        # Notification button
        self._imgs["button_Noti"] = PhotoImage(file=relative_to_assets("button_Noti.png"))
        self.button_Noti = Button(
            self,
            image=self._imgs["button_Noti"],
            borderwidth=0, highlightthickness=0,
            command=lambda: print("Notification"),
            relief="flat"
        )
        self.button_Noti.place(x=1292.0, y=17.0, width=49.0, height=49.0)

        # ---- Search block ----
        canvas.create_text(
            389.0, 144.0, anchor="nw",
            text="Customer research",
            fill="#374A5A", font=("Young Serif", 24 * -1),
        )

        self._imgs["image_2"] = PhotoImage(file=relative_to_assets("image_2.png"))
        canvas.create_image(832.0, 215.0, image=self._imgs["image_2"])

        self._imgs["button_Search"] = PhotoImage(file=relative_to_assets("button_Search.png"))
        self.button_Search = Button(
            self,
            image=self._imgs["button_Search"],
            borderwidth=0, highlightthickness=0,
            command=self.search_action,
            relief="flat"
        )
        self.button_Search.place(x=1287.0, y=193.0, width=102.0, height=45.0)

        self._imgs["entry_1"] = PhotoImage(file=relative_to_assets("entry_1.png"))
        canvas.create_image(823.0, 217.0, image=self._imgs["entry_1"])
        self.entry_search = Entry(
            self, bd=0, bg="#FFFFFF", fg="#000716",
            highlightthickness=0, font=("Crimson Pro Regular", 16 * -1)
        )
        self.entry_search.place(x=400.0, y=206.0, width=846.0, height=20.0)

        # Bind Enter key to search
        self.entry_search.bind("<Return>", lambda e: self.search_action())
        self.entry_search.bind("<KP_Enter>", lambda e: self.search_action())  # Numpad Enter

        # ---- White card containing table ----
        canvas.create_rectangle(359.0, 298.0, 1419.0, 1001.0, fill="#FFFFFF", outline="")
        canvas.create_text(
            389.0, 328.0, anchor="nw",
            text="Recommendations Table",
            fill="#374A5A", font=("Young Serif", 24 * -1),
        )

        # ==== TABLE WITH SCROLLBAR (only inside card) ====
        # Table dimensions and position inside card
        TABLE_X = 389  # left edge of table display area
        TABLE_Y = 366  # below title
        TABLE_W = 1389 - 389
        TABLE_H = 1001 - 366

        self.table_holder = Frame(self, bg="#FFFFFF")
        self.table_holder.place(x=TABLE_X, y=TABLE_Y, width=TABLE_W, height=TABLE_H)

        # Build table with recommendation data
        display_data = self.filtered_data if self.filtered_data is not None else self.df_recommendations
        self.table_canvas = build_table(self.table_holder, TABLE_W, TABLE_H, data=display_data, search_term=None)
        self.table_canvas.pack(fill="both", expand=True)

        # Store current search term
        self.current_search = None

        # ---- Sidebar & mascot ----
        # Sidebar background gradient
        self._imgs["image_sidebar_bg"] = PhotoImage(file=relative_to_assets("image_sidebar_bg.png"))
        canvas.create_image(168.0, 512.0, image=self._imgs["image_sidebar_bg"])

        # Logo text
        canvas.create_text(
            92.0, 942.0, anchor="nw",
            text="ChuLiBi", fill="#FDE5F4",
            font=("Rubik Burned Regular", 35 * -1)
        )

        # Logo icon
        self._imgs["image_5"] = PhotoImage(file=relative_to_assets("image_5.png"))
        canvas.create_image(169.0, 103.0, image=self._imgs["image_5"])

        # ---- Sidebar navigation buttons ----
        self._make_button("button_dashboard.png", cmd=lambda: self.controller.show_frame("Frame06"), x=0.0, y=198.0,
                          w=335.0, h=97.0)
        self._make_button("button_customer.png", cmd=lambda: self.controller.show_frame("Frame07"), x=0.0, y=289.0,
                          w=335.0, h=90.0)
        self._make_button("button_churn.png", cmd=lambda: self.controller.show_frame("Frame08"), x=0.0, y=378.0,
                          w=335.0, h=92.0)
        self._make_button("button_recommend_1.png", cmd=lambda: print("Recommend (current page)"), x=0.0, y=543.0,
                          w=335.0, h=98.0)  # Current page
        self._make_button("button_delivery.png", cmd=lambda: self.controller.show_frame("Frame09"), x=0.0, y=464.0,
                          w=335.0, h=89.0)
        self._make_button("button_report.png", cmd=lambda: print("Report"), x=0.0, y=634.0, w=335.0, h=96.0)

    def _make_button(self, filename, cmd, x, y, w, h):
        """Helper method to create buttons with images"""
        if filename not in self._imgs:
            try:
                self._imgs[filename] = PhotoImage(file=relative_to_assets(filename))
            except Exception as e:
                print(f"[Frame10] Could not load button image {filename}: {e}")
                return None

        btn = Button(
            self,
            image=self._imgs[filename],
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            command=cmd
        )
        btn.place(x=x, y=y, width=w, height=h)
        return btn

    def _load_recommendations(self):
        """Load recommendation data from file or generate new"""
        try:
            if get_recommendations:
                # Try to get recommendations
                # Path: Frame/Frame10/ui_Frame10.py -> Frame/Frame10 -> Frame -> ChuLiBi -> Dataset/Output
                data_path = Path(__file__).parent.parent.parent / "Dataset" / "Output"
                print(f"📂 Looking for data in: {data_path}")

                if not data_path.exists():
                    print(f"⚠️ Data path does not exist: {data_path}")
                    self.df_recommendations = None
                    self.filtered_data = None
                    return

                self.df_recommendations = get_recommendations(data_path)
                self.filtered_data = self.df_recommendations.copy()
                print(f"✅ Loaded {len(self.df_recommendations)} recommendations")
            else:
                print("⚠️ Recommendation engine not available - using sample data")
                self.df_recommendations = None
                self.filtered_data = None
        except Exception as e:
            print(f"⚠️ Error loading recommendations: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showwarning(
                "Data Loading",
                f"Could not load recommendations:\n{str(e)}\n\nUsing sample data instead."
            )
            self.df_recommendations = None
            self.filtered_data = None

    def search_action(self):
        """Handle search action - filter by Customer_ID"""
        search_text = self.entry_search.get().strip()

        if not search_text:
            # Reset to show all data
            self.current_search = None
            self.filtered_data = self.df_recommendations.copy() if self.df_recommendations is not None else None
            self.refresh_table()
            print("🔄 Showing all customers")
            return

        if self.df_recommendations is None or self.df_recommendations.empty:
            messagebox.showinfo("Search", "No data available to search")
            return

        # Store search term for highlighting
        self.current_search = search_text

        # Filter by Customer_ID (case-insensitive partial match)
        search_lower = search_text.lower()
        mask = self.df_recommendations["Customer_ID"].astype(str).str.lower().str.contains(search_lower, na=False,
                                                                                           regex=False)

        self.filtered_data = self.df_recommendations[mask].copy()

        if self.filtered_data.empty:
            messagebox.showinfo(
                "Search Result",
                f"No customers found with ID containing '{search_text}'\n\nShowing all data."
            )
            self.current_search = None
            self.filtered_data = self.df_recommendations.copy()
        else:
            count = len(self.filtered_data)
            print(f"🔍 Found {count} customer(s) matching '{search_text}'")

        self.refresh_table()

    def refresh_table(self):
        """Refresh the table with current filtered data"""
        # Clear current table
        for widget in self.table_holder.winfo_children():
            widget.destroy()

        # Rebuild table with new data and search term for highlighting
        TABLE_W = 1389 - 389
        TABLE_H = 1001 - 366
        display_data = self.filtered_data if self.filtered_data is not None else self.df_recommendations
        search_term = self.current_search if hasattr(self, 'current_search') else None
        self.table_canvas = build_table(self.table_holder, TABLE_W, TABLE_H, data=display_data, search_term=search_term)
        self.table_canvas.pack(fill="both", expand=True)

    def on_show(self):
        """Called when Frame10 is displayed"""
        print("Frame10 (Recommendations) displayed")
        # Refresh data when shown
        if self.df_recommendations is None:
            self._load_recommendations()
            if self.df_recommendations is not None:
                self.refresh_table()
        # Clear search field and reset filter
        self.entry_search.delete(0, tk.END)
        self.current_search = None
        self.filtered_data = self.df_recommendations.copy() if self.df_recommendations is not None else None


# -------------------------
# Standalone preview runner
# -------------------------
if __name__ == "__main__":
    import sys


    # Dummy controller for testing
    class _DummyController:
        def show_frame(self, frame_name):
            print(f"Navigate to: {frame_name}")


    root = tk.Tk()
    root.title("Recommendations - Frame10")
    root.geometry("1440x1024")
    root.configure(bg="#ECE7EB")

    app = Frame10(root, _DummyController())
    app.pack(fill="both", expand=True)

    root.resizable(False, False)
    root.mainloop()
