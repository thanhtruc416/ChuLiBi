

from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH /Path("assets_Frame12")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1440x1024")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 1024,
    width = 1440,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    891.0,
    554.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    890.0,
    554.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    889.0,
    329.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    889.0,
    795.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    929.0,
    1493.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    889.0,
    153.0,
    image=image_image_6
)

canvas.create_text(
    423.0,
    130.0,
    anchor="nw",
    text="Change Information",
    fill="#000000",
    font=("Young Serif", 35 * -1)
)

entry_image_FullName = PhotoImage(
    file=relative_to_assets("entry_FullName.png"))
entry_bg_FullName = canvas.create_image(
    558.0,
    295.5,
    image=entry_image_FullName
)
entry_FullName = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_FullName.place(
    x=431.0,
    y=272.0,
    width=254.0,
    height=45.0
)

entry_image_Username = PhotoImage(
    file=relative_to_assets("entry_Username.png"))
entry_bg_Username = canvas.create_image(
    558.0,
    417.5,
    image=entry_image_Username
)
entry_Username = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_Username.place(
    x=431.0,
    y=394.0,
    width=254.0,
    height=45.0
)

entry_image_Business_Name = PhotoImage(
    file=relative_to_assets("entry_Business_Name.png"))
entry_bg_Business_Name = canvas.create_image(
    890.0,
    295.5,
    image=entry_image_Business_Name
)
entry_Business_Name = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_Business_Name.place(
    x=763.0,
    y=272.0,
    width=254.0,
    height=45.0
)

entry_image_Password = PhotoImage(
    file=relative_to_assets("entry_Password.png"))
entry_bg_Password = canvas.create_image(
    890.0,
    417.5,
    image=entry_image_Password
)
entry_Password = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_Password.place(
    x=763.0,
    y=394.0,
    width=254.0,
    height=45.0
)

entry_image_Your_Role = PhotoImage(
    file=relative_to_assets("entry_Your_Role.png"))
entry_bg_Your_Role = canvas.create_image(
    1221.0,
    295.5,
    image=entry_image_Your_Role
)
entry_Your_Role = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_Your_Role.place(
    x=1094.0,
    y=272.0,
    width=254.0,
    height=45.0
)

entry_image_Gmail = PhotoImage(
    file=relative_to_assets("entry_Gmail.png"))
entry_bg_Gmail = canvas.create_image(
    1221.0,
    417.5,
    image=entry_image_Gmail
)
entry_Gmail = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font=("Crimson Pro Regular", 24 * -1)
)
entry_Gmail.place(
    x=1094.0,
    y=394.0,
    width=254.0,
    height=45.0
)

canvas.create_text(
    466.0,
    238.0,
    anchor="nw",
    text="FullName",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

canvas.create_text(
    466.0,
    362.0,
    anchor="nw",
    text="Username",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    441.0,
    254.0,
    image=image_image_7
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    441.0,
    378.0,
    image=image_image_8
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    442.0,
    236.0,
    image=image_image_9
)

image_image_10 = PhotoImage(
    file=relative_to_assets("image_10.png"))
image_10 = canvas.create_image(
    442.0,
    360.0,
    image=image_image_10
)

canvas.create_text(
    798.0,
    237.0,
    anchor="nw",
    text="Business Name",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

image_image_11 = PhotoImage(
    file=relative_to_assets("image_11.png"))
image_11 = canvas.create_image(
    773.0,
    252.0,
    image=image_image_11
)

image_image_12 = PhotoImage(
    file=relative_to_assets("image_12.png"))
image_12 = canvas.create_image(
    773.0,
    254.0,
    image=image_image_12
)

image_image_13 = PhotoImage(
    file=relative_to_assets("image_13.png"))
image_13 = canvas.create_image(
    773.0,
    243.0,
    image=image_image_13
)

canvas.create_text(
    1136.0,
    240.0,
    anchor="nw",
    text="Your Role",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

canvas.create_text(
    1123.0,
    362.0,
    anchor="nw",
    text="Gmail",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

image_image_14 = PhotoImage(
    file=relative_to_assets("image_14.png"))
image_14 = canvas.create_image(
    1106.8125,
    242.0,
    image=image_image_14
)

image_image_15 = PhotoImage(
    file=relative_to_assets("image_15.png"))
image_15 = canvas.create_image(
    1107.0,
    255.5,
    image=image_image_15
)

image_image_16 = PhotoImage(
    file=relative_to_assets("image_16.png"))
image_16 = canvas.create_image(
    1119.75,
    247.90625,
    image=image_image_16
)

canvas.create_text(
    799.0,
    363.0,
    anchor="nw",
    text="Password",
    fill="#000000",
    font=("Crimson Pro SemiBold", 28 * -1)
)

image_image_17 = PhotoImage(
    file=relative_to_assets("image_17.png"))
image_17 = canvas.create_image(
    775.0,
    377.0,
    image=image_image_17
)

image_image_18 = PhotoImage(
    file=relative_to_assets("image_18.png"))
image_18 = canvas.create_image(
    775.0,
    366.0,
    image=image_image_18
)

image_image_19 = PhotoImage(
    file=relative_to_assets("image_19.png"))
image_19 = canvas.create_image(
    1104.0,
    371.0,
    image=image_image_19
)

image_image_20 = PhotoImage(
    file=relative_to_assets("image_20.png"))
image_20 = canvas.create_image(
    1104.0,
    374.0,
    image=image_image_20
)

image_image_21 = PhotoImage(
    file=relative_to_assets("image_21.png"))
image_21 = canvas.create_image(
    889.0,
    619.0,
    image=image_image_21
)

canvas.create_text(
    423.0,
    598.0,
    anchor="nw",
    text="Our Team",
    fill="#000000",
    font=("Young Serif", 35 * -1)
)

button_image_save = PhotoImage(
    file=relative_to_assets("button_save.png"))
button_save = Button(
    image=button_image_save,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_save clicked"),
    relief="flat"
)
button_save.place(
    x=1226.0,
    y=458.0,
    width=137.0,
    height=54.0
)

image_image_22 = PhotoImage(
    file=relative_to_assets("image_22.png"))
image_22 = canvas.create_image(
    1091.0,
    822.0,
    image=image_image_22
)

image_image_23 = PhotoImage(
    file=relative_to_assets("image_23.png"))
image_23 = canvas.create_image(
    1089.86767578125,
    741.0,
    image=image_image_23
)

canvas.create_text(
    1045.0,
    785.0,
    anchor="nw",
    text="John Đỗ",
    fill="#706093",
    font=("Crimson Pro SemiBold", 24 * -1)
)

image_image_24 = PhotoImage(
    file=relative_to_assets("image_24.png"))
image_24 = canvas.create_image(
    1290.0,
    819.0,
    image=image_image_24
)

image_image_25 = PhotoImage(
    file=relative_to_assets("image_25.png"))
image_25 = canvas.create_image(
    1289.0,
    744.0,
    image=image_image_25
)

canvas.create_text(
    1246.0,
    785.0,
    anchor="nw",
    text="US Link",
    fill="#706093",
    font=("Crimson Pro SemiBold", 24 * -1)
)

image_image_26 = PhotoImage(
    file=relative_to_assets("image_26.png"))
image_26 = canvas.create_image(
    892.0,
    822.0,
    image=image_image_26
)

image_image_27 = PhotoImage(
    file=relative_to_assets("image_27.png"))
image_27 = canvas.create_image(
    889.0,
    744.0,
    image=image_image_27
)

canvas.create_text(
    837.0,
    785.0,
    anchor="nw",
    text="quàng uk",
    fill="#706093",
    font=("Crimson Pro SemiBold", 24 * -1)
)

canvas.create_text(
    872.0,
    816.0,
    anchor="nw",
    text="hello",
    fill="#B992B9",
    font=("Crimson Pro Regular", 15 * -1)
)

image_image_28 = PhotoImage(
    file=relative_to_assets("image_28.png"))
image_28 = canvas.create_image(
    688.0,
    824.0,
    image=image_image_28
)

image_image_29 = PhotoImage(
    file=relative_to_assets("image_29.png"))
image_29 = canvas.create_image(
    687.0,
    744.0,
    image=image_image_29
)

canvas.create_text(
    605.0,
    785.0,
    anchor="nw",
    text="Quỳnh Cheese",
    fill="#706093",
    font=("Crimson Pro SemiBold", 24 * -1)
)

canvas.create_text(
    645.0,
    816.0,
    anchor="nw",
    text="Data Analyst",
    fill="#B992B9",
    font=("Crimson Pro Regular", 15 * -1)
)

image_image_30 = PhotoImage(
    file=relative_to_assets("image_30.png"))
image_30 = canvas.create_image(
    492.0,
    822.0,
    image=image_image_30
)

image_image_31 = PhotoImage(
    file=relative_to_assets("image_31.png"))
image_31 = canvas.create_image(
    491.0,
    744.0,
    image=image_image_31
)

canvas.create_text(
    408.0,
    785.0,
    anchor="nw",
    text="Thank Bamboo",
    fill="#706093",
    font=("Crimson Pro SemiBold", 24 * -1)
)

canvas.create_text(
    467.0,
    816.0,
    anchor="nw",
    text="Leader",
    fill="#B992B9",
    font=("Crimson Pro Regular", 15 * -1)
)

canvas.create_text(
    448.0,
    841.0,
    anchor="nw",
    text="Generate and ",
    fill="#FFFFFF",
    font=("Crimson Pro Regular", 15 * -1)
)

canvas.create_text(
    438.0,
    858.0,
    anchor="nw",
    text="export analytics ",
    fill="#FFFFFF",
    font=("Crimson Pro Regular", 15 * -1)
)

canvas.create_text(
    468.0,
    875.0,
    anchor="nw",
    text="reports",
    fill="#FFFFFF",
    font=("Crimson Pro Regular", 15 * -1)
)

image_image_32 = PhotoImage(
    file=relative_to_assets("image_32.png"))
image_32 = canvas.create_image(
    172.0,
    512.0,
    image=image_image_32
)

canvas.create_text(
    103.0,
    927.0,
    anchor="nw",
    text="ChuLiBi",
    fill="#FDE5F4",
    font=("Rubik Burned Regular", 35 * -1)
)

image_image_33 = PhotoImage(
    file=relative_to_assets("image_33.png"))
image_33 = canvas.create_image(
    167.0,
    101.0,
    image=image_image_33
)

button_image_Customer_Analysis = PhotoImage(
    file=relative_to_assets("button_Customer_Analysis.png"))
button_Customer_Analysis = Button(
    image=button_image_Customer_Analysis,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Customer_Analysis clicked"),
    relief="flat"
)
button_Customer_Analysis.place(
    x=5.0,
    y=302.0,
    width=337.0,
    height=77.0
)

button_image_Recommendation = PhotoImage(
    file=relative_to_assets("button_Recommendation.png"))
button_Recommendation = Button(
    image=button_image_Recommendation,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Recommendation clicked"),
    relief="flat"
)
button_Recommendation.place(
    x=5.0,
    y=468.0,
    width=336.0,
    height=82.0
)

button_image_Churn = PhotoImage(
    file=relative_to_assets("button_Churn.png"))
button_Churn = Button(
    image=button_image_Churn,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Churn clicked"),
    relief="flat"
)
button_Churn.place(
    x=5.0,
    y=381.0,
    width=336.0,
    height=86.0
)

button_image_Delivery = PhotoImage(
    file=relative_to_assets("button_Delivery.png"))
button_Delivery = Button(
    image=button_image_Delivery,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Delivery clicked"),
    relief="flat"
)
button_Delivery.place(
    x=5.0,
    y=552.0,
    width=337.0,
    height=90.0
)

button_image_Report = PhotoImage(
    file=relative_to_assets("button_Report.png"))
button_Report = Button(
    image=button_image_Report,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Report clicked"),
    relief="flat"
)
button_Report.place(
    x=5.0,
    y=646.0,
    width=338.0,
    height=88.0
)

button_image_Dashboard = PhotoImage(
    file=relative_to_assets("button_Dashboard.png"))
button_Dashboard = Button(
    image=button_image_Dashboard,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Dashboard clicked"),
    relief="flat"
)
button_Dashboard.place(
    x=6.0,
    y=226.0,
    width=337.0,
    height=64.0
)

image_image_34 = PhotoImage(
    file=relative_to_assets("image_34.png"))
image_34 = canvas.create_image(
    890.0,
    44.0,
    image=image_image_34
)

image_image_35 = PhotoImage(
    file=relative_to_assets("image_35.png"))
image_35 = canvas.create_image(
    1320.0,
    38.0,
    image=image_image_35
)

button_image_Profile = PhotoImage(
    file=relative_to_assets("button_Profile.png"))
button_Profile = Button(
    image=button_image_Profile,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_Profile clicked"),
    relief="flat"
)
button_Profile.place(
    x=1361.0,
    y=16.0,
    width=44.0,
    height=45.0
)
window.resizable(False, False)
window.mainloop()
