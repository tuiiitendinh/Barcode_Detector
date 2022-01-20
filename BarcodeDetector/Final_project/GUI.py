from tkinter import *
from ttkbootstrap import Style
from tkinter import ttk
from Funct import *
#
# Window and Style
#
window = Tk()
window = Style(theme='cosmo').master
window.title("DarkSouls's App")
window.geometry("880x620")
window.resizable(False, False)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
##
# Label Frame
##
##
BD_LFrame = ttk.LabelFrame(window, style='TLabelframe',text="Nhận dạng barcode")
SL_LFrame = ttk.LabelFrame(window, style='TLabelframe', text="Chỉnh độ sáng")
SS_LFrame = ttk.LabelFrame(window, style='TLabelframe', text="Chỉnh độ nét")
DB_LFrame = ttk.LabelFrame(window, style='TLabelframe', text="Xóa background")
RN_LFrame = ttk.LabelFrame(window, style='TLabelframe', text="Lọc nhiễu")
DE_LFrame = ttk.LabelFrame(window, style='TLabelframe', text="Nhận dạng cạnh")
RC_LFrame = ttk.LabelFrame(window, style='TLabelframe',
                           text="Xoay ảnh đúng vị trí")
##
# MenuButton
##
option_var = StringVar()
##########################Nhận dạng bar code#############################
BDetect_MButton = ttk.Menubutton(BD_LFrame,text="Choose the medthod",style='TMenubutton')
BDetect_Menu = Menu(BDetect_MButton)
BDetect_labels = ['Image','Camera']
BDetect_command = [
    lambda: bcDetect_Image(canvas),
    lambda: bcDetect_Camera(BD_LFrame,window)
]
for option, command in zip(BDetect_labels, BDetect_command):
    BDetect_Menu.add_radiobutton(
        label=option, command=command, variable=option_var)
    BDetect_Menu.add_separator()
BDetect_MButton["menu"] = BDetect_Menu

##########################Chỉnh độ sáng#############################
SLight_MButton = ttk.Menubutton(
    SL_LFrame, text="Choose the method", style='TMenubutton')
SLight_Menu = Menu(SLight_MButton)
SLight_labels = ['Histogram equalization', 'Gamma', 'LoG', 'Power']
SLight_command = [
    lambda: Histogram(canvas),
    lambda: Gamma_Slide(SL_LFrame, canvas),
    lambda: Log_Slide(SL_LFrame, canvas),
    lambda: Power_Slide(SL_LFrame, canvas)]
for option, command in zip(SLight_labels, SLight_command):
    SLight_Menu.add_radiobutton(
        label=option, command=command, variable=option_var)
    SLight_Menu.add_separator()
SLight_MButton["menu"] = SLight_Menu

###############################Chỉnh độ nét#############################
SSmooth_MButton = ttk.Menubutton(
    SS_LFrame, text="Choose the method", style='TMenubutton')
SSmooth_Menu = Menu(SSmooth_MButton)
for option in ['Quantization']:
    SSmooth_Menu.add_radiobutton(
        label=option, value=option, variable=option_var, command=lambda: Quanz_Slide(SS_LFrame, canvas))
    SSmooth_Menu.add_separator()
SSmooth_MButton["menu"] = SSmooth_Menu

###############################Xóa Background#############################
DBackground_MButton = ttk.Menubutton(
    DB_LFrame, text="Choose the method", style='TMenubutton')
DBackground_Menu = Menu(DBackground_MButton)
for option in ['Subtracting background']:
    DBackground_Menu.add_radiobutton(
        label=option, value=option, variable=option_var, command=lambda: Sub_bg_Slide(DB_LFrame, canvas))
    DBackground_Menu.add_separator()
DBackground_MButton["menu"] = DBackground_Menu

###############################Lọc nhiễu#############################
RNoise_MButton = ttk.Menubutton(
    RN_LFrame, text="Choose the method", style='TMenubutton')
RNoise_Menu = Menu(RNoise_MButton)
RNoise_labels = ['Directional fillter',
                 'Threshold median fillter', 'Gaussian Blur']
RNoise_command = [
    lambda: Direct_Slide(RN_LFrame, canvas),
    lambda: Median_Slide(RN_LFrame, canvas),
    lambda: Gaussian_Slide(RN_LFrame, canvas)]
for option, command in zip(RNoise_labels, RNoise_command):
    RNoise_Menu.add_radiobutton(
        label=option, value=option, variable=option_var, command=command)
    RNoise_Menu.add_separator()
RNoise_MButton["menu"] = RNoise_Menu

###############################Nhận dạng cạnh#############################
DEdge_MButton = ttk.Menubutton(
    DE_LFrame, text="Choose the method", style='TMenubutton')
DEdge_Menu = Menu(DEdge_MButton)
DEdge_labels = ['Zero-Crossing of Laplcian',
                'Canny', 'Hough transform']
DEdge_command = [lambda: ZeroCross(canvas),
                 lambda: Canny_Slide(DE_LFrame, canvas),
                 lambda: HoughTransform(canvas)]
for option, command in zip(DEdge_labels, DEdge_command):
    DEdge_Menu.add_radiobutton(label=option, value=option,
                               variable=option_var, command=command)
    DEdge_Menu.add_separator()
DEdge_MButton["menu"] = DEdge_Menu

##
# Board
# 002b36
img_logo = PhotoImage(file='.\DarkSouls_logo.png')
canvas = Canvas(window, height=599, width=640, bg="#98D6EA")
canvas.create_image(225, 160, image=img_logo, anchor=NW)
##
# Button
##
img_open = PhotoImage(file='.\icon\open_img.png').subsample(8, 8)
img_ANoise = PhotoImage(file='.\icon\Add_noise_icon.png').subsample(8, 8)
img_save = PhotoImage(file='.\icon\save_img.png').subsample(8, 8)
img_Barcode = PhotoImage(file='.\icon\\barcode.png').subsample(8, 8)
Open_Btn = ttk.Button(window, style='primary.Outline.TButton', text="Open Image",
                      image=img_open, compound=LEFT, command=lambda: open_File(canvas))
Save_Btn = ttk.Button(window, style='primary.Outline.TButton',
                      text="Save Image", image=img_save, compound=LEFT, command=lambda: save_File(canvas))
addNoise_Btn = ttk.Button(window, style='primary.Outline.TButton',
                          text="Add noise", image=img_ANoise, compound=LEFT, command=lambda: add_Noise(canvas))
barCode_Btn = ttk.Button(BD_LFrame, style='primary.Outline.TButton',
                         text="Bar code detection", image=img_Barcode, compound=LEFT, command=lambda: bcDetect_Image(canvas))
##
# Setting Place
##

Open_Btn.grid(row=0, column=0, ipadx=30, padx=(22, 0), pady=5, sticky="nw")
Save_Btn.grid(row=1, column=0, ipadx=31.75, padx=(15.5, 0), pady=(0, 5), sticky="n")
addNoise_Btn.grid(row=2, column=0, ipadx=37, padx=(19, 0), sticky="n")
BD_LFrame.grid(row=3, column=0, padx=(16,0))
BDetect_MButton.grid(row=0, column=0, padx=7, pady=10, columnspan=2)
SL_LFrame.grid(row=4, column=0, padx=(16, 0))
SLight_MButton.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
SS_LFrame.grid(row=5, column=0, padx=(16, 0))
SSmooth_MButton.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
DB_LFrame.grid(row=6, column=0, padx=(16, 0))
DBackground_MButton.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
RN_LFrame.grid(row=7, column=0, padx=(16, 0))
RNoise_MButton.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
DE_LFrame.grid(row=8, column=0, padx=(16, 0))
DEdge_MButton.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
# canvas.grid(column=1, row=0, rowspan=10, padx=5, columnspan=6, pady=(10, 0))
canvas.place(x=215, y=5)
# Histrogram_B`utton.grid(row=0,column=0,padx=10,pady=10)`
# window.after(10000,lambda:window.destroy())
# show_frame(lmain,cap)
window.mainloop()

