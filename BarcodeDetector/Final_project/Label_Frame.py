from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
from ttkbootstrap import Style
root = window = Style(theme='solar').master


img = ImageTk.PhotoImage(Image.open("zoom_in.png"))  # PIL solution
btn = ttk.Button(root, image=img)
btn.grid(row=0,column=0)
mainloop()
