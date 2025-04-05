import customtkinter, tkinter
from tkinter import *
import threading
import numpy as np
import cv2 as cv


lh = 0
uh = 179
ls = 49
us = 255
lv = 165
uv = 255


def process():
    global lh
    global uh
    global ls
    global us
    global lv
    global uv

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        # Color Filter
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_filter = np.array([lh, ls, lv])
        upper_filter = np.array([uh, us, uv])
        colorMask = cv.inRange(hsv, lower_filter, upper_filter)
        frame_colorFiltered = cv.bitwise_and(frame, frame, mask=colorMask)

        disp = np.append(frame, frame_colorFiltered, axis=1)
        cv.imshow("footage", disp)

        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()
    return


# ------------------------------------------------------------- #

t = threading.Thread(target=process, args=())
t.start()


def slider_l1_event(value):
    global lh
    lh = int(value)


def slider_h1_event(value):
    global uh
    uh = int(value)


def slider_l2_event(value):
    global ls
    ls = int(value)


def slider_h2_event(value):
    global us
    us = int(value)


def slider_l3_event(value):
    global lv
    lv = int(value)


def slider_h3_event(value):
    global uv
    uv = int(value)


app = customtkinter.CTk()
frm = customtkinter.CTkFrame(master=app, width=800, height=200, corner_radius=10)
frm.pack(padx=20, pady=20)
var = tkinter.IntVar(value="lThreshold")
slider_l1 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_l1_event
)
slider_h1 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_h1_event
)
slider_l2 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_l2_event
)
slider_h2 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_h2_event
)
slider_l3 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_l3_event
)
slider_h3 = customtkinter.CTkSlider(
    master=frm, from_=0, to=255, command=slider_h3_event
)
slider_l1.pack(side=tkinter.TOP)
slider_h1.pack(side=tkinter.TOP)
slider_l2.pack(side=tkinter.TOP)
slider_h2.pack(side=tkinter.TOP)
slider_l3.pack(side=tkinter.TOP)
slider_h3.pack(side=tkinter.TOP)
app.mainloop()
