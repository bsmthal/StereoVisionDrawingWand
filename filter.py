import concurrent.futures
import customtkinter, tkinter
from tkinter import *
import threading
import numpy as np
import cv2 as cv
import utils


MIN_CONTOUR_AREA = 30
MAX_CONTOUR_AREA = 120000
MIN_Z_VALUE = 85
MAX_Z_VALUE = 8
MIN_PEN_THICKNESS = 2
MAX_PEN_THICKNESS = 80
FADE_RATE = 5
PEN_COLOR = (182, 120, 36, 255)


# Orange Ping-Pong Ball
# lh = 15
# uh = 45
# ls = 36
# us = 255
# lv = 206
# uv = 255

# Orange Ping-Pong Ball (Home)
lh = 9
uh = 16
ls = 188
us = 255
lv = 47
uv = 255

# Illumminated Orange Ping-Pong Ball
# lh = 7
# uh = 41
# ls = 21
# us = 36
# lv = 236
# uv = 255

# Blue Master Locks
# lh = 73
# uh = 124
# ls = 146
# us = 255
# lv = 20
# uv = 255


def process():
    global lh
    global uh
    global ls
    global us
    global lv
    global uv

    cap_L = cv.VideoCapture(2)
    cap_R = cv.VideoCapture(1)

    while True:
        ret_L, frame_L = cap_L.read()
        ret_R, frame_R = cap_R.read()

        if not ret_L or not ret_R:
            continue

        # Color Filter
        lower_filter = np.array([lh, ls, lv])
        upper_filter = np.array([uh, us, uv])

        # --------------------------------- #

        frame_colorFiltered_L = utils.filterColor(
            frame=frame_L,
            lower_filter=lower_filter,
            upper_filter=upper_filter,
        )
        frame_colorFiltered_R = utils.filterColor(
            frame=frame_R,
            lower_filter=lower_filter,
            upper_filter=upper_filter,
        )

        # Find "Pen" Tip
        contour_L = utils.findPenTipContour(frame_colorFiltered_L)
        contour_R = utils.findPenTipContour(frame_colorFiltered_R)

        frame_contour_L = utils.revealContour(frame_colorFiltered_L, contour_L)
        frame_contour_R = utils.revealContour(frame_colorFiltered_R, contour_R)

        # --------------------------------- #

        # Display Image Processing
        original_footage = np.append(frame_L, frame_R, axis=1)
        filtered_footage = np.append(
            frame_colorFiltered_L, frame_colorFiltered_R, axis=1
        )
        contour_footage = np.append(frame_contour_L, frame_contour_R, axis=1)
        disp = np.append(original_footage, contour_footage, axis=0)
        cv.imshow("footage", disp)

        # Print Info
        area = 0
        if contour_L is not None:
            area = cv.contourArea(contour_L)
        print(f"lh={lh} uh={uh} ls={ls} us={us} lv={lv} uv={uv}\tarea={area}")

        # Escape
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
