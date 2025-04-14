import concurrent.futures
from tkinter import *
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

    # Define left image processing
    def process_left():
        filtered = utils.filterColor(frame_L, lower_filter, upper_filter)
        contour = utils.findPenTipContour(filtered)
        return contour

    # Define right image processing
    def process_right():
        filtered = utils.filterColor(frame_R, lower_filter, upper_filter)
        contour = utils.findPenTipContour(filtered)
        return contour

    # Run both processing functions in parallel
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    future_L = executor.submit(process_left)
    future_R = executor.submit(process_right)

    # Get results
    contour_L = future_L.result()
    contour_R = future_R.result()

    pt_3d = utils.getBallCenter3D(contour_L=contour_L, contour_R=contour_R)
    print(pt_3d)
    try:
        # Logarithmic Scaling
        t = int(
            MIN_PEN_THICKNESS
            + (MAX_PEN_THICKNESS - MIN_PEN_THICKNESS)
            * (np.log(pt_3d[2]) - np.log(MIN_Z_VALUE))
            // (np.log(MAX_Z_VALUE) - np.log(MIN_Z_VALUE))
        )

        # Linear Scaling
        # t = int(
        #     MIN_PEN_THICKNESS
        #     + (pt_3d[2] - MIN_Z_VALUE)
        #     * (MAX_PEN_THICKNESS - MIN_PEN_THICKNESS)
        #     / (MAX_Z_VALUE - MIN_Z_VALUE)
        # )
    except:
        t = 1

    def p_left():
        drawn = cv.flip(utils.draw(frame_L, contour_L, isLeft=True, thick=t), 1)
        return drawn

    def p_right():
        drawn = cv.flip(utils.draw(frame_R, contour_R, isLeft=False, thick=t), 1)
        return drawn

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futu_L = executor.submit(p_left)
    futu_R = executor.submit(p_right)

    frame_drawn_L = futu_L.result()
    frame_drawn_R = futu_R.result()

    # --------------------------------- #

    drawn_footage = np.append(frame_drawn_R, frame_drawn_L, axis=1)
    cv.imshow("footage", drawn_footage)

    # Escape
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
