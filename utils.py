import numpy as np
import cv2 as cv


MIN_CONTOUR_AREA = 25
MAX_CONTOUR_AREA = 120000
MIN_PEN_THICKNESS = 1
MAX_PEN_THICKNESS = 20


def filterColor(frame, lower_filter, upper_filter):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    colorMask = cv.inRange(hsv, lower_filter, upper_filter)
    frame_colorFiltered = cv.bitwise_and(frame, frame, mask=colorMask)

    return frame_colorFiltered


# ---------------------------- #


def thresholdImage(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, frame_low = cv.threshold(frame_gray, 5, 255, 0)
    ret, frame_high = cv.threshold(frame_gray, 255, 255, cv.THRESH_BINARY_INV)
    frame_thresh = cv.bitwise_and(frame_low, frame_high)
    cv.imshow("", frame_thresh)
    return frame_thresh


def findPenTipContour(frame):
    img = thresholdImage(frame)  # returns grayscale

    contours, hierarchy = cv.findContours(
        img,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )

    max_contour = None
    max_contour_area = MIN_CONTOUR_AREA
    max_allowable_area = MAX_CONTOUR_AREA
    for c in contours:
        if (
            cv.contourArea(c) > max_contour_area
            and cv.contourArea(c) < max_allowable_area
        ):
            max_contour = c
            max_contour_area = cv.contourArea(c)

    return max_contour


def revealContour(frame, contour):
    frame_contour = frame.copy()
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(frame_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.drawContours(frame_contour, contour, -1, (0, 0, 255), 2)

    return frame_contour


# --------------------------------- #


def getBallCenter2D(contour):
    if contour is None:
        return np.array([-1, -1], dtype=np.int16)

    M = cv.moments(contour)
    if M["m00"] != 0:
        # cx = int(M["m10"] / M["m00"])
        # cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv.boundingRect(contour)
        cx = int(x + w // 2)
        cy = int(y + h // 2)

        point2D = np.array([cx, cy], dtype=np.int16)
    else:
        point2D = np.array([-1, -1], dtype=np.int16)

    return point2D


def getBallCenter3D(contour_L, contour_R):
    point3D = None
    Q = None

    M_L = cv.moments(contour_L)
    M_R = cv.moments(contour_R)
    if M_L["m00"] != 0 and M_R["m00"] != 0:
        cx_L = int(M_L["m10"] / M_L["m00"])
        cy_L = int(M_L["m01"] / M_L["m00"])
        cx_R = int(M_R["m10"] / M_R["m00"])
        cy_R = int(M_R["m01"] / M_R["m00"])

        d = cx_L - cx_R
        point2D = np.array([[[cx_L, cy_L, d]]], dtype=np.float32)
        point3D = cv.perspectiveTransform(point2D, Q).T

    return point3D


frame_pen = np.zeros((480, 640, 4), dtype=np.uint8)


def mergePenAndFrame(frame):
    global frame_pen

    alpha = frame_pen[:, :, 3].copy().astype(float) / 255
    frame_drawn = (
        frame_pen[:, :, :3].astype(float) * alpha[..., None].astype(float) / 255
        + frame * (1 - alpha[..., None])
    ).astype(np.uint8)

    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            if frame_pen[x, y, 3] > 0:
                frame_pen[x, y, 3] -= 5

    # frame_drawn = frame.copy()
    # for x in range(frame.shape[0]):
    #     for y in range(frame.shape[1]):
    #         if frame_pen[x, y, 3] > 0:
    #             frame_drawn[x, y, :] = frame_pen[x, y, :3]
    #             frame_pen[x, y, 3] -= 1

    return frame_drawn


previousPoint = [-1, -1]


def draw(frame, contour):
    global frame_pen
    global previousPoint

    pt = getBallCenter2D(contour)
    # getBallCenter3D()

    frame_drawn = frame.copy()
    # print(pt)
    if pt[0] < 0 or previousPoint[0] < 0:
        pass
    else:
        if contour is None:
            print("No contour")
        # r = np.sqrt(cv.contourArea(contour)) / 1000
        # t = int(MIN_PEN_THICKNESS + (cv.contourArea(contour) - MIN_CONTOUR_AREA) * (MAX_PEN_THICKNESS - MIN_PEN_THICKNESS) / (MAX_CONTOUR_AREA - MIN_CONTOUR_AREA))
        # scaled = out_min + (out_max - out_min) * (math.log(x) - math.log(in_min)) / (
        #     math.log(in_max) - math.log(in_min)
        # )
        t = int(
            MIN_PEN_THICKNESS
            + (MAX_PEN_THICKNESS - MIN_PEN_THICKNESS)
            * (np.log(cv.contourArea(contour)) - np.log(MIN_CONTOUR_AREA))
            // (np.log(MAX_CONTOUR_AREA) - np.log(MIN_CONTOUR_AREA))
        )
        cv.line(frame_pen, previousPoint, pt, (255, 0, 255, 255), thickness=t)
        # cv.circle(frame_pen, pt, radius=int(r), color=(255, 0, 255, 255), thickness=1)

    previousPoint = pt

    frame_drawn = mergePenAndFrame(frame)
    return frame_drawn
