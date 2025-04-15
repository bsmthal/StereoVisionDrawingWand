import numpy as np
import cv2 as cv


MIN_CONTOUR_AREA = 50
MAX_CONTOUR_AREA = 120000
MIN_Z_VALUE = 200
MAX_Z_VALUE = 10
MIN_PEN_THICKNESS = 1
MAX_PEN_THICKNESS = 50
FADE_RATE = 17
PEN_COLOR = (182, 120, 36, 255)
PEN_ERR_COLOR = (0, 0, 255, 255)

cameraMatrix1 = np.array(
    [
        [757.22843882, 0.0, 320.3740059],
        [0.0, 758.51441095, 233.85838898],
        [0.0, 0.0, 1.0],
    ]
)
distCoeffs1 = np.array(
    [[-3.89929360e-02, 2.10226478e00, -4.38657735e-03, 1.97721858e-03, -9.46339572e00]]
)
cameraMatrix2 = np.array(
    [
        [760.67003412, 0.0, 334.76383869],
        [0.0, 761.42430642, 238.78456012],
        [0.0, 0.0, 1.0],
    ]
)
distCoeffs2 = np.array(
    [[-8.59026936e-02, 3.65916735e00, -2.69853348e-04, 5.15314334e-03, -2.19480083e01]]
)
imageSize = (640, 480)
R = np.array(
    [
        [0.99995199, -0.00318981, -0.00926529],
        [0.00299109, 0.99976687, -0.02138379],
        [0.00933134, 0.02135505, 0.99972841],
    ]
)
T = np.array([[-6.58059429], [-0.0243418], [0.48546735]])
R1, R2, P1, P2, Q, roi_L, roi_R = cv.stereoRectify(
    cameraMatrix1=cameraMatrix1,
    distCoeffs1=distCoeffs1,
    cameraMatrix2=cameraMatrix2,
    distCoeffs2=distCoeffs2,
    imageSize=imageSize,
    R=R,
    T=T,
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=0.5,
)


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
    # cv.imshow("", frame_thresh)
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

    M_L = cv.moments(contour_L)
    M_R = cv.moments(contour_R)
    if M_L["m00"] != 0 and M_R["m00"] != 0:
        cx_L = int(M_L["m10"] / M_L["m00"])
        cy_L = int(M_L["m01"] / M_L["m00"])
        cx_R = int(M_R["m10"] / M_R["m00"])
        cy_R = int(M_R["m01"] / M_R["m00"])

        d = cx_L - cx_R
        point2D = np.array([[[cx_L, cy_L, d]]], dtype=np.float32)
        point3D = cv.perspectiveTransform(point2D, Q).T.reshape(3)

    return point3D


frame_pen_L = np.zeros((480, 640, 4), dtype=np.uint8)
frame_pen_R = np.zeros((480, 640, 4), dtype=np.uint8)


def mergePenAndFrame(frame, isLeft: bool):
    global frame_pen_L
    global frame_pen_R
    if isLeft:
        frame_pen = frame_pen_L
    else:
        frame_pen = frame_pen_R

    # cv.imshow("pen", frame_pen)

    alpha = frame_pen[:, :, 3].astype(float) / 255
    frame_drawn = (
        frame_pen[:, :, :3].astype(float) * alpha[..., None]
        + frame * (1 - alpha[..., None])
    ).astype(np.uint8)

    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            if frame_pen[x, y, 3] > 0:
                frame_pen[x, y, 3] -= FADE_RATE

    # frame_drawn = frame.copy()
    # for x in range(frame.shape[0]):
    #     for y in range(frame.shape[1]):
    #         if frame_pen[x, y, 3] > 0:
    #             frame_drawn[x, y, :] = frame_pen[x, y, :3]
    #             frame_pen[x, y, 3] -= 1

    return frame_drawn


previousPoint_L = [-1, -1]
previousPoint_R = [-1, -1]


def draw(frame, contour, isLeft: bool, thick: int):
    global frame_pen_L
    global frame_pen_R
    global previousPoint_L
    global previousPoint_R
    if isLeft:
        previousPoint = previousPoint_L
    else:
        previousPoint = previousPoint_R
    pt = getBallCenter2D(contour)

    frame_drawn = frame.copy()
    # print(pt)
    if pt[0] < 0 or previousPoint[0] < 0:
        pass
    else:
        try:
            if thick == 1:
                penColor = PEN_ERR_COLOR
            else:
                penColor = PEN_COLOR

            if isLeft:
                cv.line(frame_pen_L, previousPoint, pt, penColor, thickness=thick)
                # cv.circle(
                #     frame_pen_L, pt, radius=int(thick), color=penColor, thickness=thick
                # )
            else:
                cv.line(frame_pen_R, previousPoint, pt, penColor, thickness=thick)
        except:
            pass

    if isLeft:
        previousPoint_L = pt
    else:
        previousPoint_R = pt

    frame_drawn = mergePenAndFrame(frame, isLeft=isLeft)
    return frame_drawn
