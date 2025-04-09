import cv2 as cv
import numpy as np


NUM_CALIB_IMAGES = 20
patternSize = (10, 7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)
winSize = (7, 7)
zeroZone = (-1, -1)


cap_L = cv.VideoCapture(1)
cap_R = cv.VideoCapture(2)

count = 0
while count < NUM_CALIB_IMAGES:
    ret_L, frame_L = cap_L.read()
    ret_R, frame_R = cap_R.read()
    img_L = frame_L.copy()
    img_R = frame_R.copy()

    if not ret_L or not ret_R:
        continue

    # --------------------------------- #

    # Find chessboard corners
    img_gray_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)
    img_gray_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)

    cornerWasFound_L, chess_corners_L = cv.findChessboardCorners(
        img_gray_L, patternSize
    )
    cornerWasFound_R, chess_corners_R = cv.findChessboardCorners(
        img_gray_R, patternSize
    )

    # Refine Corners
    if cornerWasFound_L and cornerWasFound_R:
        refined_corners_L = cv.cornerSubPix(
            img_gray_L, chess_corners_L, winSize, zeroZone, criteria
        )
        refined_corners_R = cv.cornerSubPix(
            img_gray_R, chess_corners_R, winSize, zeroZone, criteria
        )
    else:
        refined_corners_L = chess_corners_L
        refined_corners_R = chess_corners_R
        # print("!!! CORNERS NOT FOUND")

    # Display chessboard corners
    cv.drawChessboardCorners(frame_L, patternSize, refined_corners_L, cornerWasFound_L)
    cv.drawChessboardCorners(frame_R, patternSize, refined_corners_R, cornerWasFound_R)

    frame = np.append(frame_L, frame_R, axis=1)
    cv.imshow("Frame", frame)

    # --------------------------------- #

    # Capture
    if cv.waitKey(1) == ord("p"):
        # Save picture here
        cv.imwrite(f"calibrationImages/SL{count:02}.png", img_L)
        cv.imwrite(f"calibrationImages/SR{count:02}.png", img_L)
        count += 1
        print(f"Stereo:\t{count}")
        pass

cv.destroyAllWindows()


count = 0
while count < NUM_CALIB_IMAGES:
    ret_L, frame_L = cap_L.read()
    img_L = frame_L.copy()

    if not ret_L:
        continue

    # --------------------------------- #

    # Find chessboard corners
    img_gray_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)

    cornerWasFound_L, chess_corners_L = cv.findChessboardCorners(
        img_gray_L, patternSize
    )

    # Refine Corners
    if cornerWasFound_L and cornerWasFound_R:
        refined_corners_L = cv.cornerSubPix(
            img_gray_L, chess_corners_L, winSize, zeroZone, criteria
        )
    else:
        refined_corners_L = chess_corners_L
        # print("!!! CORNERS NOT FOUND")

    # Display chessboard corners
    cv.drawChessboardCorners(frame_L, patternSize, refined_corners_L, cornerWasFound_L)

    cv.imshow("Frame", frame_L)

    # --------------------------------- #

    # Capture
    if cv.waitKey(1) == ord("p"):
        # Save picture here
        cv.imwrite(f"calibrationImages/L{count:02}.png", img_L)
        count += 1
        print(f"Left:\t{count}")
        pass

cv.destroyAllWindows()

count = 0
while count < NUM_CALIB_IMAGES:
    ret_R, frame_R = cap_R.read()
    img_R = frame_R.copy()

    if not ret_R:
        continue

    # --------------------------------- #

    # Find chessboard corners
    img_gray_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)

    cornerWasFound_R, chess_corners_R = cv.findChessboardCorners(
        img_gray_R, patternSize
    )

    # Refine Corners
    if cornerWasFound_R and cornerWasFound_R:
        refined_corners_R = cv.cornerSubPix(
            img_gray_R, chess_corners_R, winSize, zeroZone, criteria
        )
    else:
        refined_corners_R = chess_corners_R
        # print("!!! CORNERS NOT FOUND")

    # Display chessboard corners
    cv.drawChessboardCorners(frame_R, patternSize, refined_corners_R, cornerWasFound_R)

    cv.imshow("Frame", frame_R)

    # --------------------------------- #

    # Capture
    if cv.waitKey(1) == ord("p"):
        # Save picture here
        cv.imwrite(f"calibrationImages/R{count:02}.png", img_R)
        count += 1
        print(f"Right:\t{count}")
        pass

cv.destroyAllWindows()
