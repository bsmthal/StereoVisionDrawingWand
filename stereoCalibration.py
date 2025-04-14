import cv2 as cv
import os
import glob
import numpy as np

# folder_path = r"C:\Users\blake\OneDrive\Documents\ECEN 631\FinalProject\StereoVisionDrawingWand\calibrationImages"
folder_path = "./calibrationImages"

left_images = glob.glob(os.path.join(folder_path, "L*.png"))
right_images = glob.glob(os.path.join(folder_path, "R*.png"))
stereo_left_images = glob.glob(os.path.join(folder_path, "SL*.png"))
stereo_right_images = glob.glob(os.path.join(folder_path, "SR*.png"))

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHECKERBOARD = (10, 7)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objpointsl = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

for frame in left_images:
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpointsl.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpointsL.append(corners2)
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)
cv.destroyAllWindows()
objpointsr = []
for frame in right_images:
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpointsr.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpointsR.append(corners2)
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(0)
cv.destroyAllWindows()

ret, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(
    objpointsl, imgpointsL, gray.shape[::-1], None, None
)
ret, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
    objpointsr, imgpointsR, gray.shape[::-1], None, None
)
print("Left Camera Matrix:\n", mtxL)
print("Left Distortion Coefficients:\n", distL)
print("Right Camera Matrix:\n", mtxR)
print("Right Distortion Coefficients:\n", distR)

# Stereo calibration
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objs = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objs[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objs = objs * 3.88
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

for framer, framel in zip(stereo_right_images, stereo_left_images):
    imgR = cv.imread(framer)
    imgL = cv.imread(framel)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD, None)
    retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD, None)
    if retR == True and retL == True:
        objpoints.append(objs)
        corners2R = cv.cornerSubPix(grayR, cornersR, (7, 7), (-1, -1), criteria)
        corners2L = cv.cornerSubPix(grayL, cornersL, (7, 7), (-1, -1), criteria)
        imgpointsL.append(corners2L)
        imgpointsR.append(corners2R)
        cv.drawChessboardCorners(imgL, CHECKERBOARD, corners2L, retL)
        cv.drawChessboardCorners(imgR, CHECKERBOARD, corners2R, retR)
        combined = np.hstack((imgL, imgR))
        cv.imshow("combined", combined)
        cv.waitKey(0)
cv.destroyAllWindows
(
    retStereo,
    newCameraMatrixL,
    distL,
    newCameraMatrixR,
    distR,
    rot,
    trans,
    essentialMatrix,
    fundamentalMatrix,
) = cv.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    gray.shape[::-1],
    criteria_stereo,
    flags,
)

print(
    f"retStereo: \n{retStereo}\nnewCameraMatrixL: \n{newCameraMatrixL}\ndistL: \n{distL}\nnewCameraMatrixR: \n{newCameraMatrixR}\ndistR: \n{distR}\nrot: \n{rot}\ntrans: \n{trans}\nessentialMatrix: \n{essentialMatrix}\nfundamentalMatrix: \n{fundamentalMatrix}"
)
