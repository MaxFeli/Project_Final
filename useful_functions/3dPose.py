import glob
import cv2
import numpy as np


def calibImg(path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            # cv2.imshow('img_old', img)
            # cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def draw(img, corners, imgpts):
    corner = tuple(corners[24].ravel().astype('uint16'))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype('uint16')), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype('uint16')), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype('uint16')), (0, 0, 255), 5)
    return img


if __name__ == "__main__":
    path = "img_cali/*.png"
    mtx, dist = calibImg(path)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7*7, 3), np.float32)
    objp[:, :2] = np.mgrid[-3:4, -3:4].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    fname = "chessboard_frame_-1.png"
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:-4]+'_edit.png', img)
    cv2.destroyAllWindows()


