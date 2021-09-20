import glob
import cv2
import numpy as np
from sklearn.cluster import MeanShift


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
    return mtx, dist, np.array(rvecs), np.array(tvecs)


def draw(img, corners, imgpts):
    corner = tuple(corners.ravel().astype('uint16'))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype('uint16')), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype('uint16')), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype('uint16')), (0, 0, 255), 5)
    return img


def clusterPoints(points):
    ms = MeanShift(bandwidth=10, cluster_all=False)
    ms.fit(points)
    cluster_centers = ms.cluster_centers_
    return cluster_centers


def find_corners(img):
    corner = []
    red = [0, 0, 255]
    X, Y = np.where(np.all(img == red, axis=2))
    zipped = np.column_stack((Y, X))
    points = clusterPoints(zipped)
    mdist_ul = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_dl = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_ur = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_dr = [img.shape[0]**2 + img.shape[1]**2, 0]
    for item in range(len(points)):
        dist_ul = (points[item][0])**2 + (points[item][1])**2
        dist_dl = (points[item][0])**2 + (points[item][1]-img.shape[0])**2
        dist_ur = (points[item][0]-img.shape[1])**2 + (points[item][1])**2
        dist_dr = (points[item][0]-img.shape[1])**2 + (points[item][1]-img.shape[0])**2
        if dist_ul < mdist_ul[0]:
            mdist_ul[0] = dist_ul
            mdist_ul[1] = item
        if dist_dl < mdist_dl[0]:
            mdist_dl[0] = dist_dl
            mdist_dl[1] = item
        if dist_ur < mdist_ur[0]:
            mdist_ur[0] = dist_ur
            mdist_ur[1] = item
        if dist_dr < mdist_dr[0]:
            mdist_dr[0] = dist_dr
            mdist_dr[1] = item
    corner.append(points[mdist_ul[1]])
    corner.append(points[mdist_ur[1]])
    corner.append(points[mdist_dl[1]])
    corner.append(points[mdist_dr[1]])
    corner = np.array(corner)
    return points, corner


def find_nearest(array, value):
    array = np.asarray(array)
    idx = ((array[:, 0] - value[0])**2+(array[:, 1]-value[1])**2).argmin()
    return array[idx]


if __name__ == "__main__":
    mtx, dist, rvecs, tvecs = calibImg("img_cali/*.png")
    objp = np.zeros((2*2, 3), np.float32)
    objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    img = cv2.imread("square_corners.png")
    points, corners = find_corners(img)
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist, rvec=rvecs, tvec=tvecs, flags=cv2.SOLVEPNP_P3P)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    sum = [0, 0]
    length = len(corners)
    for item in range(length):
        sum[0] += corners[item][0] / length
        sum[1] += corners[item][1] / length
    middle_point = find_nearest(points, sum)
    """
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    """
    dst = img
    f_img = draw(dst, middle_point, imgpts)
    cv2.imshow('Result', f_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
