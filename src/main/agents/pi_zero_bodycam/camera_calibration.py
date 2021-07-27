from src.main.agents.pi_zero_bodycam.pizero_capture_client import PiZeroCaptureClient
import argparse
import cv2
import numpy as np
import time


# From https://learnopencv.com/camera-calibration-using-opencv/
def calibrate(images):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard.')
    parser.add_argument('host', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('xres', type=int)
    parser.add_argument('yres', type=int)
    parser.add_argument('n', type=int)

    args = parser.parse_args()
    capture_client = PiZeroCaptureClient(host=args.host,
                                         port=args.port,
                                         res=(args.xres, args.yres))
    capture_client.connect()

    images = []
    for i in range(args.n):
        print(f"Capturing image {i}/{args.n}")
        img = capture_client.capture_image()
        images.append(img)
        time.sleep(2)

    calibrate(images)
