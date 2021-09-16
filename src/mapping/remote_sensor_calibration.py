import cv2
import numpy as np
from sockets import SocketListener


class RemoteSensorCalibration:
    images = []

    def __init__(self, ip: str, port: int, n_imgs: int):
        self.n_imgs = n_imgs

        self.server_socket = SocketListener(ip, port)
        self.server_socket.listen(callback=self.add_calibration_image)

    def add_calibration_image(self, stream):
        data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, 1)
        self.images.append(image)

        if len(self.images) >= self.n_imgs:
            mtx, dist = RemoteSensorCalibration.calibrate(self.images)
            print(mtx, dist)

    # Adapted from https://learnopencv.com/camera-calibration-using-opencv/
    @staticmethod
    def calibrate(images):
        """Calibrate camera from sequence of images of checkerboard pattern."""

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (6, 9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        for img in images:
            # Find the chess board corners. If desired number of corners are found in the image then ret = true
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)

                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist


if __name__ == "__main__":
    calibration = RemoteSensorCalibration('0.0.0.0', 8000, 20)
