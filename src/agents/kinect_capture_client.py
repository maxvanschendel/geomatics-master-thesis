import pickle
import socket
import time

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A



class KinectCaptureClient:
    def __init__(self, host: str, port: int, res, agent: str):
        self.host = host
        self.port = port
        self.res = res
        self.agent = agent
        self.client_socket = socket.socket()

        self.capturing = False

    def connect(self):
        print(f"Connecting to socket | host: {self.host} | port: {self.port}")
        self.client_socket.connect((self.host, self.port))

    def disconnect(self):
        print(f"Disconnecting from socket | host: {self.host} | port: {self.port}")
        self.client_socket.close()

    def capture(self, interval):
        k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
        )
        k4a.start()

        # getters and setters directly get and set on device
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        k4a.whitebalance = 4510
        assert k4a.whitebalance == 4510

        self.capturing = True

        while self.capturing:
            imu = k4a.get_imu_sample()
            imu["acc_x"], imu["acc_y"], imu["acc_z"] = imu.pop("acc_sample")
            imu["gyro_x"], imu["gyro_y"], imu["gyro_z"] = imu.pop("gyro_sample")

            capture = k4a.get_capture()

            if np.any(capture.depth) and np.any(capture.color):
                depth = capture.depth
                ir = capture.ir
                color = capture.color[:, :, :3]

                cv2.imshow("", color)
                cv2.waitKey(interval)

                msg = pickle.dumps(Frame.Frame(self.agent, get_unix_milli_time, color, FrameType.RGB))

                self.client_socket.send(msg)

        k4a.stop()
            

def get_unix_milli_time():
    return int(round(time.time() * 1000))

if __name__ == "__main__":
    client = KinectCaptureClient(host="127.0.0.1",
                                 port=8000,
                                 res=None,
                                 agent="kinectdk")

    client.connect()
    client.capture(10)
