import main
import pickle
import socket
import struct
import time

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
from model.frame import *


class KinectCaptureClient:
    def __init__(self, host: str, port: int, agent: str, frame_type: FrameType, k4a_config: Config, compression_quality: int):
        self.host = host
        self.port = port
        self.agent = agent

        self.frame_type = frame_type
        self.k4a = PyK4A(k4a_config)
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]

        self.client_socket = socket.socket()
        self.capturing = False

    def connect(self):
        print(f"Connecting to socket | host: {self.host} | port: {self.port}")
        self.client_socket.connect((self.host, self.port))

    def disconnect(self):
        print(f"Disconnecting from socket | host: {self.host} | port: {self.port}")
        self.client_socket.close()

    def get_imu(self):
        imu = self.k4a.get_imu_sample()

        imu_data = np.zeros((2, 3))
        imu_data[0] = np.array(imu.pop("acc_sample"))
        imu_data[1] = imu.pop("gyro_sample")

        return IMU(self.agent, imu['acc_timestamp'] + imu['gyro_timestamp'], imu_data)

    def get_frame(self):
        capture = self.k4a.get_capture()
        if np.any(capture.depth) and np.any(capture.color):
            if self.frame_type == FrameType.RGB:
                img = capture.color[:, :, :3]
            elif self.frame_type == FrameType.IR:
                img = capture.ir
            elif self.frame_type == FrameType.D:
                img = capture.depth
            elif self.frame_type == FrameType.RGBD:
                rgb = capture.transformed_color[:, :, :3]
                d = capture.depth
                img = np.dstack((rgb, d))
            elif self.frame_type == FrameType.RGBIR:  # TODO get color in IR sensor frame
                rgb = capture.transformed_color[:, :, :3]
                ir = capture.ir
                img = np.dstack((rgb, ir))
                
            elif self.frame_type == FrameType.IRD:  # TODO find way to compress 2-channel images
                ir = capture.ir
                d = capture.depth
                img = np.dstack((ir, d))
            elif self.frame_type == FrameType.RGBDI:  # TODO find way to compress 5-channel images
                rgb = capture.transformed_color[:, :, :3]
                d = capture.depth
                ir = capture.ir
                img = np.dstack((rgb, d, ir))

            # _, img = cv2.imencode('.png', img)
            print(img.dtype)

        return Frame(self.agent, get_unix_milli_time(), img, self.frame_type)
            
    def send(self, data):
        data_msg = pickle.dumps(data)
        self.client_socket.sendall(struct.pack("<L", len(data_msg))+data_msg)

    def capture(self):
        print("Starting data capture...")

        self.k4a.start()
        self.capturing = True

        while self.capturing:
            imu = self.get_imu()
            self.send(imu)

            frame = self.get_frame()
            self.send(frame)

        self.k4a.stop()
            

def get_unix_milli_time():
    return int(round(time.time() * 1000))

if __name__ == "__main__":
    client = KinectCaptureClient(host="192.168.53.22",
                                 port=8000,
                                 agent="kinectdk",
                                 frame_type=FrameType.RGBD,
                                 compression_quality=80,
                                 k4a_config=Config(
                                    color_resolution=pyk4a.ColorResolution.RES_720P,
                                    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                                    synchronized_images_only=True,
                                ))

    client.connect()
    client.capture()
