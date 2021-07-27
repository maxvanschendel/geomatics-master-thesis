import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import Imu
from typing import Callable
from abc import ABC, abstractmethod


class RosDataPublisher(ABC):
    @abstractmethod
    def init_node(self):
        pass

    @abstractmethod
    def publish(self, data):
        pass


class RosOdometryPublisher(RosDataPublisher):
    def __init__(self, node_name: str = "ROS_ODOMETRY_PUBLISHER", publish_path: str = "/imu"):
        self.node_name = node_name
        self.publish_path = publish_path
        self.publisher = self.init_node()

    def init_node(self):
        rospy.init_node(self.node_name)
        return rospy.Publisher(self.publish_path, Imu)

    def publish(self, msg: Imu):
        self.publisher.publish(msg)


class RosImagePublisher(RosDataPublisher):
    def __init__(self, node_name: str = "ROS_IMAGE_PUBLISHER", publish_path: str = "/camera/image_raw"):
        self.node_name = node_name
        self.publish_path = publish_path
        self.publisher = self.init_node()

    def init_node(self):
        rospy.init_node(self.node_name)
        return rospy.Publisher(self.publish_path, SensorImage)

    def publish(self, msg: SensorImage):
        self.publisher.publish(msg)


class SensorCaptureServerException(Exception):
    pass


class SocketListener:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket, self.connection = self.open_connection()

    def open_connection(self):
        server_socket = socket.socket()
        server_socket.bind((self.ip, self.port))
        server_socket.listen(0)

        # Accept a single connection and make a file-like object out of it
        connection = server_socket.accept()[0].makefile('rb')
        return server_socket, connection

    def close_connection(self):
        self.connection.close()
        self.server_socket.close()

    def listen(self, callback: Callable[[object], None] = None):
        try:
            while True:
                # Read the length of the data as a 32-bit unsigned int. If the length is zero, quit the loop
                stream_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not stream_len:
                    break

                # Construct a stream to hold the data
                stream = io.BytesIO()
                stream.write(self.connection.read(stream_len))

                # Rewind the stream, open it as an image with PIL and do some processing on it
                stream.seek(0)
                callback(stream)
        finally:
            self.close_connection()


class SensorCaptureServer:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket, self.connection = self.open_connection()

    def open_connection(self):
        server_socket = socket.socket()
        server_socket.bind((self.ip, self.port))
        server_socket.listen(0)

        # Accept a single connection and make a file-like object out of it
        connection = server_socket.accept()[0].makefile('rb')
        return server_socket, connection

    def close_connection(self):
        self.connection.close()
        self.server_socket.close()

    def read_stream(self, stream, show_imgs: bool, img_callback: Callable[[SensorImage], None] = None,):
        img = Image.open(stream)
        img_np = np.array(img)

        sensor_img = CvBridge().cv2_to_imgmsg(img_np)
        img_callback(sensor_img)

        if show_imgs:
            cv2.imshow('incoming_image_window', img_np)
            cv2.waitKey(10)

    def listen(self, show_imgs: bool,
               img_callback: Callable[[SensorImage], None] = None,
               imu_callback: Callable[[Imu], None] = None):

        # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means all interfaces)
        try:
            while True:
                # Read the length of the image as a 32-bit unsigned int. If the length is zero, quit the loop
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break

                # Construct a stream to hold the image data and read the image data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.connection.read(image_len))

                # Rewind the stream, open it as an image with PIL and do some processing on it
                image_stream.seek(0)
                img = Image.open(image_stream)
                img_np = np.array(img)

                sensor_img = CvBridge().cv2_to_imgmsg(img_np)
                img_callback(sensor_img)

                if show_imgs:
                    cv2.imshow('incoming_image_window', img_np)
                    cv2.waitKey(10)
        finally:
            self.close_connection()


if __name__ == "__main__":
    ros_image_publisher = RosImagePublisher("SENSOR_CAPTURE_SERVER", "/camera/image_raw")
    #ros_odom_publisher = RosOdometryPublisher("ROS_ODOMETRY_PUBLISHER", "/imu")

    sensor_capture_server = SensorCaptureServer('0.0.0.0', 8000)
    sensor_capture_server.listen(show_imgs=True,
                                 img_callback=ros_image_publisher.publish,
                                 imu_callback=None)
