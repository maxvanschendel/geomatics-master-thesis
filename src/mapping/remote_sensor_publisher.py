import numpy as np
from cv_bridge import CvBridge
import cv2
from model.frame import *
from rospy import init_node, Publisher
from mapping.ros import RosImagePublisher
from comm.sensor_listener import SensorListener


class SensorCaptureNodeException(Exception):
    pass


class RemoteSensorPublisher:
    def __init__(self, ip: str, port: int, name: str, rgb_publish_path: str, depth_publish_path: str):
        self.name = name
        self.rgb_publish_path = rgb_publish_path
        self.depth_publish_path = depth_publish_path

        self.rgb_publisher = RosImagePublisher(self.name, self.rgb_publish_path)
        self.depth_publisher = RosImagePublisher("D", self.depth_publish_path)

        self.sensor_listener = SensorListener(ip, port)
        self.sensor_listener.listen(callback=self.publish_image)

    def publish_image(self, stream):
        print("Publishing image")
        init_node("OLOELAD")

        if type(stream) == Frame:
            if stream.type == FrameType.RGB:
                data = stream.data
                image = cv2.imdecode(data, 1)
                image = image[:, :, ::-1]

                sensor_img = CvBridge().cv2_to_imgmsg(image)
                self.rgb_publisher.publish(sensor_img)
            if stream.type == FrameType.RGBD:
                data = stream.data
                image = cv2.imdecode(data, 1)
                image = image[:, :, ::-1]

                rgb = image[:, :, :3]
                d = image[:, :, -1:]


                rgb_msg = CvBridge().cv2_to_imgmsg(rgb)
                d_msg = CvBridge().cv2_to_imgmsg(d)

                self.rgb_publisher.publish(rgb_msg)
                self.depth_publisher.publish(d_msg)


if __name__ == "__main__":
    sensor_capture_server = RemoteSensorPublisher('0.0.0.0', 8000, "IMAGE_CAPTURE_NODE", "/camera/image_raw", "/camera/depth_registered/image_raw")
