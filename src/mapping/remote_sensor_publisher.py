import numpy as np
from cv_bridge import CvBridge
import cv2

from ros import RosImagePublisher
from sockets import SocketListener


class SensorCaptureNodeException(Exception):
    pass


class RemoteSensorPublisher:
    def __init__(self, ip: str, port: int, name: str, publish_path: str):
        self.name = name
        self.publish_path = publish_path
        self.publisher = RosImagePublisher(self.name, self.publish_path)

        self.server_socket = SocketListener(ip, port)
        self.server_socket.listen(callback=self.publish_image)

    def publish_image(self, stream):
        data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, 1)
        image = image[:, :, ::-1]

        sensor_img = CvBridge().cv2_to_imgmsg(image)
        self.publisher.publish(sensor_img)


if __name__ == "__main__":
    sensor_capture_server = RemoteSensorPublisher('0.0.0.0', 8000, "IMAGE_CAPTURE_NODE", "/camera/image_raw")
