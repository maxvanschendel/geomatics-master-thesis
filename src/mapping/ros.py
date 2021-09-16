from abc import ABC, abstractmethod
from rospy import init_node, Publisher
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import Imu



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
        return Publisher(self.publish_path, Imu)

    def publish(self, msg: Imu):
        self.publisher.publish(msg)


class RosImagePublisher(RosDataPublisher):
    def __init__(self, node_name: str = "ROS_IMAGE_PUBLISHER", publish_path: str = "/camera/image_raw"):
        self.node_name = node_name
        self.publish_path = publish_path
        self.publisher = self.init_node()

    def init_node(self):
        return Publisher(self.publish_path, SensorImage, queue_size=100)

    def publish(self, msg: SensorImage):
        self.publisher.publish(msg)
