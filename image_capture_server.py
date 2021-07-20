import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image as SensorImage

node_name = "ORB_SLAM3_OVER_SOCKET"
rospy.init_node(node_name)
image_pub = rospy.Publisher("/camera/image_raw",SensorImage)
bridge = CvBridge()

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # Rad the length of the image as a 32-bit unsigned int. If the length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break

        # Construct a stream to hold the image data and read the image data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        # Rewind the stream, open it as an image with PIL and do some processing on it
        image_stream.seek(0)
        img = Image.open(image_stream)
        img_np = np.array(img)

        image_pub.publish(bridge.cv2_to_imgmsg(img_np))

        cv2.imshow('incoming_image_window', img_np)
        key = cv2.waitKey(10)
        # check for 'q' key-press
        if key == ord("q"):
            # if 'q' key-pressed break out
            break

finally:
    connection.close()
    server_socket.close()
