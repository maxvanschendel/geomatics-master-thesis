import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image as SensorImage

# FROM: https://medium.com/analytics-vidhya/using-homography-for-pose-estimation-in-opencv-a7215f260fdd
def transformation_from_homography(H, K):
    '''H is the homography matrix
    K is the camera calibration matrix
    T is translation
    R is rotation
    '''

    H = H.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))

    return T, R


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
