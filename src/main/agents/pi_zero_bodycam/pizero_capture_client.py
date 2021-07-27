import io
import socket
import struct
import time
import picamera
import argparse
from PIL import Image
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import json

class SensorMessageType(Enum):
    RGB = 1
    RGBD = 2
    DEPTH = 3
    IMU = 4


class SensorMessage(ABC):
    def __init__(self, msg_type: SensorMessageType, agent: str, time: int, data):
        self.msg_type = msg_type
        self.agent = agent
        self.time = time
        self.data = data

    def serialize(self):
        return json.dumps(self)


class PiZeroCaptureClient:
    def __init__(self, host: str, port: int, res: (int, int)):
        self.host = host
        self.port = port
        self.res = res
        self.client_socket = socket.socket()

    def connect(self):
        print(f"Connecting to socket | host: {self.host} | port: {self.port}")
        self.client_socket.connect((self.host, self.port))

    def disconnect(self):
        print(f"Disconnecting from socket | host: {self.host} | port: {self.port}")
        self.client_socket.close()

    def capture_image(self):
        camera = picamera.PiCamera()
        camera.resolution = self.res

        # Create the in-memory stream
        stream = io.BytesIO()
        with picamera.PiCamera() as camera:
            camera.capture(stream, format='jpeg')

        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)

        return Image.open(stream)

    def capture_video(self):
        print("Initializing video streaming")

        # Make a file-like object out of the connection
        connection = self.client_socket.makefile('wb')
        try:
            camera = picamera.PiCamera()
            camera.resolution = self.res

            print("Warming up camera")
            time.sleep(2)

            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                msg = SensorMessage(SensorMessageType.RGB, "raspberry", 0, stream.read())
                msg_serialized = msg.serialize()

                connection.send(msg_serialized.encode(encoding="utf-8"))

                stream.seek(0)
                stream.truncate()
            # Write a length of zero to the stream to signal we're done
            connection.write(struct.pack('<L', 0))
        finally:
            connection.close()
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Raspberry Pi Zero video streaming over websocket.')
    parser.add_argument('host', type=str)
    parser.add_argument('port', type=int)
    parser.add_argument('xres', type=int)
    parser.add_argument('yres', type=int)

    args = parser.parse_args()
    client = PiZeroCaptureClient(host=args.host,
                                 port=args.port,
                                 res=(args.xres, args.yres))

    client.connect()
    client.capture_video()
