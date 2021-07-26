import io
import socket
import struct
import time
import picamera
import argparse


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

    def stream_video(self):
        print("Initializing video streaming")

        # Make a file-like object out of the connection
        connection = self.client_socket.makefile('wb')
        try:
            camera = picamera.PiCamera()
            camera.resolution = self.res

            # Let the camera warm up for 2 seconds
            time.sleep(2)

            # Note the start time and construct a stream to hold image data
            # temporarily (we could write it directly to connection but in this
            # case we want to find out the size of each capture first to keep
            # our protocol simple)
            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                # Write the length of the capture to the stream and flush to ensure it actually gets sent
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                # Rewind the stream and send the image data over the wire
                stream.seek(0)
                connection.write(stream.read())

                # Reset the stream for the next capture
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
