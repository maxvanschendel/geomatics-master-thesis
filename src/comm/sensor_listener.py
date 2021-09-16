import main
from io import BytesIO
from socket import socket
from struct import unpack, calcsize
import struct
from typing import Callable
import pickle
import cv2

from model.frame import *

class SensorListener:
    '''Creates a websocket server and starts listening for pickled sensor data.'''
    
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket = self.open_connection()
        self.listening = False

    def open_connection(self):
        server_socket = socket()
        server_socket.bind((self.ip, self.port))
        server_socket.listen(0)

        return server_socket

    def close_connection(self):
        self.connection.close()
        self.server_socket.close()

    def listen(self, callback: Callable[[], None] = None):
        print(f"Listening for sensor data at {self.ip}:{self.port}")
        conn, addr = self.server_socket.accept()
        
        data = b""
        payload_size = struct.calcsize("<L")

        self.listening = True
        while self.listening:
            try:
                
                
                # Get size of message from header
                while len(data) < payload_size:
                    data += conn.recv(4096)
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                
                # Download the rest of the message
                msg_size = struct.unpack("<L", packed_msg_size)[0]

                if msg_size > 1000000:
                    print(msg_size)
                    continue

                while len(data) < msg_size:
                    data += conn.recv(4096)

                data, sensor_data = data[msg_size:], data[:msg_size]
                sensor_object = pickle.loads(sensor_data)

                if callback is not None:
                    callback(sensor_object)
            except Exception as e:
                print(f"Listening failed: {e}")
        

def display_sensor_data(sensor_data):
    if type(sensor_data) == Frame:
        cv2.imshow("", cv2.imdecode(sensor_data.data, cv2.IMREAD_UNCHANGED))
        cv2.waitKey(10)
    else: 
        print(sensor_data)

if __name__ == "__main__":
    server_socket = SensorListener('0.0.0.0', 8000)
    server_socket.listen(callback=display_sensor_data)