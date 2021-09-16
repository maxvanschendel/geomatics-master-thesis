from io import BytesIO
from socket import socket
from struct import unpack, calcsize
from typing import Callable
import pickle


class SocketListener:
    '''Creates a websocket server and starts listening for pickled objects.'''
    
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket = self.open_connection()

    def open_connection(self):
        server_socket = socket()
        server_socket.bind((self.ip, self.port))
        server_socket.listen(0)

        return server_socket

    def close_connection(self):
        self.connection.close()
        self.server_socket.close()

    def listen(self, callback: Callable[[], None] = None):
        conn, addr = self.server_socket.accept()

        while True:
            try:
                data = b""
                while True:
                    packet = conn.recv(4096)
                    if not packet: break
                    data += packet
            
                msg_object = pickle.loads(data)
                callback(msg_object)
            except Exception as e:
                pass
            finally:
                self.server_socket.close()
