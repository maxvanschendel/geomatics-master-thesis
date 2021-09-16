from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class DataException(Exception):
    pass

class GenericSensorData(ABC):
    '''Data that is shared between all sensor captures. '''

    @abstractmethod
    def __init__(self, agent: str, time: int, data: np.array):
        self.agent = agent
        self.time = time
        self.data = data

    @abstractmethod
    def __str__(self):
        return f"agent: {self.agent}\n" \
                "time: {self.time}\n" \
                "data: {self.data}"

    @abstractmethod
    def is_valid(self):
        return  None not in (self.agent, self.time, self.data) and \
                self.time > 0 and \
                self.agent != ""


class IMU(GenericSensorData):
    '''6DOF IMU sensor data'''

    shape = (2, 3)

    def __init__(self, agent: str, time: int, data: np.array):
        super().__init__(agent, time, data)

        if not self.is_valid():
            raise DataException(f"IMU data is not valid. \n{str(self)}")

    def __str__(self):
        return super().__str__()

    def is_valid(self):
        return super().is_valid() and self.data.shape == IMU.shape


class FrameType(Enum):
    RGB = 1
    RGBIR = 2
    RGBD = 3
    IR = 4
    IRD = 5
    D = 6
    RGBDI = 7

    dim = {RGB: 3, RGBIR: 4, RGBD: 4, IR: 1, IRD: 2, D: 1, RGBDI: 5}


class Frame:
    def __init__(self, agent: str, time: int, data: np.array, type: FrameType):
        super().__init__(agent, time, data)
        self.type = type

        if not self.is_valid():
            raise DataException(f"Frame data is not valid. \n{str(self)}")

    def __str__(self):
        return f"{super().__str__()}\n" \
                "type: {self.type}"

    def is_valid(self):
        return  super().is_valid() and \
                self.data.shape[2] == FrameType.dim[type] and \
                self.type is not None


