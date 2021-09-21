from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class DataException(Exception):
    pass


class GenericSensorData(ABC):
    '''Data that is shared between all sensor captures.'''

    @abstractmethod
    def __init__(self, agent: str, time: int, data: np.array):
        self.agent = agent
        self.time = time
        self.data = data

    @abstractmethod
    def __str__(self):
        return f"agent: {self.agent}\n" \
               f"time: {self.time}\n" \
               f"data: {self.data}"

    @abstractmethod
    def is_valid(self):
        return  self.time > 0 and \
                self.agent != ""


class IMU(GenericSensorData):
    '''6DOF IMU sensor data.'''

    shape = (2, 3)  # First row accelerometer, second row gyro.

    def __init__(self, agent: str, time: int, data: np.array):
        super().__init__(agent, time, data)

        if not self.is_valid():
            raise DataException(f"IMU data is not valid. \n{str(self)}")

    def __str__(self):
        return super().__str__()

    def is_valid(self):
        return super().is_valid() and self.data.shape == IMU.shape


class FrameType(Enum):
    '''The different frame types that are supported by the application.'''

    RGB = 1
    RGBD = 3
    IR = 4
    D = 6


class Frame(GenericSensorData):
    '''Array of sensor data, can be one of different types defined in FrameType enum.'''

    # TODO: add intrinsic parameters

    dim = {FrameType.RGB: 3, FrameType.RGBD: 4, FrameType.IR: 1, FrameType.D: 1}

    def __init__(self, agent: str, time: int, data: np.array, type: FrameType):
        super().__init__(agent, time, data)
        self.type = type

        if not self.is_valid():
            raise DataException(f"Frame data is not valid. \n{str(self)}")

    def __str__(self):
        return f"{super().__str__()}\n" \
               f"type: {self.type}"

    def is_valid(self):
        return  super().is_valid() and \
                self.type is not None
