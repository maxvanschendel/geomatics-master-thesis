FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

RUN apt update
RUN apt -y install gnupg gnupg2 gnupg1 curl software-properties-common

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list
RUN curl http://repo.ros2.org/repos.key | apt-key add 
RUN apt update

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt install -y ros-noetic-ros-base
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN tail ~/.bashrc

RUN apt -y install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

RUN rosdep init
RUN rosdep update

RUN apt -y install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt -y install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libjasper-dev
RUN apt -y install libeigen3-dev libglew-dev libboost-all-dev libssl-dev

RUN apt -y install python3-pip
RUN pip3 install opencv-python

ADD src/ /app/src/
ADD Thirdparty/ /app/thirdparty/

RUN apt-get -y install ros-noetic-cv-bridge ros-noetic-vision-opencv
RUN rosdep update

ENTRYPOINT roscore && export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages && python3 /app/src/image_capture_server.py && /bin/bash

