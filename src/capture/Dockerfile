FROM ros:melodic

# install ros package
RUN apt-get update && apt-get install -y
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get install ros-melodic-catkin

RUN sudo apt install -y apt-transport-https ca-certificates curl software-properties-common wget
RUN sudo curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add - 
RUN sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/multiarch/prod
RUN sudo apt-get update

# Install Azure Kinect DK SDK
RUN sudo ACCEPT_EULA=Y apt-get install -y libk4a1.4 libk4a1.4-dev k4a-tools

# Download ROS Azure Kinect DK driver
ENV ROS_WS /home/catkin_ws
RUN mkdir -p $ROS_WS/src
WORKDIR $ROS_WS
RUN git -C src clone https://github.com/microsoft/Azure_Kinect_ROS_Driver.git
	  
# install ros package dependencies
RUN apt-get update && rosdep update

WORKDIR $ROS_WS/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace $ROS_WS/src'

WORKDIR $ROS_WS
RUN sudo apt install -y ros-melodic-theora-image-transport
RUN sudo apt install -y ros-melodic-tf2
RUN sudo apt install -y ros-melodic-tf2-ros
RUN sudo apt install -y ros-melodic-tf2-geometry-msgs
RUN sudo apt install -y ros-melodic-camera-info-manager
RUN sudo apt install -y ros-melodic-angles
RUN sudo apt install -y ros-melodic-xacro
RUN sudo apt install -y ros-melodic-rgbd-launch
RUN sudo apt install -y ros-melodic-joint-state-publisher
RUN sudo apt install -y ros-melodic-robot-state-publisher

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd $ROS_WS; catkin_make --force-cmake; catkin_make install'
RUN /bin/bash -c  'echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc'
RUN /bin/bash -c  'echo "source $ROS_WS/devel/setup.sh" >> ~/.bashrc'
ENTRYPOINT /bin/bash