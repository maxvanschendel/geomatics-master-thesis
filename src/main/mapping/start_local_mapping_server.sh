export DISPLAY=192.168.1.208:0
rosrun ORB_SLAM3 Mono ~/Dev/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/Dev/ORB_SLAM3/Examples/Monocular/EuRoC.yaml &
pkill -9 -f sensor_capture_server.py &
python3 sensor_capture_server.py &
