#!/bin/bash

export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python/

python3 python/yolov5_bmcv.py \
    --channelCode a498268d2e354e0a88b3ae1913e83e7d \
    --url http://1.192.171.31:19480/hik/petrol/camera/video/rtsp \
    --dev_id 0 &
PID1=$!

python3 python/yolov5_bmcv.py \
    --channelCode 60054da19e2245919e05b3e01460aa52 \
    --url http://1.192.171.31:19480/hik/petrol/camera/video/rtsp \
    --dev_id 0 &
PID2=$!

sleep 60

kill $PID1 $PID2

wait
