# se7-micro-server-running-YOLOv5
使用算丰盒子运行YOLOv5模型，监控多个视频流并应用多个模型

# 简介
此项目基于sophon/yolov5的示例进行修改。
yolov5_bmcv.py是一个基于 Sophon Sail 和 OpenCV 的视频处理和分析工具，主要使用 YOLOv5 模型进行目标检测。它能够从指定的 RTSP 流中读取视频数据，进行实时的目标检测，并将结果保存为视频文件。
实现一个视频流被多个BModel串行的检测，并且应用对应的多个算法。
 
start.sh使用了多进程的方法，在se7盒子上可以同时检测多个视频流


