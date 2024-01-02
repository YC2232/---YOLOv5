# se7-micro-server-running-YOLOv5
使用算丰盒子运行YOLOv5模型，监控多个视频流并应用多个模型

# 简介
此项目基于sophon/yolov5的示例进行修改。
yolov5_bmcv.py实现一个视频流被多个BModel串行的检测，并且应用对应的多个算法。
start.sh使用了多进程的形式
