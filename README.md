# se7-micro-server-running-YOLOv5
使用算丰盒子运行YOLOv5模型，监控多个视频流并应用多个模型

# 简介
此项目基于sophon/yolov5的示例进行修改。
yolov5_bmcv.py是一个基于 Sophon Sail 和 OpenCV 的视频处理和分析工具，主要使用 YOLOv5 模型进行目标检测。它能够从指定的 RTSP 流中读取视频数据，进行实时的目标检测，并将结果保存为视频文件。并且实现了一个视频流被多个BModel串行的检测，并且应用对应的多个算法。
 
start.sh使用了多进程的方法，在se7盒子上可以同时检测多个视频流。此程序运行60秒后会自动终止运行的所有进程。

# 运行环境和方法

1，准备已经配置好环境的se7盒子 \n
2，运行命令把代码复制进入盒子
scp yolov5_bmcv.py linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/python/
scp start.sh linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/python/
3，进入盒子
ssh linaro@192.168.31.50
4，进入路径
cd /data/sophon-demo/sample/YOLOv5
5，给予权限
chmod +x python/start.sh
6，运行代码
./python/start.sh










