# se7-micro-server-running-YOLOv5
使用算丰盒子运行YOLOv5模型，监控多个视频流并应用多个模型

# 简介
此项目基于sophon/yolov5的示例进行修改。
yolov5_bmcv.py是一个基于 Sophon Sail 和 OpenCV 的视频处理和分析工具，主要使用 YOLOv5 模型进行目标检测。
它能够从指定的 RTSP 流中读取视频数据，进行实时的目标检测，并将结果保存为视频文件。并且实现了一个视频流被多个BModel串行的检测，并且应用对应的多个算法。
 
start.sh使用了多进程的方法，在se7盒子上可以同时检测多个视频流。此程序运行60秒后会自动终止运行的所有进程。  

best.pt是检测灭火器的模型  

yolov5s_v6.1_3output_fp32_1b_extinguisher.bmodel是量化后的检测灭火器的模型  

# 运行环境和方法

1，准备已经配置好环境的se7盒子   
2，运行命令把代码复制进入盒子  
scp yolov5_bmcv.py linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/python/  
scp start.sh linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/python/  
scp yolov5s_v6.1_3output_fp32_1b_extinguisher.bmodel linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/models/BM1684X  
3，进入盒子  
ssh linaro@192.168.31.50  
4，进入路径  
cd /data/sophon-demo/sample/YOLOv5  
5，给予权限  
chmod +x python/start.sh  
6，运行代码  
./python/start.sh  
7，等待1min  
8，在开始的地方查看视频保存路径  
![image](https://github.com/YC2232/se7-micro-server-running-YOLOv5/assets/143778084/2b1ebdc9-89c6-4e98-967f-d727e838e2e3)  
9，把视频从盒子复制到本机上  
scp linaro@192.168.31.50:/data/sophon-demo/sample/YOLOv5/results/[你跑出来的结果] C:\Users\28482\Desktop\video-stream\test  








