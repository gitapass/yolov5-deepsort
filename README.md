# yolov5-deepsort
yolov5结合deepsort实现目标追踪<br>
这是一个将yolov5和deepsort快速结合的代码，可以实现对于目标的识别追踪<br>
## HOW TO USE
下载权重文件
```
curl -o .\weights\ckpt.t7 "https://drive.usercontent.google.com/download?id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN&export=download&authuser=0&confirm=t&uuid=9d009b8f-b64a-406b-8a46-7c33e6004cd8&at=AIrpjvNGj0j20tZR4W-wUIRnvVVm:1736751393900"
```
```
curl -o .\weights\yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```
将自己的video.mp4测试视频放入data文件夹里<br>

安装依赖
```
pip install -r requirements.txt
```
## RUN

```
python main.py --video data/video.mp4
```
## Disclaimer
This project is using code from:<br>
ultralytics/yolov5 https://github.com/ultralytics/yolov5<br>
nwojke/deep_sort https://github.com/nwojke/deep_sort<br>
