import sys
import os
import cv2
import torch
import numpy as np

# 将 yolov5 和 deep_sort 的路径添加到 sys.path
yolov5_path = os.path.abspath("yolov5")
deep_sort_path = os.path.abspath("deep_sort")
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)
if deep_sort_path not in sys.path:
    sys.path.append(deep_sort_path)

# 现在可以导入 yolov5 和 deep_sort 的模块
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# 导入 DeepSORT 相关模块
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# 加载 YOLOv5 模型
device = select_device('')  # 选择设备（CPU 或 GPU）
model = attempt_load('weights/yolov5s.pt')  # 加载 YOLOv5 模型
model.to(device)  # 将模型移动到指定设备
stride = int(model.stride.max())  # 模型步长
names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称

# 初始化 DeepSORT
max_cosine_distance = 0.2  # 余弦距离阈值
nn_budget = 100  # 特征缓存大小
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# 打开视频文件
video_path = 'data/video.mp4'
cap = cv2.VideoCapture(video_path)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 调整图像尺寸为 32 的倍数
    img = letterbox(frame, new_shape=(640, 640), stride=stride, auto=True)[0]  # 调整尺寸
    img = img.transpose((2, 0, 1))  # 将 [H, W, C] 转换为 [C, H, W]
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 归一化
    img = img.unsqueeze(0)  # 添加批次维度 [1, C, H, W]

    # YOLOv5 目标检测
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    # 处理检测结果
    detections = []
    for i, det in enumerate(pred):
        if det is not None and len(det):
            # 使用 scale_boxes 替换 scale_coords
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # 提取检测框和置信度
            bboxes = det[:, :4].cpu().numpy()  # [x1, y1, x2, y2]
            confidences = det[:, 4].cpu().numpy()  # 置信度
            classes = det[:, 5].cpu().numpy()  # 类别 ID


            # 将 YOLOv5 的检测结果转换为 DeepSORT 的 Detection 对象
            for bbox, confidence, cls in zip(bboxes, confidences, classes):
                tlwh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])  # 转换为 [x, y, w, h]
                feature = np.random.rand(128)  # 随机生成特征向量（实际应用中应使用 ReID 模型生成）
                detections.append(Detection(tlwh, confidence, feature))

    # 更新 DeepSORT 追踪器
    tracker.predict()
    tracker.update(detections)


    # 直接使用 OpenCV 绘制检测框和追踪结果
    for detection in detections:
        bbox = detection.tlwh
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制检测框

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制追踪框
        cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # 绘制追踪 ID

    # 显示结果
    cv2.imshow('YOLOv5 + DeepSORT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()