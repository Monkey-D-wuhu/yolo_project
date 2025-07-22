from ultralytics import YOLO
import cv2
import onnx
model = YOLO("yolov8n.pt")
model.export(format="onnx",simplify=True)
# result = model("dataset/images/val2017/000000000139.jpg")
# cv2.imshow("result", result[0].plot())
# cv2.waitKey(0)