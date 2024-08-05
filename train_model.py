from ultralytics import YOLO
import multiprocessing
multiprocessing.freeze_support()

model = YOLO("yolov8n.yaml")
model.train(data=".\\datasets\\data.yaml", epochs=42, imgsz=640)


