from ultralytics import YOLO
import os

model = YOLO("yolo11n.pt")

data = "C:/Users/CAE-USER/Desktop/virinchi/Medicine-Vial-Detection/final_data/data.yaml"

results = model.train(data=data, epochs=10, imgsz=640, batch=1024)