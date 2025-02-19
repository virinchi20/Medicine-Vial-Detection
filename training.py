from ultralytics import YOLO
import os
import sys

model = YOLO("yolo11n.pt")


data = os.path.join(os.getcwd(), "final_data/data.yaml")


results = model.train(data=data, epochs=10, imgsz=640, device="mps", save=True).