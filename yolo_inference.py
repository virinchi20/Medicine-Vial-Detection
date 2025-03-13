from ultralytics import YOLO

#model = YOLO('yolov8x.pt')
model = YOLO("trained_models_from_colab/models/classify/best.pt")

#results = model.predict('input_videos/08fd33_0.mp4', save=True, stream=True)
results = model.predict('input_videos/1.jpeg')

for r in results:
    print(r.probs.top1)



"""
print("=============================================================")
for box in results[0].boxes:
    #print(box)
    #print(int(box.cls.numpy()[0])==39)
    print(int(box.cls.cpu().numpy()[0]) == 39)
    print(box.xywhn)
    #value = box.xywhn.item()
    #print(value[0])
    print(box.xywhn.cpu().numpy())
"""


