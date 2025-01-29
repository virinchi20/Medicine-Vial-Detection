from ultralytics import YOLO

model = YOLO('yolov8x')


#results = model.predict('input_videos/08fd33_0.mp4', save=True, stream=True)
results = model.predict('input_videos/a.jpg', save=True)

print(results)


print("=============================================================")
for box in results[0].boxes:
    #print(box)
    print(int(box.cls.numpy()[0])==39)
    print(box.xywhn)
    #value = box.xywhn.item()
    #print(value[0])
    print(box.xywhn.numpy())


