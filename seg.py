from ultralytics import YOLO

model = YOLO('yolov8x-seg')

#result = model.predict('input_videos/a.jpg', save=True)

result = model('input_videos/a.jpg')

result[0].show()

print(result)