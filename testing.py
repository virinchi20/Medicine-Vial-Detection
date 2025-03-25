from ultralytics import YOLO
import os
import cv2


model = YOLO("models/detection/best.pt")

# temp_model = model

# temp_model.model.names = {0: 'medicine 1',
#                  1: 'medicine 2',
#                  3: 'medicine 3',
#                  4: 'medicine 4'}

# model.names = {0: 'medicine 1',
#                1: 'medicine 2',
#                2: 'medicine 3',
#                3: 'medicine 4'}

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open vidio")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    results = model.predict(frame, conf=0.85)
    annotated_frame = results[0].plot()

    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()