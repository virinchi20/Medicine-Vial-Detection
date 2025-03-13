from ultralytics import YOLO
import os
import cv2


def displayText(text):
    #text = "OpenCV Video Window"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    thickness = 2
    position = (50, 50)  # (x, y) coordinates
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    

#object_detection_model = YOLO("trained_models_from_colab/models/best3.pt")
object_detection_model = YOLO("yolo11n.pt")
classification_model = YOLO("trained_models_from_colab/models/classify/best.pt")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open vidio")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    
    results = object_detection_model.predict(frame)
    for r in results:
        if int(r.boxes.cls.cpu().numpy()[0]) == 39:
            image = frame
            height, width, _ = image.shape
            xywhn = r.boxes.xywhn.cpu().numpy()
            print(xywhn)
            x_center, y_center, bbox_width, bbox_height = xywhn[0]
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)

            cropped_bottle_img = image[y1:y2, x1:x2]
            final_res = classification_model.predict(cropped_bottle_img)
            for r in final_res:
                if r.probs.top1 == 0:
                    print("Detected Fentynal")
                    displayText("Fentynal")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif r.probs.top1 == 1:
                    print("Detected Midazolm")
                    displayText("Midazolem")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                else:
                    pass


            #print(final_res)

    annotated_frame = results[0].plot()
    #cv2.imshow('frame', annotated_frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()