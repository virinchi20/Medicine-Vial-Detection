from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8x.pt')

# Open the webcam
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open  nawQQ 1e4d    qadsvideo.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference
    results = model(frame)

    # Draw the detection results on the frame
    #annotated_frame = results.render()[0]

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
video.release()
cv2.destroyAllWindows() 