from ultralytics import YOLO
import cv2
import os
import time

# Load the YOLO model (update the path to your model if necessary)
model = YOLO('yolov8x.pt')
#model.to('cpu')

# Open the webcam using DirectShow backend
video = cv2.VideoCapture(0)

classifier = 0

output_dir = 'data/' + str(classifier) + "/"
os.makedirs(output_dir, exist_ok=True)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

data_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference
    results = model(frame)
    #print(results)

    for r in results:
        #if int(r.boxes.cls[0]) == 39:
        if int(r.boxes.cls.cpu().numpy()[0]) == 39:
        #int(box.cls.cpu().numpy()[0]) == 39
        
            print("bottle found")

            #saving the data (image)
            img_name = str(data_count)
            cv2.imwrite(os.path.join(output_dir, f'{img_name}.jpg'), frame)

            #saving the annotated data file
            with open(os.path.join(output_dir, f'{img_name}.txt'), 'w') as file:
                xywhn = r.boxes.xywhn.cpu().numpy()
                print(r.boxes.xywhn.cpu().numpy())
                file.write("0 " + str(xywhn[0][0]) + " " + str(xywhn[0][1]) + " " + str(xywhn[0][2]) + " " + str(xywhn[0][3]))

    # Draw the detection results on the frame
    #annotated_frame = results.render()[0]
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('frame', annotated_frame)
    #cv2.imshow('frame', frame)

    data_count += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
video.release()
cv2.destroyAllWindows()
