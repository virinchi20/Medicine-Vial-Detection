from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from datetime import datetime
import sys

class BottleDataCollector:
    def __init__(self, model_path='yolov8x.pt', camera_id=0, output_dir='data', classifier='0' ):
        """
        Initialize the bottle data collector
        """
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.data_count = 713
        self.frame_count = 0  # To track frames
        
        # Create output directories
        self.image_dir = os.path.join(output_dir, classifier)
        self.label_dir = os.path.join(output_dir, classifier)
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    

    def validate_bbox(self, bbox):
        """
        Validate if bounding box coordinates are within valid range
        """
        x, y, w, h = bbox
        return (0 <= x <= 1) and (0 <= y <= 1) and (w > 0) and (h > 0)

    def save_metadata(self, frame_shape, bbox_data):
        """
        Save metadata for the collected data
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_dimensions': {
                'height': frame_shape[0],
                'width': frame_shape[1],
                'channels': frame_shape[2]
            },
            'bbox_data': bbox_data.tolist()
        }
        
        metadata_path = os.path.join(self.metadata_dir, f'{self.data_count}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def collect_data(self):
        """
        Main data collection loop
        """
        video = cv2.VideoCapture(self.camera_id)
        
        if not video.isOpened():
            raise RuntimeError(f"Error: Could not open video capture device {self.camera_id}")
        
        print("Starting data collection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = video.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                self.frame_count += 1

                # Perform inference
                results = self.model(frame)
                
                # Process only on 5th or 8th frame
                if self.frame_count % 5 == 0 or self.frame_count % 10 == 0:
                    results = self.model(frame)
                    
                    for r in results:
                        # Filter detections for class ID 39 (bottle)
                        mask = r.boxes.cls == 39  
                        if mask.any():
                            xywhn = r.boxes.xywhn[mask].cpu().numpy()
                            adjusted_bboxes = []
                            for bbox in xywhn:
                                x, y, w, h = bbox
                                w = max(w * 0.96, 0.06)  # Reduce width slightly but keep it reasonable
                                h = max(h * 0.96, 0.06)  # Reduce height slightly but keep it reasonable
                                adjusted_bboxes.append([x, y, w, h])
                            adjusted_bboxes = np.array(adjusted_bboxes)    
                                                        
                            
                            if not all(self.validate_bbox(bbox) for bbox in adjusted_bboxes):
                                print(f"Warning: Invalid bounding box detected in frame {self.data_count}")
                                continue

                            try:
                                # Save original image
                                image_path = os.path.join(self.image_dir, f'{self.data_count}.jpg')
                                cv2.imwrite(image_path, frame)

                                # Save annotations
                                label_path = os.path.join(self.label_dir, f'{self.data_count}.txt')
                                with open(label_path, 'w') as file:
                                    for bbox in adjusted_bboxes:
                                        file.write(f"1 {' '.join(map(str, bbox))}\n")

                                # Save metadata
                                self.save_metadata(frame.shape, adjusted_bboxes)

                                self.data_count += 1
                                print(f"Saved data sample {self.data_count}")

                            except Exception as e:
                                print(f"Error processing frame {self.data_count}: {str(e)}")
                                continue
                
                # Display the frame with detections
                annotated_frame = results[0].plot()
                cv2.imshow('Bottle Detection', annotated_frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            video.release()
            cv2.destroyAllWindows()
            print(f"\nData collection completed. Total samples collected: {self.data_count}")

if __name__ == "__main__":
    try:
        collector = BottleDataCollector(
            model_path='yolov8x.pt',
            camera_id=0,
            output_dir='data',
            classifier=sys.argv[1]
        )
        collector.collect_data()
    except Exception as e:
        print(f"Error during data collection: {str(e)}")