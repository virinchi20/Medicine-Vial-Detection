import os
import cv2

def load_annotations(annotation_file):
    """Load YOLO annotations from a text file."""
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    return [list(map(float, line.strip().split()[1:])) for line in lines]

def crop_and_save_images(folder, output_folder):
    """Crop images based on YOLO annotations and save them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            annotation_path = os.path.join(folder, filename.replace('.jpg', '.txt'))
            
            if not os.path.exists(annotation_path):
                print(f"No annotation for {filename}, skipping...")
                continue
            
            # Load image
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            # Load annotations
            annotations = load_annotations(annotation_path)
            
            for i, (x_center, y_center, bbox_width, bbox_height) in enumerate(annotations):
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - bbox_width / 2) * width)
                y1 = int((y_center - bbox_height / 2) * height)
                x2 = int((x_center + bbox_width / 2) * width)
                y2 = int((y_center + bbox_height / 2) * height)
                
                # Crop image
                cropped_img = image[y1:y2, x1:x2]
                
                # Save cropped image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_img)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    folder = "data"  # Folder containing images and annotations
    output_folder = "cropped_images"  # Folder to save cropped images
    
    crop_and_save_images(folder, output_folder)