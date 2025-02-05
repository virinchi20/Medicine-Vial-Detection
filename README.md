# Bottle Detection Data Collector

A Python script for collecting and annotating bottle detection data using YOLOv8 and OpenCV. This tool automatically captures images of bottles, generates bounding box annotations, and saves associated metadata.

## Features

- Real-time bottle detection using YOLOv8
- Automatic data collection and annotation
- YOLO format annotation generation
- Metadata storage for each captured image
- Customizable output directory structure
- Support for multiple classification categories

## Requirements

- Python 3.x
- ultralytics
- OpenCV (cv2)
- NumPy
- Webcam or camera device

## Installation

1. Install the required Python packages:
```bash
pip install ultralytics opencv-python numpy
```

2. Download the YOLOv8x model (if not already present)

## Usage

Run the script from the command line with a classification category argument:

```bash
python script_name.py <category>
```

Example:
```bash
python script_name.py class1
```

### Parameters

You can modify these parameters in the script:

- `model_path`: Path to the YOLO model (default: 'yolov8x.pt')
- `camera_id`: Camera device ID (default: 0)
- `output_dir`: Base directory for saved data (default: 'data')
- `classifier`: Subdirectory name for specific category (provided as command line argument)

## Directory Structure

The script creates the following directory structure:
```
data/
├── <category>/
│   ├── 0.jpg         # Image files
│   ├── 0.txt         # Annotation files
│   ├── 1.jpg
│   ├── 1.txt
│   └── ...
└── metadata/
    ├── 0.json        # Metadata files
    ├── 1.json
    └── ...
```

## File Formats

### Image Files
- Format: JPEG
- Naming: Sequential numbering (0.jpg, 1.jpg, etc.)
- Location: `data/<category>/`

### Annotation Files
- Format: YOLO text format (class x_center y_center width height)
- Naming: Matches image files (0.txt, 1.txt, etc.)
- Location: `data/<category>/`

### Metadata Files
- Format: JSON
- Content:
  - Timestamp
  - Original image dimensions
  - Bounding box data
- Location: `data/metadata/`

## Controls

- Press 'q' to quit the data collection process

## Notes

- The script uses COCO class ID 39 for bottle detection
- All bounding box coordinates are normalized (0-1 range)
- Images are saved in their original resolution
- The script validates bounding boxes before saving

## Error Handling

The script includes error handling for:
- Camera initialization failures
- Frame capture errors
- Invalid bounding boxes
- File saving errors

## Contributing

Feel free to submit issues and enhancement requests!