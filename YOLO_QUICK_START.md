# YOLO Sliding Counter - Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from yolo_sliding_counter import YOLOSlidingCounter

# Initialize
counter = YOLOSlidingCounter(model_path="yolov8n.pt")

# Count objects
result = counter.count_kernels_with_yolo("image.jpg")

# Get results
print(f"Found {result.total_detections} objects")
print(f"Processing time: {result.processing_time:.2f}s")
```

## Key Features

- ✅ ROI visualization with red borders
- ✅ Intermediate image saving
- ✅ Auto-scaling for large images
- ✅ Batch processing support
- ✅ Comprehensive error handling
- ✅ Detailed statistics
- ✅ Progress tracking

## API Quick Reference

### YOLOSlidingCounter

```python
counter = YOLOSlidingCounter(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,  # Detection confidence
    nms_threshold=0.45,         # Non-max suppression
    window_size=(640, 640),     # Sliding window size
    overlap_ratio=0.2,          # Window overlap
    output_dir="output"         # Output directory
)
```

### count_kernels_with_yolo

```python
result = counter.count_kernels_with_yolo(
    image_path="image.jpg",
    roi=(x1, y1, x2, y2),      # Optional ROI
    save_intermediate=True,     # Save cropped images
    save_visualizations=True    # Save result images
)
```

### DetectionResult

```python
result.total_detections      # Number of objects found
result.detection_boxes       # List of bounding boxes
result.detection_scores      # Confidence scores
result.processing_time       # Time taken (seconds)
result.window_count          # Number of windows processed
result.roi_image_path        # Path to ROI annotation
result.cropped_image_path    # Path to cropped image
result.result_image_path     # Path to result image
```

## Examples

See `examples/yolo_counter_examples.py` for 5 detailed examples:
1. Basic counting
2. ROI detection
3. Batch processing
4. Custom parameters
5. Detailed detection info

## Testing

```bash
python tests/test_yolo_sliding_counter.py
```

## Documentation

Full documentation: `docs/YOLO_SLIDING_COUNTER_README.md`
