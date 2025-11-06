"""
YOLO Sliding Window Counter - Enhanced Version

This module implements an improved YOLO-based object counting system using sliding window approach.
It includes enhanced image processing, error handling, performance optimization, and detailed statistics.

Author: Enhanced YOLO Counter Team
Version: 2.0
Date: 2025-11-06

Features:
- ROI (Region of Interest) visualization with red borders
- Intermediate image saving (cropped ROI)
- Final detection result visualization
- Comprehensive error handling for image I/O and model loading
- Input parameter validation
- Large image auto-scaling
- Batch processing with progress display
- Memory optimization
- Detailed logging and debugging
- Statistics tracking (detection count, processing time, etc.)
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_WINDOW_SIZE = (640, 640)
DEFAULT_OVERLAP_RATIO = 0.2
MAX_IMAGE_DIMENSION = 4096
AUTO_SCALE_THRESHOLD = 2048
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class to store detection results and statistics"""
    total_detections: int
    detection_boxes: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    detection_scores: List[float]
    detection_classes: List[int]
    processing_time: float
    image_size: Tuple[int, int]
    window_count: int
    roi_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    result_image_path: Optional[str] = None


class YOLOSlidingCounter:
    """
    Enhanced YOLO-based sliding window counter for object detection and counting.
    
    This class provides comprehensive functionality for detecting and counting objects
    in images using YOLO model with sliding window approach.
    
    Attributes:
        model: YOLO model instance
        confidence_threshold: Minimum confidence score for detections
        nms_threshold: Non-maximum suppression threshold
        window_size: Size of sliding window (width, height)
        overlap_ratio: Overlap ratio between adjacent windows
        output_dir: Directory to save output images
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        output_dir: str = "output"
    ):
        """
        Initialize YOLO Sliding Counter.
        
        Args:
            model_path: Path to YOLO model weights file
            confidence_threshold: Confidence threshold for detections (0-1)
            nms_threshold: NMS threshold for duplicate removal (0-1)
            window_size: Size of sliding window as (width, height)
            overlap_ratio: Overlap ratio between windows (0-1)
            output_dir: Directory to save output images
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        logger.info("Initializing YOLO Sliding Counter...")
        
        # Validate parameters
        self._validate_init_parameters(
            model_path, confidence_threshold, nms_threshold,
            window_size, overlap_ratio
        )
        
        # Store configuration
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        logger.info(f"YOLO Sliding Counter initialized successfully")
        logger.info(f"Configuration: confidence={confidence_threshold}, "
                   f"nms={nms_threshold}, window_size={window_size}, "
                   f"overlap={overlap_ratio}")
    
    def _validate_init_parameters(
        self,
        model_path: str,
        confidence_threshold: float,
        nms_threshold: float,
        window_size: Tuple[int, int],
        overlap_ratio: float
    ) -> None:
        """Validate initialization parameters"""
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("model_path must be a non-empty string")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        
        if not isinstance(window_size, tuple) or len(window_size) != 2:
            raise ValueError("window_size must be a tuple of (width, height)")
        
        if window_size[0] <= 0 or window_size[1] <= 0:
            raise ValueError("window_size dimensions must be positive")
        
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be between 0 and 1 (exclusive)")
    
    def _load_model(self, model_path: str):
        """
        Load YOLO model from file.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded YOLO model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading YOLO model from: {model_path}")
            
            # Try to import ultralytics YOLO
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                logger.info("Model loaded successfully using ultralytics")
                return model
            except ImportError:
                logger.warning("ultralytics not available, trying opencv")
                
                # Fallback to OpenCV DNN
                import cv2
                model = cv2.dnn.readNetFromDarknet(
                    model_path.replace('.pt', '.cfg'),
                    model_path.replace('.pt', '.weights')
                )
                logger.info("Model loaded successfully using OpenCV DNN")
                return model
                
        except Exception as e:
            error_msg = f"Failed to load YOLO model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_image_path(self, image_path: str) -> None:
        """Validate image path and format"""
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("image_path must be a non-empty string")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {file_ext}. "
                f"Supported formats: {SUPPORTED_IMAGE_FORMATS}"
            )
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image with error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            ValueError: If image cannot be loaded
        """
        try:
            import cv2
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            logger.info(f"Image loaded: {image_path}, shape: {image.shape}")
            return image
            
        except Exception as e:
            error_msg = f"Error loading image {image_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _save_image(self, image: np.ndarray, output_path: str) -> str:
        """
        Save image with error handling.
        
        Args:
            image: Image to save
            output_path: Path to save image
            
        Returns:
            Path where image was saved
            
        Raises:
            IOError: If image cannot be saved
        """
        try:
            import cv2
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            success = cv2.imwrite(output_path, image)
            
            if not success:
                raise IOError(f"Failed to save image to: {output_path}")
            
            logger.info(f"Image saved successfully: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving image to {output_path}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
    
    def _auto_scale_image(
        self,
        image: np.ndarray,
        max_dimension: int = AUTO_SCALE_THRESHOLD
    ) -> Tuple[np.ndarray, float]:
        """
        Automatically scale large images for better processing.
        
        Args:
            image: Input image
            max_dimension: Maximum allowed dimension
            
        Returns:
            Tuple of (scaled_image, scale_factor)
        """
        import cv2
        
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim <= max_dimension:
            logger.info("Image size is acceptable, no scaling needed")
            return image, 1.0
        
        scale_factor = max_dimension / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        logger.info(f"Scaling image from {width}x{height} to {new_width}x{new_height} "
                   f"(scale factor: {scale_factor:.2f})")
        
        scaled_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        
        return scaled_image, scale_factor
    
    def _draw_roi_annotation(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Draw ROI annotation with red border.
        
        Args:
            image: Input image
            roi: ROI coordinates as (x1, y1, x2, y2)
            
        Returns:
            Image with ROI annotation
        """
        import cv2
        
        annotated = image.copy()
        x1, y1, x2, y2 = roi
        
        # Draw red border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add label
        label = f"ROI: {x2-x1}x{y2-y1}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        
        logger.debug(f"Drew ROI annotation: {roi}")
        return annotated
    
    def _crop_roi(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop image to ROI.
        
        Args:
            image: Input image
            roi: ROI coordinates as (x1, y1, x2, y2)
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = roi
        cropped = image[y1:y2, x1:x2]
        logger.debug(f"Cropped ROI: {roi}, shape: {cropped.shape}")
        return cropped
    
    def _draw_detections(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        classes: List[int]
    ) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            scores: List of confidence scores
            classes: List of class IDs
            
        Returns:
            Image with detections drawn
        """
        import cv2
        
        result = image.copy()
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class {cls}: {score:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw count
        count_text = f"Total Detections: {len(boxes)}"
        cv2.putText(
            result,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        logger.debug(f"Drew {len(boxes)} detections on image")
        return result
    
    def _generate_sliding_windows(
        self,
        image_shape: Tuple[int, int],
        window_size: Tuple[int, int],
        overlap_ratio: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding window coordinates.
        
        Args:
            image_shape: Image shape as (height, width)
            window_size: Window size as (width, height)
            overlap_ratio: Overlap ratio between windows
            
        Returns:
            List of window coordinates as (x1, y1, x2, y2)
        """
        height, width = image_shape[:2]
        win_w, win_h = window_size
        
        step_w = int(win_w * (1 - overlap_ratio))
        step_h = int(win_h * (1 - overlap_ratio))
        
        windows = []
        
        for y in range(0, height - win_h + 1, step_h):
            for x in range(0, width - win_w + 1, step_w):
                windows.append((x, y, x + win_w, y + win_h))
        
        # Add edge windows if needed
        if width > win_w:
            for y in range(0, height - win_h + 1, step_h):
                windows.append((width - win_w, y, width, y + win_h))
        
        if height > win_h:
            for x in range(0, width - win_w + 1, step_w):
                windows.append((x, height - win_h, x + win_w, height))
        
        # Add corner window
        if width > win_w and height > win_h:
            windows.append((width - win_w, height - win_h, width, height))
        
        # Remove duplicates
        windows = list(set(windows))
        
        logger.info(f"Generated {len(windows)} sliding windows")
        return windows
    
    def _detect_in_window(
        self,
        image: np.ndarray,
        window: Tuple[int, int, int, int]
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
        """
        Perform detection in a single window.
        
        Args:
            image: Full image
            window: Window coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (boxes, scores, classes)
        """
        x1, y1, x2, y2 = window
        window_img = image[y1:y2, x1:x2]
        
        try:
            # Use ultralytics YOLO
            results = self.model.predict(
                window_img,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False
            )
            
            boxes = []
            scores = []
            classes = []
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Convert to absolute coordinates
                    x1_rel, y1_rel, x2_rel, y2_rel = box.xyxy[0].cpu().numpy()
                    boxes.append((
                        int(x1_rel + x1),
                        int(y1_rel + y1),
                        int(x2_rel + x1),
                        int(y2_rel + y1)
                    ))
                    scores.append(float(box.conf[0].cpu().numpy()))
                    classes.append(int(box.cls[0].cpu().numpy()))
            
            return boxes, scores, classes
            
        except Exception as e:
            logger.warning(f"Detection failed in window {window}: {str(e)}")
            return [], [], []
    
    def _apply_nms(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        classes: List[int],
        threshold: float
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            boxes: List of bounding boxes
            scores: List of confidence scores
            classes: List of class IDs
            threshold: NMS threshold
            
        Returns:
            Filtered (boxes, scores, classes)
        """
        if len(boxes) == 0:
            return [], [], []
        
        try:
            import cv2
            
            # Convert to numpy arrays
            boxes_array = np.array(boxes)
            scores_array = np.array(scores)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_array.tolist(),
                scores_array.tolist(),
                self.confidence_threshold,
                threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return (
                    [boxes[i] for i in indices],
                    [scores[i] for i in indices],
                    [classes[i] for i in indices]
                )
            
            return [], [], []
            
        except Exception as e:
            logger.warning(f"NMS failed: {str(e)}, returning original detections")
            return boxes, scores, classes
    
    def count_kernels_with_yolo(
        self,
        image_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        save_intermediate: bool = True,
        save_visualizations: bool = True
    ) -> DetectionResult:
        """
        Count kernels/objects in image using YOLO with sliding window approach.
        
        This is the main API function that performs complete object counting with
        all enhancements including ROI visualization, intermediate image saving,
        and detailed statistics.
        
        Args:
            image_path: Path to input image
            roi: Optional ROI as (x1, y1, x2, y2). If None, uses full image.
            save_intermediate: Whether to save intermediate images
            save_visualizations: Whether to save visualization images
            
        Returns:
            DetectionResult object containing all results and statistics
            
        Raises:
            ValueError: If image path is invalid
            FileNotFoundError: If image file doesn't exist
            IOError: If image processing fails
            
        Example:
            >>> counter = YOLOSlidingCounter('yolov8n.pt')
            >>> result = counter.count_kernels_with_yolo('image.jpg')
            >>> print(f"Detected {result.total_detections} objects")
            >>> print(f"Processing time: {result.processing_time:.2f}s")
        """
        start_time = time.time()
        
        logger.info(f"Starting kernel counting for: {image_path}")
        
        # Validate and load image
        self._validate_image_path(image_path)
        image = self._load_image(image_path)
        
        # Auto-scale if needed
        original_shape = image.shape
        image, scale_factor = self._auto_scale_image(image)
        
        # Get base filename for output
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Handle ROI
        if roi is not None:
            # Scale ROI if image was scaled
            if scale_factor != 1.0:
                roi = tuple(int(x * scale_factor) for x in roi)
            
            # Save ROI annotation image
            roi_image_path = None
            if save_visualizations:
                roi_annotated = self._draw_roi_annotation(image, roi)
                roi_image_path = str(self.output_dir / f"{base_name}_roi_annotated.jpg")
                self._save_image(roi_annotated, roi_image_path)
            
            # Crop to ROI
            processing_image = self._crop_roi(image, roi)
            
            # Save cropped image
            cropped_image_path = None
            if save_intermediate:
                cropped_image_path = str(self.output_dir / f"{base_name}_cropped.jpg")
                self._save_image(processing_image, cropped_image_path)
        else:
            # Use full image
            processing_image = image
            roi_image_path = None
            cropped_image_path = None
            roi = (0, 0, image.shape[1], image.shape[0])
        
        # Generate sliding windows
        windows = self._generate_sliding_windows(
            processing_image.shape,
            self.window_size,
            self.overlap_ratio
        )
        
        # Detect in each window with progress tracking
        all_boxes = []
        all_scores = []
        all_classes = []
        
        logger.info(f"Processing {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            if (i + 1) % 10 == 0 or i == 0:
                progress = (i + 1) / len(windows) * 100
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(windows)})")
            
            boxes, scores, classes = self._detect_in_window(processing_image, window)
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_classes.extend(classes)
        
        logger.info(f"Found {len(all_boxes)} raw detections before NMS")
        
        # Apply NMS to remove duplicates
        final_boxes, final_scores, final_classes = self._apply_nms(
            all_boxes,
            all_scores,
            all_classes,
            self.nms_threshold
        )
        
        logger.info(f"Final detections after NMS: {len(final_boxes)}")
        
        # Draw and save final results
        result_image_path = None
        if save_visualizations:
            result_image = self._draw_detections(
                processing_image,
                final_boxes,
                final_scores,
                final_classes
            )
            result_image_path = str(self.output_dir / f"{base_name}_result.jpg")
            self._save_image(result_image, result_image_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result object
        result = DetectionResult(
            total_detections=len(final_boxes),
            detection_boxes=final_boxes,
            detection_scores=final_scores,
            detection_classes=final_classes,
            processing_time=processing_time,
            image_size=(processing_image.shape[1], processing_image.shape[0]),
            window_count=len(windows),
            roi_image_path=roi_image_path,
            cropped_image_path=cropped_image_path,
            result_image_path=result_image_path
        )
        
        # Log summary
        logger.info(f"Counting completed successfully!")
        logger.info(f"Total detections: {result.total_detections}")
        logger.info(f"Processing time: {result.processing_time:.2f}s")
        logger.info(f"Windows processed: {result.window_count}")
        if roi_image_path:
            logger.info(f"ROI annotation saved to: {roi_image_path}")
        if cropped_image_path:
            logger.info(f"Cropped image saved to: {cropped_image_path}")
        if result_image_path:
            logger.info(f"Result image saved to: {result_image_path}")
        
        return result
    
    def batch_count(
        self,
        image_paths: List[str],
        **kwargs
    ) -> List[DetectionResult]:
        """
        Batch process multiple images.
        
        Args:
            image_paths: List of image paths to process
            **kwargs: Additional arguments to pass to count_kernels_with_yolo
            
        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.count_kernels_with_yolo(image_path, **kwargs)
                results.append(result)
                
                logger.info(f"Image {i+1} completed: {result.total_detections} detections")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                # Continue with next image
                continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(image_paths)} successful")
        
        return results


def main():
    """Example usage of YOLO Sliding Counter"""
    import sys
    
    # Example configuration
    model_path = "yolov8n.pt"  # Replace with actual model path
    image_path = "test_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please download a YOLO model first")
        logger.info("Example: pip install ultralytics && yolo download yolov8n.pt")
        return
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        logger.info("Please provide a valid image path")
        return
    
    try:
        # Initialize counter
        counter = YOLOSlidingCounter(
            model_path=model_path,
            confidence_threshold=0.25,
            nms_threshold=0.45,
            window_size=(640, 640),
            overlap_ratio=0.2,
            output_dir="output/yolo_results"
        )
        
        # Count objects
        result = counter.count_kernels_with_yolo(
            image_path=image_path,
            roi=None,  # or specify ROI as (x1, y1, x2, y2)
            save_intermediate=True,
            save_visualizations=True
        )
        
        # Print results
        print("\n" + "="*50)
        print("YOLO Sliding Window Counter Results")
        print("="*50)
        print(f"Total Detections: {result.total_detections}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Image Size: {result.image_size}")
        print(f"Windows Processed: {result.window_count}")
        print(f"Average Time per Window: {result.processing_time/result.window_count:.3f}s")
        
        if result.roi_image_path:
            print(f"\nROI Annotation: {result.roi_image_path}")
        if result.cropped_image_path:
            print(f"Cropped Image: {result.cropped_image_path}")
        if result.result_image_path:
            print(f"Result Image: {result.result_image_path}")
        
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
