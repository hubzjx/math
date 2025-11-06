"""
Unit tests for YOLO Sliding Counter

This module contains comprehensive tests for the YOLOSlidingCounter class,
including initialization, parameter validation, image processing, and detection.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import from parent package
try:
    from yolo_sliding_counter import (
        YOLOSlidingCounter,
        DetectionResult,
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_NMS_THRESHOLD,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_OVERLAP_RATIO,
        SUPPORTED_IMAGE_FORMATS
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from yolo_sliding_counter import (
        YOLOSlidingCounter,
        DetectionResult,
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_NMS_THRESHOLD,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_OVERLAP_RATIO,
        SUPPORTED_IMAGE_FORMATS
    )


class TestYOLOSlidingCounterInitialization(unittest.TestCase):
    """Test YOLOSlidingCounter initialization and parameter validation"""
    
    def setUp(self):
        """Create temporary directory and mock model file"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pt")
        
        # Create dummy model file
        with open(self.model_path, 'w') as f:
            f.write("dummy model")
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_defaults(self, mock_load_model):
        """Test initialization with default parameters"""
        mock_load_model.return_value = Mock()
        
        counter = YOLOSlidingCounter(
            model_path=self.model_path
        )
        
        self.assertEqual(counter.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(counter.nms_threshold, DEFAULT_NMS_THRESHOLD)
        self.assertEqual(counter.window_size, DEFAULT_WINDOW_SIZE)
        self.assertEqual(counter.overlap_ratio, DEFAULT_OVERLAP_RATIO)
        mock_load_model.assert_called_once()
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_custom_parameters(self, mock_load_model):
        """Test initialization with custom parameters"""
        mock_load_model.return_value = Mock()
        
        counter = YOLOSlidingCounter(
            model_path=self.model_path,
            confidence_threshold=0.5,
            nms_threshold=0.3,
            window_size=(512, 512),
            overlap_ratio=0.3,
            output_dir="custom_output"
        )
        
        self.assertEqual(counter.confidence_threshold, 0.5)
        self.assertEqual(counter.nms_threshold, 0.3)
        self.assertEqual(counter.window_size, (512, 512))
        self.assertEqual(counter.overlap_ratio, 0.3)
        self.assertEqual(counter.output_dir, Path("custom_output"))
    
    def test_initialization_with_invalid_model_path(self):
        """Test initialization with non-existent model file"""
        with self.assertRaises(FileNotFoundError):
            YOLOSlidingCounter(model_path="nonexistent_model.pt")
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_invalid_confidence(self, mock_load_model):
        """Test initialization with invalid confidence threshold"""
        with self.assertRaises(ValueError):
            YOLOSlidingCounter(
                model_path=self.model_path,
                confidence_threshold=1.5
            )
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_invalid_nms(self, mock_load_model):
        """Test initialization with invalid NMS threshold"""
        with self.assertRaises(ValueError):
            YOLOSlidingCounter(
                model_path=self.model_path,
                nms_threshold=-0.1
            )
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_invalid_window_size(self, mock_load_model):
        """Test initialization with invalid window size"""
        with self.assertRaises(ValueError):
            YOLOSlidingCounter(
                model_path=self.model_path,
                window_size=(0, 640)
            )
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_initialization_with_invalid_overlap(self, mock_load_model):
        """Test initialization with invalid overlap ratio"""
        with self.assertRaises(ValueError):
            YOLOSlidingCounter(
                model_path=self.model_path,
                overlap_ratio=1.5
            )


class TestImageProcessing(unittest.TestCase):
    """Test image processing functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pt")
        
        # Create dummy model file
        with open(self.model_path, 'w') as f:
            f.write("dummy model")
        
        # Create test image
        self.test_image_path = os.path.join(self.test_dir, "test_image.jpg")
        self._create_test_image(self.test_image_path, (800, 600))
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_image(self, path, size=(640, 480)):
        """Create a test image"""
        try:
            import cv2
            image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            cv2.imwrite(path, image)
        except ImportError:
            # If cv2 not available, skip image creation
            pass
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_validate_image_path_valid(self, mock_load_model):
        """Test image path validation with valid path"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        # Should not raise exception
        counter._validate_image_path(self.test_image_path)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_validate_image_path_nonexistent(self, mock_load_model):
        """Test image path validation with non-existent file"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        with self.assertRaises(FileNotFoundError):
            counter._validate_image_path("nonexistent.jpg")
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_validate_image_path_unsupported_format(self, mock_load_model):
        """Test image path validation with unsupported format"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        # Create file with unsupported extension
        unsupported = os.path.join(self.test_dir, "test.txt")
        with open(unsupported, 'w') as f:
            f.write("test")
        
        with self.assertRaises(ValueError):
            counter._validate_image_path(unsupported)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    @patch('cv2.imread')
    def test_auto_scale_image_no_scaling_needed(self, mock_imread, mock_load_model):
        """Test auto-scale when image is already small enough"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        # Small image that doesn't need scaling
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        scaled, scale_factor = counter._auto_scale_image(image, max_dimension=2048)
        
        self.assertEqual(scale_factor, 1.0)
        np.testing.assert_array_equal(scaled, image)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    @patch('cv2.resize')
    def test_auto_scale_image_scaling_needed(self, mock_resize, mock_load_model):
        """Test auto-scale when image is too large"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        # Large image that needs scaling
        image = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
        mock_resize.return_value = np.random.randint(0, 255, (1536, 2048, 3), dtype=np.uint8)
        
        scaled, scale_factor = counter._auto_scale_image(image, max_dimension=2048)
        
        self.assertLess(scale_factor, 1.0)
        mock_resize.assert_called_once()


class TestSlidingWindow(unittest.TestCase):
    """Test sliding window generation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pt")
        
        with open(self.model_path, 'w') as f:
            f.write("dummy model")
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_generate_sliding_windows_no_overlap(self, mock_load_model):
        """Test sliding window generation without overlap"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        image_shape = (640, 640, 3)
        window_size = (320, 320)
        overlap_ratio = 0.0
        
        windows = counter._generate_sliding_windows(
            image_shape, window_size, overlap_ratio
        )
        
        # With no overlap, should have 2x2 = 4 windows
        self.assertGreaterEqual(len(windows), 4)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_generate_sliding_windows_with_overlap(self, mock_load_model):
        """Test sliding window generation with overlap"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        image_shape = (640, 640, 3)
        window_size = (320, 320)
        overlap_ratio = 0.5
        
        windows = counter._generate_sliding_windows(
            image_shape, window_size, overlap_ratio
        )
        
        # With 50% overlap, should have more windows
        self.assertGreater(len(windows), 4)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_generate_sliding_windows_single_window(self, mock_load_model):
        """Test sliding window generation when window equals image size"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        image_shape = (640, 640, 3)
        window_size = (640, 640)
        overlap_ratio = 0.2
        
        windows = counter._generate_sliding_windows(
            image_shape, window_size, overlap_ratio
        )
        
        # Should have exactly 1 window
        self.assertEqual(len(windows), 1)


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult dataclass"""
    
    def test_detection_result_creation(self):
        """Test creating DetectionResult object"""
        result = DetectionResult(
            total_detections=5,
            detection_boxes=[(10, 10, 50, 50)],
            detection_scores=[0.95],
            detection_classes=[0],
            processing_time=1.5,
            image_size=(640, 480),
            window_count=4
        )
        
        self.assertEqual(result.total_detections, 5)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.window_count, 4)
        self.assertIsNone(result.roi_image_path)
    
    def test_detection_result_with_paths(self):
        """Test DetectionResult with image paths"""
        result = DetectionResult(
            total_detections=3,
            detection_boxes=[],
            detection_scores=[],
            detection_classes=[],
            processing_time=2.0,
            image_size=(800, 600),
            window_count=6,
            roi_image_path="/path/to/roi.jpg",
            cropped_image_path="/path/to/cropped.jpg",
            result_image_path="/path/to/result.jpg"
        )
        
        self.assertEqual(result.roi_image_path, "/path/to/roi.jpg")
        self.assertEqual(result.cropped_image_path, "/path/to/cropped.jpg")
        self.assertEqual(result.result_image_path, "/path/to/result.jpg")


class TestVisualization(unittest.TestCase):
    """Test visualization functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pt")
        
        with open(self.model_path, 'w') as f:
            f.write("dummy model")
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    def test_draw_roi_annotation(self, mock_putText, mock_rectangle, mock_load_model):
        """Test ROI annotation drawing"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        roi = (100, 100, 400, 300)
        
        result = counter._draw_roi_annotation(image, roi)
        
        # Check that rectangle was called
        mock_rectangle.assert_called()
        mock_putText.assert_called()
        
        # Check that result has same shape as input
        self.assertEqual(result.shape, image.shape)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    def test_crop_roi(self, mock_load_model):
        """Test ROI cropping"""
        mock_load_model.return_value = Mock()
        counter = YOLOSlidingCounter(model_path=self.model_path)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        roi = (100, 100, 400, 300)
        
        cropped = counter._crop_roi(image, roi)
        
        # Check cropped size
        expected_height = 300 - 100
        expected_width = 400 - 100
        self.assertEqual(cropped.shape[0], expected_height)
        self.assertEqual(cropped.shape[1], expected_width)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pt")
        
        with open(self.model_path, 'w') as f:
            f.write("dummy model")
        
        # Create multiple test images
        self.image_paths = []
        for i in range(3):
            path = os.path.join(self.test_dir, f"test_{i}.jpg")
            self.image_paths.append(path)
            try:
                import cv2
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(path, image)
            except ImportError:
                pass
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('yolo_sliding_counter.YOLOSlidingCounter._load_model')
    @patch('yolo_sliding_counter.YOLOSlidingCounter.count_kernels_with_yolo')
    def test_batch_count(self, mock_count, mock_load_model):
        """Test batch processing"""
        mock_load_model.return_value = Mock()
        
        # Mock count_kernels_with_yolo to return results
        mock_result = DetectionResult(
            total_detections=5,
            detection_boxes=[],
            detection_scores=[],
            detection_classes=[],
            processing_time=1.0,
            image_size=(640, 480),
            window_count=4
        )
        mock_count.return_value = mock_result
        
        counter = YOLOSlidingCounter(model_path=self.model_path)
        results = counter.batch_count(self.image_paths)
        
        # Check that count was called for each image
        self.assertEqual(mock_count.call_count, len(self.image_paths))
        self.assertEqual(len(results), len(self.image_paths))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestYOLOSlidingCounterInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestImageProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSlidingWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionResult))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessing))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
