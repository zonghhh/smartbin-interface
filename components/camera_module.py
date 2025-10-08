"""
Camera Module for Smart Bin Interface
Handles camera operations and image capture functionality.
"""

import streamlit as st
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class CameraModule:
    """Handles camera operations for the smart bin interface"""
    
    def __init__(self):
        self.camera_available = self._check_camera_availability()
    
    def _check_camera_availability(self):
        """Check if camera is available on the system"""
        if not CV2_AVAILABLE:
            return False
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                return True
        except Exception:
            pass
        return False
    
    def capture_image(self):
        """Capture an image from the camera"""
        if not self.camera_available:
            st.warning("Camera not available. Using mock image for testing.")
            return self._get_mock_image()
        
        if not CV2_AVAILABLE:
            return self._get_mock_image()
        
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            else:
                st.error("Failed to capture image")
                return None
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            return self._get_mock_image()
    
    def _get_mock_image(self):
        """Generate a mock image for testing when camera is not available"""
        # Create a simple mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image[:] = (100, 150, 200)  # Light blue background
        
        # Add some text to indicate it's a mock
        cv2.putText(mock_image, "MOCK CAMERA", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(mock_image, "Testing Mode", (220, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return Image.fromarray(mock_image)
    
    def process_image(self, image):
        """Process the captured image for better recognition"""
        if not CV2_AVAILABLE:
            return image
        
        if isinstance(image, Image.Image):
            # Convert PIL to OpenCV format
            img_array = np.array(image)
        else:
            img_array = image
        
        # Basic image processing
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply some basic filters
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Convert back to RGB
        processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(processed)
    
    def get_camera_status(self):
        """Get the current camera status"""
        return {
            'available': self.camera_available,
            'status': 'Ready' if self.camera_available else 'Not Available'
        }
