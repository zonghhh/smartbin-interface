#!/usr/bin/env python3
"""
Demo script for Smart Bin Interface components
Test the individual modules without running the full Streamlit app
"""

import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.camera_module import CameraModule
from components.classification_module import ClassificationModule
from utils.qr_generator import QRGenerator

def test_camera_module():
    """Test the camera module"""
    print("Testing Camera Module...")
    camera = CameraModule()
    
    status = camera.get_camera_status()
    print(f"Camera available: {status['available']}")
    print(f"Camera status: {status['status']}")
    
    # Test mock image generation
    mock_image = camera._get_mock_image()
    print(f"Mock image generated: {mock_image.size}")
    
    return camera

def test_classification_module():
    """Test the classification module"""
    print("\nTesting Classification Module...")
    classifier = ClassificationModule()
    
    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:] = (100, 150, 200)  # Light blue
    test_pil = Image.fromarray(test_image)
    
    # Test classification
    detected_type = classifier.classify_image(test_pil)
    print(f"Detected type: {detected_type}")
    
    # Test confidence
    confidence = classifier.get_classification_confidence(test_pil)
    print(f"Confidence: {confidence:.2f}")
    
    # Test supported types
    types = classifier.get_supported_types()
    print(f"Supported types: {types}")
    
    return classifier

def test_qr_generator():
    """Test the QR code generator"""
    print("\nTesting QR Generator...")
    qr_gen = QRGenerator()
    
    # Test basic QR generation
    qr_image = qr_gen.generate_qr("TEST_TRANSACTION_123")
    print(f"Basic QR generated: {qr_image.size}")
    
    # Test transaction QR
    try:
        txn_qr = qr_gen.generate_transaction_qr("TXN_20241201_123456", 10, "USER_001")
        print(f"Transaction QR generated: {txn_qr.size}")
    except Exception as e:
        print(f"Error generating transaction QR: {e}")
    
    # Test user QR
    try:
        user_qr = qr_gen.generate_user_qr("USER_001", "John Doe")
        print(f"User QR generated: {user_qr.size}")
    except Exception as e:
        print(f"Error generating user QR: {e}")
    
    return qr_gen

def main():
    """Run all tests"""
    print("Smart Bin Interface - Component Demo")
    print("=" * 40)
    
    try:
        # Test camera module
        camera = test_camera_module()
        
        # Test classification module
        classifier = test_classification_module()
        
        # Test QR generator
        qr_gen = test_qr_generator()
        
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        print("\nTo run the full application, use:")
        print("  python run.py")
        print("  or")
        print("  streamlit run main.py")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
