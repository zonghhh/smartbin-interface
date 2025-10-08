#!/usr/bin/env python3
"""
Test script for plastic bottle detection
Creates synthetic images to test the enhanced classification
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.classification_module import ClassificationModule

def create_plastic_bottle_image():
    """Create a synthetic plastic bottle image for testing"""
    # Create a white background
    img = Image.new('RGB', (300, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a bottle shape
    # Bottle body (cylindrical)
    draw.rectangle([100, 150, 200, 350], fill='lightblue', outline='blue', width=2)
    
    # Bottle neck
    draw.rectangle([120, 100, 180, 150], fill='lightblue', outline='blue', width=2)
    
    # Bottle cap
    draw.rectangle([115, 80, 185, 100], fill='darkblue', outline='black', width=2)
    
    # Add some vertical lines to simulate bottle ridges
    for x in range(110, 200, 10):
        draw.line([x, 150, x, 350], fill='blue', width=1)
    
    # Add label area
    draw.rectangle([110, 200, 190, 280], fill='white', outline='blue', width=1)
    draw.text((120, 220), "WATER", fill='blue')
    
    return img

def create_paper_image():
    """Create a synthetic paper image for testing"""
    img = Image.new('RGB', (300, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw crumpled paper effect
    draw.rectangle([50, 50, 250, 350], fill='white', outline='gray', width=2)
    
    # Add some text lines
    draw.text((60, 100), "Document", fill='black')
    draw.text((60, 130), "Important", fill='black')
    draw.text((60, 160), "Information", fill='black')
    
    # Add some wrinkles
    for i in range(5):
        y = 200 + i * 20
        draw.line([60, y, 240, y], fill='lightgray', width=1)
    
    return img

def create_food_image():
    """Create a synthetic food image for testing"""
    img = Image.new('RGB', (300, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw an apple
    draw.ellipse([100, 100, 200, 200], fill='red', outline='darkred', width=2)
    
    # Apple stem
    draw.rectangle([145, 80, 155, 100], fill='brown', outline='black', width=1)
    
    # Apple leaf
    draw.ellipse([150, 70, 170, 90], fill='green', outline='darkgreen', width=1)
    
    return img

def test_classification():
    """Test the enhanced classification on synthetic images"""
    print("Testing Enhanced Classification Model")
    print("=" * 40)
    
    classifier = ClassificationModule()
    
    # Test plastic bottle
    print("\n1. Testing Plastic Bottle Detection:")
    bottle_img = create_plastic_bottle_image()
    bottle_img.save("test_bottle.png")
    result = classifier.classify_image(bottle_img)
    print(f"   Detected: {result}")
    print(f"   Expected: plastic")
    print(f"   [CORRECT]" if result == 'plastic' else f"   [WRONG]")
    
    # Test paper
    print("\n2. Testing Paper Detection:")
    paper_img = create_paper_image()
    paper_img.save("test_paper.png")
    result = classifier.classify_image(paper_img)
    print(f"   Detected: {result}")
    print(f"   Expected: paper")
    print(f"   [CORRECT]" if result == 'paper' else f"   [WRONG]")
    
    # Test food
    print("\n3. Testing Food Detection:")
    food_img = create_food_image()
    food_img.save("test_food.png")
    result = classifier.classify_image(food_img)
    print(f"   Detected: {result}")
    print(f"   Expected: food")
    print(f"   [CORRECT]" if result == 'food' else f"   [WRONG]")
    
    print("\n" + "=" * 40)
    print("Test completed! Check the generated images:")
    print("- test_bottle.png")
    print("- test_paper.png") 
    print("- test_food.png")

if __name__ == "__main__":
    test_classification()
