# AI Models for Smart Bin Interface

This document describes the available AI models for image recognition in the Smart Bin Interface.

## Available Models

### 1. Simple Rule-Based Model (Default)
- **Type**: Custom rule-based classification
- **Dependencies**: None (always available)
- **Accuracy**: Low to Medium
- **Speed**: Very Fast
- **Use Case**: Basic functionality, testing, fallback

**How it works:**
- Analyzes basic image features (color, shape, texture)
- Uses simple heuristics to classify trash types
- Fast but not very accurate

### 2. ResNet18 Model
- **Type**: Pre-trained ResNet18 from PyTorch
- **Dependencies**: `torch`, `torchvision`
- **Accuracy**: Medium to High
- **Speed**: Medium
- **Use Case**: Better accuracy with moderate computational requirements

**How it works:**
- Uses pre-trained ResNet18 model from ImageNet
- Maps ImageNet classes to trash types
- Requires PyTorch installation

**Installation:**
```bash
pip install torch torchvision
```

### 3. Hugging Face ResNet-50 Model
- **Type**: Pre-trained ResNet-50 from Hugging Face
- **Dependencies**: `transformers`
- **Accuracy**: High
- **Speed**: Medium to Slow
- **Use Case**: Best accuracy for production use

**How it works:**
- Uses Microsoft's ResNet-50 model from Hugging Face
- More sophisticated classification
- Requires transformers library

**Installation:**
```bash
pip install transformers
```

## Model Selection

### For Development/Testing:
- Use **Simple** model for quick testing
- No additional dependencies required

### For Better Accuracy:
- Use **ResNet18** for moderate accuracy with reasonable speed
- Use **Hugging Face ResNet-50** for best accuracy

### For Production:
- Consider training a custom model on trash-specific datasets
- Use transfer learning with pre-trained models
- Implement model ensemble for better accuracy

## Recommended Pre-trained Models for Trash Classification

### 1. TrashNet Dataset Models
- **Dataset**: TrashNet (6 classes: glass, paper, cardboard, plastic, metal, trash)
- **Models**: Custom CNN, ResNet, MobileNet
- **Accuracy**: 80-95% on trash-specific data
- **GitHub**: https://github.com/garythung/trashnet

### 2. Waste Classification Models
- **Dataset**: Various waste classification datasets
- **Models**: EfficientNet, DenseNet, Vision Transformer
- **Accuracy**: 85-98% on waste data
- **Platform**: Kaggle, Papers with Code

### 3. Real-time Models for Edge Devices
- **MobileNetV2**: Lightweight, fast inference
- **EfficientNet-Lite**: Optimized for mobile/edge
- **YOLOv5**: Object detection + classification

## Custom Model Training

To train a custom model for your specific trash types:

1. **Collect Data**: Gather images of different trash types
2. **Label Data**: Annotate images with correct categories
3. **Preprocess**: Resize, augment, and split data
4. **Train Model**: Use transfer learning or train from scratch
5. **Deploy**: Integrate with the smart bin interface

### Example Training Script:
```python
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# Load pre-trained model
model = models.resnet18(pretrained=True)
num_classes = 5  # plastic, paper, electronics, food, general
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training code here...
```

## Performance Comparison

| Model | Accuracy | Speed | Memory | Dependencies |
|-------|----------|-------|--------|--------------|
| Simple | 30-50% | Very Fast | Low | None |
| ResNet18 | 60-80% | Medium | Medium | torch, torchvision |
| Hugging Face | 70-90% | Medium-Slow | High | transformers |
| Custom TrashNet | 80-95% | Medium | Medium | Custom training |

## Integration Notes

- Models are loaded lazily (only when selected)
- Fallback to simple model if advanced models fail
- Model switching requires app restart for best performance
- Consider GPU acceleration for production deployment

## Future Improvements

1. **Real-time Object Detection**: Use YOLO or similar for multiple objects
2. **Multi-modal Classification**: Combine image + text descriptions
3. **Edge Optimization**: Quantize models for Raspberry Pi deployment
4. **Continuous Learning**: Update models with user feedback
5. **Ensemble Methods**: Combine multiple models for better accuracy
