# Installing ML Libraries for Better Image Recognition

This guide helps you install the optional machine learning libraries for improved image recognition in the Smart Bin Interface.

## Quick Installation

### Option 1: Install All ML Libraries
```bash
pip install torch torchvision transformers
```

### Option 2: Install Specific Libraries
```bash
# For ResNet18 model
pip install torch torchvision

# For Hugging Face models
pip install transformers

# For better performance (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Platform-Specific Installation

### Windows
```bash
# CPU version (recommended for most users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### macOS
```bash
# CPU version
pip install torch torchvision

# M1/M2 Mac (Apple Silicon)
pip install torch torchvision
```

### Linux
```bash
# CPU version
pip install torch torchvision

# GPU version (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Raspberry Pi Installation

For Raspberry Pi deployment, use lighter versions:

```bash
# Install PyTorch for ARM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or use ONNX runtime for even lighter deployment
pip install onnxruntime
```

## Verification

Test if the libraries are installed correctly:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('Torchvision version:', torchvision.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Use CPU-only versions
   - Reduce batch size
   - Use smaller models

2. **Slow Performance**
   - Install CPU-optimized versions
   - Use smaller models (ResNet18 instead of ResNet50)
   - Consider model quantization

3. **Import Errors**
   - Check Python version compatibility
   - Reinstall with correct index URL
   - Use virtual environment

### Performance Tips

1. **For Development**: Use simple model (no ML libraries needed)
2. **For Testing**: Use ResNet18 (moderate accuracy, reasonable speed)
3. **For Production**: Use Hugging Face models or custom trained models
4. **For Edge Devices**: Use quantized models or ONNX runtime

## Model Download

The models will be downloaded automatically on first use:
- ResNet18: ~45MB
- Hugging Face ResNet-50: ~100MB

Make sure you have internet connection for the first run.

## Memory Requirements

- **Simple Model**: ~50MB RAM
- **ResNet18**: ~200MB RAM
- **Hugging Face ResNet-50**: ~500MB RAM

Choose the model based on your available resources.
