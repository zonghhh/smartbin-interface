# Smart Bin Interface

A Streamlit-based interface for smart recycling bins with camera recognition, manual trash type selection, and QR code generation for points collection.

## Features

- 📸 **Camera Recognition**: Automatic trash type detection using computer vision
- 🎯 **Manual Selection**: Override or confirm detected trash types manually
- ⏰ **Bin Operation**: Simulated bin opening with countdown timer
- 📱 **QR Code Generation**: Generate QR codes for transaction tracking and points collection
- 😊 **User Feedback**: Rating system for user experience feedback
- 🔄 **Auto-return**: Automatic return to start screen after timeout

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
smart-bin-interface/
├── main.py                 # Main Streamlit application
├── components/
│   ├── camera_module.py    # Camera operations and image capture
│   ├── classification_module.py  # Image recognition and trash classification
│   └── ui_screens.py       # UI components and screen layouts
├── utils/
│   ├── timer.py           # Timer utilities for countdown functionality
│   └── qr_generator.py    # QR code generation for transactions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

1. **Start Screen**: Camera view with manual selection buttons
2. **Recognition**: Automatic trash type detection or manual selection
3. **Bin Operation**: Confirmation message and 20-second countdown timer
4. **QR Display**: Transaction QR code and user feedback options
5. **Auto-return**: Automatic return to start screen after 30 seconds

## Configuration

The app uses session state to manage different screens and user interactions. Key configuration options:

- **Timer Duration**: 20 seconds for bin open time
- **Auto-return Timeout**: 30 seconds before returning to start
- **Camera Mode**: Mock mode when camera is not available
- **QR Code Format**: Transaction ID with timestamp

## Integration Notes

This interface is designed to integrate with:
- Backend API for transaction processing
- User mobile app for points collection
- Admin dashboard for monitoring
- Hardware sensors for bin control

## Development

To extend the application:

1. **Add new trash types**: Update `classification_module.py` and `ui_screens.py`
2. **Improve recognition**: Enhance the classification logic in `classification_module.py`
3. **Customize UI**: Modify components in `ui_screens.py`
4. **Add features**: Extend the main application flow in `main.py`

## Hardware Requirements

- Raspberry Pi 4 or similar single-board computer
- Touchscreen display (7" or larger recommended)
- Camera module (optional, mock mode available)
- Internet connection for backend integration

## Troubleshooting

- **Camera not working**: The app will automatically switch to mock mode
- **Recognition issues**: Use manual selection buttons as fallback
- **Performance**: Ensure adequate RAM and processing power for smooth operation

## License

This project is part of the Smart Recycling System and is intended for educational and demonstration purposes.

