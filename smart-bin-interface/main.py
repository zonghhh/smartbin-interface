"""
Smart Bin Interface - Main Application
A Streamlit-based interface for smart recycling bins with camera recognition,
manual trash type selection, and QR code generation for points collection.
"""

import streamlit as st
import time
import threading
from datetime import datetime
from components.camera_module import CameraModule
from components.classification_module import ClassificationModule
from components.ui_screens import UIScreens
from utils.qr_generator import QRGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Bin Interface",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'start'
if 'detected_type' not in st.session_state:
    st.session_state.detected_type = None
if 'selected_type' not in st.session_state:
    st.session_state.selected_type = None
if 'transaction_id' not in st.session_state:
    st.session_state.transaction_id = None
if 'timer_active' not in st.session_state:
    st.session_state.timer_active = False
if 'countdown' not in st.session_state:
    st.session_state.countdown = 0
if 'timer_start_time' not in st.session_state:
    st.session_state.timer_start_time = 0

# Initialize components
camera_module = CameraModule()
classification_module = ClassificationModule()
ui_screens = UIScreens()
qr_generator = QRGenerator()

def reset_app_state():
    """Reset the application to initial state"""
    st.session_state.app_state = 'start'
    st.session_state.detected_type = None
    st.session_state.selected_type = None
    st.session_state.transaction_id = None
    st.session_state.timer_active = False
    st.session_state.countdown = 0

def start_bin_timer():
    """Start the 20-second bin open timer"""
    st.session_state.timer_active = True
    st.session_state.countdown = 20
    st.session_state.timer_start_time = time.time()

def generate_transaction_id():
    """Generate a unique transaction ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"TXN_{timestamp}"

def main():
    """Main application logic"""
    
    # Display header
    st.title("🗑️ Smart Recycling Bin")
    
    # Model selection sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Model Settings")
        model_type = st.selectbox(
            "Choose AI Model:",
            ["reciclapi", "huggingface_free", "enhanced", "google_vision", "azure_cognitive", "aws_rekognition", "simple", "resnet", "huggingface"],
            index=0,
            help="Select the AI model for image recognition"
        )
        
        if st.button("Update Model"):
            classification_module.set_model_type(model_type)
        
        # Show model info
        model_info = classification_module.get_model_info()
        st.markdown("**Model Status:**")
        st.write(f"• Current: {model_info['model_type']}")
        st.write(f"• PyTorch: {'✅' if model_info['torch_available'] else '❌'}")
        st.write(f"• Transformers: {'✅' if model_info['transformers_available'] else '❌'}")
        st.write(f"• OpenCV: {'✅' if model_info['cv2_available'] else '❌'}")
    
    st.markdown("---")
    
    # State-based UI rendering
    if st.session_state.app_state == 'start':
        render_start_screen()
    elif st.session_state.app_state == 'bin_open':
        render_bin_open_screen()
    elif st.session_state.app_state == 'qr_display':
        render_qr_display_screen()

def render_start_screen():
    """Render the initial camera and manual selection screen"""
    
    st.markdown("### 📸 Insert your item for recognition")
    
    # Camera input
    captured_image = st.camera_input("Take a photo of your item", key="camera_input")
    
    if captured_image is not None:
        # Process the image for classification
        with st.spinner("Analyzing item..."):
            detected_type = classification_module.classify_image(captured_image)
            st.session_state.detected_type = detected_type
        
        if detected_type:
            st.success(f"🔍 Detected: {detected_type.title()}")
        else:
            st.warning("❌ Couldn't recognize item")
    
    st.markdown("### 🎯 Or select manually:")
    
    # Manual selection buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧴 Plastic", use_container_width=True):
            st.session_state.selected_type = "plastic"
            proceed_with_selection()
    
    with col2:
        if st.button("📄 Paper", use_container_width=True):
            st.session_state.selected_type = "paper"
            proceed_with_selection()
    
    with col3:
        if st.button("⚡ Electronics", use_container_width=True):
            st.session_state.selected_type = "electronics"
            proceed_with_selection()
    
    col4, col5 = st.columns(2)
    
    with col4:
        if st.button("🍎 Food", use_container_width=True):
            st.session_state.selected_type = "food"
            proceed_with_selection()
    
    with col5:
        if st.button("🗑 General", use_container_width=True):
            st.session_state.selected_type = "general"
            proceed_with_selection()
    
    # Confirm detected type if available
    if st.session_state.detected_type and not st.session_state.selected_type:
        st.markdown("### ✅ Confirm detected type:")
        if st.button(f"Confirm: {st.session_state.detected_type.title()}", type="primary"):
            st.session_state.selected_type = st.session_state.detected_type
            proceed_with_selection()

def proceed_with_selection():
    """Proceed to bin opening after type selection"""
    if st.session_state.selected_type:
        st.session_state.app_state = 'bin_open'
        st.session_state.transaction_id = generate_transaction_id()
        start_bin_timer()
        st.rerun()

def render_bin_open_screen():
    """Render the bin open screen with countdown"""
    
    trash_type = st.session_state.selected_type
    emoji_map = {
        'plastic': '🧴',
        'paper': '📄',
        'electronics': '⚡',
        'food': '🍎',
        'general': '🗑'
    }
    
    emoji = emoji_map.get(trash_type, '🗑')
    
    st.markdown(f"### {emoji} {trash_type.title()} bin opened!")
    st.success(f"✅ {trash_type.title()} bin opened for {trash_type}")
    
    # Countdown display
    if st.session_state.timer_active:
        # Calculate remaining time
        elapsed = time.time() - st.session_state.timer_start_time
        remaining = max(0, 20 - elapsed)
        st.session_state.countdown = int(remaining)
        
        if remaining > 0:
            st.markdown(f"### ⏰ Bin will close in: {st.session_state.countdown} seconds")
            
            # Progress bar
            progress = (20 - remaining) / 20
            st.progress(progress)
            
            # Auto-refresh every second
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.timer_active = False
            st.session_state.app_state = 'qr_display'
            st.rerun()
    else:
        st.markdown("### ⏰ Bin closed!")
        st.session_state.app_state = 'qr_display'
        st.rerun()

def render_qr_display_screen():
    """Render the QR code display and rating screen"""
    
    st.markdown("### 🎉 Thank you for recycling!")
    st.markdown("Scan this QR code to collect your recycling points:")
    
    # Generate and display QR code
    if st.session_state.transaction_id:
        qr_image = qr_generator.generate_qr(st.session_state.transaction_id)
        st.image(qr_image, caption="Transaction QR Code", use_column_width=True)
        
        st.markdown(f"**Transaction ID:** `{st.session_state.transaction_id}`")
    
    # Rating buttons
    st.markdown("### How was your experience?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Great!", use_container_width=True):
            st.success("Thank you for your feedback!")
            time.sleep(1)
            reset_app_state()
            st.rerun()
    
    with col2:
        if st.button("😐 Okay", use_container_width=True):
            st.info("Thank you for your feedback!")
            time.sleep(1)
            reset_app_state()
            st.rerun()
    
    with col3:
        if st.button("😞 Poor", use_container_width=True):
            st.warning("Thank you for your feedback!")
            time.sleep(1)
            reset_app_state()
            st.rerun()
    
    # Done button
    if st.button("✅ Done", type="primary", use_container_width=True):
        reset_app_state()
        st.rerun()
    
    # Auto-return after 30 seconds
    if 'auto_return_timer' not in st.session_state:
        st.session_state.auto_return_timer = time.time()
    
    elapsed = time.time() - st.session_state.auto_return_timer
    remaining = max(0, 30 - elapsed)
    
    if remaining > 0:
        st.markdown(f"⏰ Auto-return in {int(remaining)} seconds...")
    else:
        st.markdown("🔄 Returning to start screen...")
        time.sleep(2)
        reset_app_state()
        st.rerun()

if __name__ == "__main__":
    main()
