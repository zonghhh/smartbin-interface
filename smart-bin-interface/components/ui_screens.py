"""
UI Screens Module for Smart Bin Interface
Contains UI components and screen layouts for different app states.
"""

import streamlit as st
from typing import Dict, List, Optional

class UIScreens:
    """Handles UI screen rendering and components"""
    
    def __init__(self):
        self.trash_type_emojis = {
            'plastic': '🧴',
            'paper': '📄',
            'electronics': '⚡',
            'food': '🍎',
            'general': '🗑'
        }
        
        self.trash_type_colors = {
            'plastic': '#1f77b4',
            'paper': '#ff7f0e',
            'electronics': '#2ca02c',
            'food': '#d62728',
            'general': '#9467bd'
        }
    
    def render_header(self, title: str = "Smart Recycling Bin"):
        """Render the main header"""
        st.title(f"🗑️ {title}")
        st.markdown("---")
    
    def render_camera_section(self, camera_available: bool = True):
        """Render the camera input section"""
        st.markdown("### 📸 Insert your item for recognition")
        
        if not camera_available:
            st.warning("⚠️ Camera not available. Using mock mode for testing.")
        
        captured_image = st.camera_input(
            "Take a photo of your item", 
            key="camera_input",
            help="Position your item in front of the camera and click to capture"
        )
        
        return captured_image
    
    def render_manual_selection(self, detected_type: Optional[str] = None):
        """Render manual trash type selection buttons"""
        st.markdown("### 🎯 Or select manually:")
        
        # Create a grid layout for buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧴 Plastic", use_container_width=True, key="btn_plastic"):
                return "plastic"
        
        with col2:
            if st.button("📄 Paper", use_container_width=True, key="btn_paper"):
                return "paper"
        
        with col3:
            if st.button("⚡ Electronics", use_container_width=True, key="btn_electronics"):
                return "electronics"
        
        col4, col5 = st.columns(2)
        
        with col4:
            if st.button("🍎 Food", use_container_width=True, key="btn_food"):
                return "food"
        
        with col5:
            if st.button("🗑 General", use_container_width=True, key="btn_general"):
                return "general"
        
        return None
    
    def render_detection_confirmation(self, detected_type: str):
        """Render confirmation for detected trash type"""
        if detected_type:
            emoji = self.trash_type_emojis.get(detected_type, '🗑')
            color = self.trash_type_colors.get(detected_type, '#000000')
            
            st.markdown("### ✅ Confirm detected type:")
            
            # Create a styled confirmation button
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; margin: 10px 0;">
                <h3>{emoji} {detected_type.title()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    return True
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    return False
        
        return None
    
    def render_bin_open_screen(self, trash_type: str, countdown: int):
        """Render the bin open screen with countdown"""
        emoji = self.trash_type_emojis.get(trash_type, '🗑')
        color = self.trash_type_colors.get(trash_type, '#000000')
        
        # Main message
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: {color}20; border-radius: 15px; margin: 20px 0;">
            <h2>{emoji} {trash_type.title()} bin opened!</h2>
            <p style="font-size: 18px;">✅ {trash_type.title()} bin opened for {trash_type}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Countdown display
        if countdown > 0:
            st.markdown(f"### ⏰ Bin will close in: {countdown} seconds")
            
            # Progress bar
            progress = (20 - countdown) / 20
            st.progress(progress)
            
            # Visual countdown
            st.markdown(f"""
            <div style="text-align: center; font-size: 48px; font-weight: bold; color: {color}; margin: 20px 0;">
                {countdown}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### ⏰ Bin closed!")
    
    def render_qr_display_screen(self, transaction_id: str, qr_image):
        """Render the QR code display and rating screen"""
        st.markdown("### 🎉 Thank you for recycling!")
        st.markdown("Scan this QR code to collect your recycling points:")
        
        # QR Code display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(qr_image, caption="Transaction QR Code", use_column_width=True)
        
        # Transaction ID
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0;">
            <strong>Transaction ID:</strong> <code>{transaction_id}</code>
        </div>
        """, unsafe_allow_html=True)
    
    def render_rating_section(self):
        """Render the rating buttons"""
        st.markdown("### How was your experience?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("😊 Great!", use_container_width=True, key="rating_great"):
                return "great"
        
        with col2:
            if st.button("😐 Okay", use_container_width=True, key="rating_okay"):
                return "okay"
        
        with col3:
            if st.button("😞 Poor", use_container_width=True, key="rating_poor"):
                return "poor"
        
        return None
    
    def render_auto_return_timer(self, remaining_seconds: int):
        """Render the auto-return timer"""
        if remaining_seconds > 0:
            st.markdown(f"⏰ Auto-return in {remaining_seconds} seconds...")
        else:
            st.markdown("🔄 Returning to start screen...")
    
    def render_error_message(self, message: str, error_type: str = "error"):
        """Render error messages"""
        if error_type == "error":
            st.error(f"❌ {message}")
        elif error_type == "warning":
            st.warning(f"⚠️ {message}")
        elif error_type == "info":
            st.info(f"ℹ️ {message}")
    
    def render_success_message(self, message: str):
        """Render success messages"""
        st.success(f"✅ {message}")
    
    def render_loading_spinner(self, message: str = "Processing..."):
        """Render loading spinner"""
        with st.spinner(message):
            pass
    
    def render_status_indicator(self, status: str, color: str = "green"):
        """Render status indicator"""
        color_map = {
            "green": "🟢",
            "yellow": "🟡",
            "red": "🔴",
            "blue": "🔵"
        }
        
        emoji = color_map.get(color, "⚪")
        st.markdown(f"{emoji} Status: {status}")
    
    def render_help_text(self, text: str):
        """Render help text"""
        st.markdown(f"💡 **Help:** {text}")
    
    def render_footer(self):
        """Render footer information"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            Smart Recycling Bin Interface v1.0<br>
            Built with Streamlit for Raspberry Pi
        </div>
        """, unsafe_allow_html=True)

