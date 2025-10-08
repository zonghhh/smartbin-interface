"""
QR Code Generator for Smart Bin Interface
Generates QR codes for transaction IDs and points collection.
"""

import qrcode
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Optional, Dict, Any

class QRGenerator:
    """Handles QR code generation for the smart bin interface"""
    
    def __init__(self):
        self.default_config = {
            'version': 1,
            'error_correction': qrcode.constants.ERROR_CORRECT_L,
            'box_size': 10,
            'border': 4,
        }
    
    def generate_qr(self, 
                   data: str, 
                   config: Optional[Dict[str, Any]] = None,
                   add_logo: bool = False,
                   logo_path: Optional[str] = None) -> Image.Image:
        """
        Generate a QR code image
        
        Args:
            data: Data to encode in QR code
            config: QR code configuration (optional)
            add_logo: Whether to add a logo in the center
            logo_path: Path to logo image (optional)
            
        Returns:
            PIL Image: Generated QR code image
        """
        # Merge default config with provided config
        qr_config = self.default_config.copy()
        if config:
            qr_config.update(config)
        
        # Create QR code
        qr = qrcode.QRCode(**qr_config)
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Add logo if requested
        if add_logo and logo_path:
            qr_image = self._add_logo(qr_image, logo_path)
        
        return qr_image
    
    def generate_qr_with_text(self, 
                             data: str, 
                             title: str = "Transaction QR Code",
                             subtitle: str = "Scan to collect points",
                             config: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Generate QR code with text labels
        
        Args:
            data: Data to encode in QR code
            title: Title text above QR code
            subtitle: Subtitle text below QR code
            config: QR code configuration (optional)
            
        Returns:
            PIL Image: QR code image with text
        """
        # Generate base QR code
        qr_image = self.generate_qr(data, config)
        
        # Calculate dimensions for text
        qr_width, qr_height = qr_image.size
        text_padding = 20
        total_width = qr_width + (text_padding * 2)
        total_height = qr_height + 100  # Extra space for text
        
        # Create new image with text
        final_image = Image.new('RGB', (total_width, total_height), 'white')
        
        # Paste QR code in center
        qr_x = text_padding
        qr_y = 50
        final_image.paste(qr_image, (qr_x, qr_y))
        
        # Add text
        draw = ImageDraw.Draw(final_image)
        
        # Draw title (simple text without font)
        draw.text((10, 10), title, fill='black')
        
        # Draw subtitle (simple text without font)
        draw.text((10, qr_y + qr_height + 10), subtitle, fill='gray')
        
        return final_image
    
    def generate_transaction_qr(self, 
                               transaction_id: str,
                               points: int = 0,
                               user_id: Optional[str] = None) -> Image.Image:
        """
        Generate QR code for a recycling transaction
        
        Args:
            transaction_id: Unique transaction identifier
            points: Points earned from recycling
            user_id: User identifier (optional)
            
        Returns:
            PIL Image: Transaction QR code image
        """
        # Create transaction data
        transaction_data = {
            'transaction_id': transaction_id,
            'points': points,
            'user_id': user_id,
            'timestamp': self._get_timestamp(),
            'type': 'recycling_transaction'
        }
        
        # Convert to string format
        data_string = self._format_transaction_data(transaction_data)
        
        # Generate simple QR code without text
        return self.generate_qr(data_string)
    
    def generate_user_qr(self, user_id: str, user_name: Optional[str] = None) -> Image.Image:
        """
        Generate QR code for user identification
        
        Args:
            user_id: User identifier
            user_name: User name (optional)
            
        Returns:
            PIL Image: User QR code image
        """
        user_data = {
            'user_id': user_id,
            'user_name': user_name,
            'type': 'user_identification'
        }
        
        data_string = self._format_user_data(user_data)
        
        # Generate simple QR code without text
        return self.generate_qr(data_string)
    
    def _add_logo(self, qr_image: Image.Image, logo_path: str) -> Image.Image:
        """
        Add logo to center of QR code
        
        Args:
            qr_image: QR code image
            logo_path: Path to logo image
            
        Returns:
            PIL Image: QR code with logo
        """
        try:
            logo = Image.open(logo_path)
            
            # Resize logo to fit in QR code
            qr_width, qr_height = qr_image.size
            logo_size = min(qr_width, qr_height) // 4
            logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
            
            # Create white background for logo
            logo_bg = Image.new('RGB', (logo_size + 10, logo_size + 10), 'white')
            logo_bg.paste(logo, (5, 5))
            
            # Calculate position for logo
            logo_x = (qr_width - logo_size - 10) // 2
            logo_y = (qr_height - logo_size - 10) // 2
            
            # Paste logo onto QR code
            qr_image.paste(logo_bg, (logo_x, logo_y))
            
        except Exception as e:
            print(f"Error adding logo: {e}")
        
        return qr_image
    
    def _format_transaction_data(self, data: Dict[str, Any]) -> str:
        """Format transaction data for QR code"""
        return f"TXN:{data['transaction_id']}|PTS:{data['points']}|UID:{data['user_id'] or 'UNKNOWN'}|TS:{data['timestamp']}"
    
    def _format_user_data(self, data: Dict[str, Any]) -> str:
        """Format user data for QR code"""
        return f"USER:{data['user_id']}|NAME:{data['user_name'] or 'Unknown'}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def qr_to_base64(self, qr_image: Image.Image) -> str:
        """
        Convert QR code image to base64 string
        
        Args:
            qr_image: QR code image
            
        Returns:
            str: Base64 encoded image string
        """
        buffer = io.BytesIO()
        qr_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def save_qr(self, qr_image: Image.Image, filepath: str) -> bool:
        """
        Save QR code image to file
        
        Args:
            qr_image: QR code image
            filepath: Path to save file
            
        Returns:
            bool: True if saved successfully
        """
        try:
            qr_image.save(filepath)
            return True
        except Exception as e:
            print(f"Error saving QR code: {e}")
            return False
