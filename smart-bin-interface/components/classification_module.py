"""
Classification Module for Smart Bin Interface
Handles image recognition and trash type classification.
"""

import streamlit as st
import numpy as np
from PIL import Image
import random
import requests
import io

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ClassificationModule:
    """Handles image classification for trash type recognition"""
    
    def __init__(self):
        self.trash_types = ['plastic', 'paper', 'electronics', 'food', 'general']
        self.confidence_threshold = 0.3
        self.model_type = 'reciclapi'  # 'simple', 'enhanced', 'reciclapi', 'huggingface_free', 'resnet', 'huggingface', 'google_vision', 'azure_cognitive', 'aws_rekognition'
        self.model = None
        self.transform = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model"""
        if TORCH_AVAILABLE and self.model_type == 'resnet':
            try:
                # Load pre-trained ResNet model
                self.model = models.resnet18(pretrained=True)
                self.model.eval()
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                st.info("✅ ResNet18 model loaded successfully")
            except Exception as e:
                st.warning(f"Failed to load ResNet model: {e}")
                self.model_type = 'simple'
        
        elif TRANSFORMERS_AVAILABLE and self.model_type == 'huggingface':
            try:
                # Load Hugging Face model for image classification
                self.model = pipeline("image-classification", 
                                    model="microsoft/resnet-50", 
                                    device=-1)  # Use CPU
                st.info("✅ Hugging Face ResNet-50 model loaded successfully")
            except Exception as e:
                st.warning(f"Failed to load Hugging Face model: {e}")
                self.model_type = 'simple'
        
        elif self.model_type == 'reciclapi':
            st.info("Using ReciclAPI - Garbage Detection (RapidAPI)")
        
        elif self.model_type == 'huggingface_free':
            st.info("Using Hugging Face free inference API (no API key required)")
        
        elif self.model_type in ['google_vision', 'azure_cognitive', 'aws_rekognition']:
            st.info(f"Using {self.model_type} API (requires API key configuration)")
        
        else:
            if self.model_type == 'enhanced':
                st.info("Using enhanced rule-based classification with bottle detection")
            else:
                st.info("Using simple rule-based classification")
    
    def classify_image(self, image):
        """
        Classify the type of trash in the image
        
        Args:
            image: PIL Image or file-like object
            
        Returns:
            str: Detected trash type or None if not recognized
        """
        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            
            # Use different classification methods based on available models
            if self.model_type == 'resnet' and self.model is not None:
                detected_type = self._classify_with_resnet(image)
            elif self.model_type == 'huggingface' and self.model is not None:
                detected_type = self._classify_with_huggingface(image)
            elif self.model_type == 'reciclapi':
                detected_type = self._classify_with_reciclapi(image)
            elif self.model_type == 'huggingface_free':
                detected_type = self._classify_with_huggingface_free(image)
            elif self.model_type == 'google_vision':
                detected_type = self._classify_with_google_vision(image)
            elif self.model_type == 'azure_cognitive':
                detected_type = self._classify_with_azure_cognitive(image)
            elif self.model_type == 'aws_rekognition':
                detected_type = self._classify_with_aws_rekognition(image)
            elif self.model_type == 'enhanced':
                # Use enhanced analysis
                img_array = np.array(image)
                detected_type = self._simple_analysis(img_array)
            else:
                # Fallback to simple analysis
                img_array = np.array(image)
                detected_type = self._analyze_image(img_array)
            
            return detected_type
            
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return None
    
    def _classify_with_resnet(self, image):
        """Classify using pre-trained ResNet model"""
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Map ImageNet classes to trash types (simplified mapping)
            # This is a basic mapping - in practice, you'd train a custom classifier
            top_prob, top_class = torch.max(probabilities, 0)
            
            # Simple mapping based on ImageNet classes
            if top_prob > 0.5:  # High confidence
                if top_class.item() in [0, 1, 2, 3, 4]:  # Some ImageNet classes
                    return 'plastic'
                elif top_class.item() in [5, 6, 7, 8, 9]:
                    return 'paper'
                elif top_class.item() in [10, 11, 12, 13, 14]:
                    return 'electronics'
                elif top_class.item() in [15, 16, 17, 18, 19]:
                    return 'food'
            
            return None
            
        except Exception as e:
            st.warning(f"ResNet classification failed: {e}")
            return None
    
    def _classify_with_huggingface(self, image):
        """Classify using Hugging Face model"""
        try:
            # Get predictions from Hugging Face model
            results = self.model(image)
            
            # Map to trash types based on predictions
            for result in results[:3]:  # Top 3 predictions
                label = result['label'].lower()
                confidence = result['score']
                
                if confidence > 0.3:  # Confidence threshold
                    if any(word in label for word in ['bottle', 'container', 'plastic']):
                        return 'plastic'
                    elif any(word in label for word in ['paper', 'cardboard', 'document']):
                        return 'paper'
                    elif any(word in label for word in ['phone', 'computer', 'electronic', 'device']):
                        return 'electronics'
                    elif any(word in label for word in ['food', 'apple', 'banana', 'fruit', 'vegetable']):
                        return 'food'
            
            return None
            
        except Exception as e:
            st.warning(f"Hugging Face classification failed: {e}")
            return None
    
    def _classify_with_reciclapi(self, image):
        """Classify using ReciclAPI - Garbage Detection (RapidAPI)"""
        try:
            # Convert image to base64
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # ReciclAPI endpoint
            url = "https://reciclapi-garbage-detection.p.rapidapi.com/detect"
            
            # Get API key from Streamlit secrets
            # api_key = st.secrets.get("RAPIDAPI_KEY", "")
            api_key = '8ec669bd50mshd0ec619c0efc7f9p14cd7cjsn357a9c1e8f42'

            if not api_key:
                st.warning("RapidAPI key not configured. Using enhanced analysis.")
                img_array = np.array(image)
                return self._simple_analysis(img_array)
            
            headers = {
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "reciclapi-garbage-detection.p.rapidapi.com",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": img_base64
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                
                # ReciclAPI returns specific garbage types
                if 'predictions' in result:
                    predictions = result['predictions']
                    
                    # Get the highest confidence prediction
                    if predictions:
                        best_prediction = max(predictions, key=lambda x: x.get('confidence', 0))
                        garbage_type = best_prediction.get('class', '').lower()
                        confidence = best_prediction.get('confidence', 0)
                        
                        if confidence > 0.5:  # High confidence threshold
                            # Map ReciclAPI classes to our trash types
                            if any(word in garbage_type for word in ['plastic', 'bottle', 'container']):
                                return 'plastic'
                            elif any(word in garbage_type for word in ['paper', 'cardboard', 'document']):
                                return 'paper'
                            elif any(word in garbage_type for word in ['electronic', 'battery', 'phone']):
                                return 'electronics'
                            elif any(word in garbage_type for word in ['food', 'organic', 'fruit', 'vegetable']):
                                return 'food'
                            elif any(word in garbage_type for word in ['glass', 'metal']):
                                return 'plastic'  # Glass often goes with plastic recycling
                            else:
                                return 'general'
                
                # Alternative response format
                elif 'class' in result:
                    garbage_type = result['class'].lower()
                    confidence = result.get('confidence', 0)
                    
                    if confidence > 0.5:
                        if any(word in garbage_type for word in ['plastic', 'bottle', 'container']):
                            return 'plastic'
                        elif any(word in garbage_type for word in ['paper', 'cardboard', 'document']):
                            return 'paper'
                        elif any(word in garbage_type for word in ['electronic', 'battery', 'phone']):
                            return 'electronics'
                        elif any(word in garbage_type for word in ['food', 'organic', 'fruit', 'vegetable']):
                            return 'food'
                        else:
                            return 'general'
            
            # Fallback if API fails
            st.warning(f"ReciclAPI failed with status {response.status_code}. Using enhanced analysis.")
            img_array = np.array(image)
            return self._simple_analysis(img_array)
            
        except Exception as e:
            st.warning(f"ReciclAPI classification failed: {e}")
            # Final fallback
            img_array = np.array(image)
            return self._simple_analysis(img_array)
    
    def _classify_with_huggingface_free(self, image):
        """Classify using Hugging Face free inference API (no API key required)"""
        try:
            # Convert image to base64
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Use Hugging Face free inference API
            # This uses Microsoft's ResNet-50 model via free API
            url = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
            
            headers = {
                "Authorization": "Bearer hf_your_token_here",  # Not needed for public models
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": img_base64
            }
            
            # Try the free API first
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Map ImageNet classes to trash types
                    for result in results:
                        label = result['label'].lower()
                        score = result['score']
                        
                        if score > 0.3:  # Confidence threshold
                            # Map common ImageNet classes to trash types
                            if any(word in label for word in ['bottle', 'container', 'drink', 'plastic']):
                                return 'plastic'
                            elif any(word in label for word in ['paper', 'document', 'cardboard', 'book']):
                                return 'paper'
                            elif any(word in label for word in ['phone', 'computer', 'electronic', 'device', 'laptop']):
                                return 'electronics'
                            elif any(word in label for word in ['food', 'apple', 'banana', 'fruit', 'vegetable', 'bread']):
                                return 'food'
                            elif any(word in label for word in ['glass', 'cup', 'bowl']):
                                return 'plastic'  # Glass often goes with plastic recycling
                
            except Exception as api_error:
                st.warning(f"Hugging Face API failed: {api_error}")
            
            # Fallback: Use a simpler approach with image analysis
            # Convert to RGB and analyze
            img_rgb = image.convert('RGB')
            img_array = np.array(img_rgb)
            
            # Simple but effective analysis
            avg_color = np.mean(img_array, axis=(0, 1))
            color_variance = np.var(img_array, axis=(0, 1))
            avg_variance = np.mean(color_variance)
            
            # Better heuristics based on color and texture
            if avg_variance < 500:  # Very smooth (likely plastic bottle)
                if avg_color[0] > 100 or avg_color[1] > 100 or avg_color[2] > 100:
                    return 'plastic'
            elif avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                return 'paper'  # Very light colors
            elif avg_color[0] < 80 and avg_color[1] < 80 and avg_color[2] < 80:
                return 'electronics'  # Dark colors
            elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                return 'food'  # Green dominant
            elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                return 'food'  # Red/orange dominant
            else:
                return 'general'
            
        except Exception as e:
            st.warning(f"Free Hugging Face classification failed: {e}")
            # Final fallback
            img_array = np.array(image)
            return self._simple_analysis(img_array)
    
    def _classify_with_google_vision(self, image):
        """Classify using Google Vision API"""
        try:
            # Convert image to base64 for API
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Google Vision API request
            api_key = st.secrets.get("GOOGLE_VISION_API_KEY", "")
            if not api_key:
                st.warning("Google Vision API key not configured. Using enhanced analysis.")
                img_array = np.array(image)
                return self._simple_analysis(img_array)
            
            url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
            
            payload = {
                "requests": [{
                    "image": {"content": img_base64},
                    "features": [{"type": "LABEL_DETECTION", "maxResults": 10}]
                }]
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                labels = response.json()['responses'][0].get('labelAnnotations', [])
                
                # Map Google Vision labels to trash types
                for label in labels:
                    label_text = label['description'].lower()
                    confidence = label['score']
                    
                    if confidence > 0.7:  # High confidence threshold
                        if any(word in label_text for word in ['bottle', 'plastic', 'container', 'drink']):
                            return 'plastic'
                        elif any(word in label_text for word in ['paper', 'document', 'cardboard']):
                            return 'paper'
                        elif any(word in label_text for word in ['phone', 'computer', 'electronic', 'device']):
                            return 'electronics'
                        elif any(word in label_text for word in ['food', 'apple', 'banana', 'fruit', 'vegetable']):
                            return 'food'
            
            # Fallback if no high-confidence matches
            img_array = np.array(image)
            return self._simple_analysis(img_array)
            
        except Exception as e:
            st.warning(f"Google Vision API failed: {e}")
            img_array = np.array(image)
            return self._simple_analysis(img_array)
    
    def _classify_with_azure_cognitive(self, image):
        """Classify using Azure Cognitive Services Computer Vision API"""
        try:
            # Convert image to base64
            import base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Azure Cognitive Services API
            api_key = st.secrets.get("AZURE_COGNITIVE_API_KEY", "")
            endpoint = st.secrets.get("AZURE_COGNITIVE_ENDPOINT", "")
            
            if not api_key or not endpoint:
                st.warning("Azure Cognitive Services API not configured. Using enhanced analysis.")
                img_array = np.array(image)
                return self._simple_analysis(img_array)
            
            url = f"{endpoint}/vision/v3.2/analyze?visualFeatures=Tags"
            
            headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'application/octet-stream'
            }
            
            response = requests.post(url, headers=headers, data=buffer.getvalue())
            
            if response.status_code == 200:
                result = response.json()
                tags = result.get('tags', [])
                
                # Map Azure tags to trash types
                for tag in tags:
                    tag_name = tag['name'].lower()
                    confidence = tag['confidence']
                    
                    if confidence > 0.7:
                        if any(word in tag_name for word in ['bottle', 'plastic', 'container']):
                            return 'plastic'
                        elif any(word in tag_name for word in ['paper', 'document']):
                            return 'paper'
                        elif any(word in tag_name for word in ['phone', 'computer', 'electronic']):
                            return 'electronics'
                        elif any(word in tag_name for word in ['food', 'fruit', 'vegetable']):
                            return 'food'
            
            # Fallback
            img_array = np.array(image)
            return self._simple_analysis(img_array)
            
        except Exception as e:
            st.warning(f"Azure Cognitive Services API failed: {e}")
            img_array = np.array(image)
            return self._simple_analysis(img_array)
    
    def _classify_with_aws_rekognition(self, image):
        """Classify using AWS Rekognition API"""
        try:
            # Convert image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            
            # AWS Rekognition (requires boto3)
            try:
                import boto3
            except ImportError:
                st.warning("boto3 not installed. Install with: pip install boto3")
                img_array = np.array(image)
                return self._simple_analysis(img_array)
            
            # Configure AWS credentials
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", "")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
            aws_region = st.secrets.get("AWS_REGION", "us-east-1")
            
            if not aws_access_key or not aws_secret_key:
                st.warning("AWS credentials not configured. Using enhanced analysis.")
                img_array = np.array(image)
                return self._simple_analysis(img_array)
            
            client = boto3.client(
                'rekognition',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            
            response = client.detect_labels(
                Image={'Bytes': img_bytes},
                MaxLabels=10,
                MinConfidence=70
            )
            
            labels = response.get('Labels', [])
            
            # Map AWS Rekognition labels to trash types
            for label in labels:
                label_name = label['Name'].lower()
                confidence = label['Confidence']
                
                if confidence > 70:
                    if any(word in label_name for word in ['bottle', 'plastic', 'container']):
                        return 'plastic'
                    elif any(word in label_name for word in ['paper', 'document']):
                        return 'paper'
                    elif any(word in label_name for word in ['phone', 'computer', 'electronic']):
                        return 'electronics'
                    elif any(word in label_name for word in ['food', 'fruit', 'vegetable']):
                        return 'food'
            
            # Fallback
            img_array = np.array(image)
            return self._simple_analysis(img_array)
            
        except Exception as e:
            st.warning(f"AWS Rekognition API failed: {e}")
            img_array = np.array(image)
            return self._simple_analysis(img_array)
    
    def _analyze_image(self, img_array):
        """
        Analyze the image to determine trash type
        
        Args:
            img_array: numpy array of the image
            
        Returns:
            str: Detected trash type or None
        """
        if not CV2_AVAILABLE:
            # Fallback to simple analysis without OpenCV
            return self._simple_analysis(img_array)
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Simple feature detection
        features = self._extract_features(img_array, hsv, gray)
        
        # Classification based on features
        detected_type = self._classify_by_features(features)
        
        return detected_type
    
    def _extract_features(self, img_array, hsv, gray):
        """Extract basic features from the image"""
        features = {}
        
        # Color analysis
        features['avg_hue'] = np.mean(hsv[:, :, 0])
        features['avg_saturation'] = np.mean(hsv[:, :, 1])
        features['avg_value'] = np.mean(hsv[:, :, 2])
        
        # Shape analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['area'] = cv2.contourArea(largest_contour)
            features['perimeter'] = cv2.arcLength(largest_contour, True)
            features['aspect_ratio'] = self._get_aspect_ratio(largest_contour)
        else:
            features['area'] = 0
            features['perimeter'] = 0
            features['aspect_ratio'] = 0
        
        # Texture analysis
        features['texture'] = self._analyze_texture(gray)
        
        return features
    
    def _simple_analysis(self, img_array):
        """Enhanced analysis with better heuristics for plastic bottle detection"""
        height, width = img_array.shape[:2]
        
        # Analyze different regions of the image
        top_region = img_array[:height//3, :]
        middle_region = img_array[height//3:2*height//3, :]
        bottom_region = img_array[2*height//3:, :]
        
        # Color analysis for each region
        top_color = np.mean(top_region, axis=(0, 1))
        middle_color = np.mean(middle_region, axis=(0, 1))
        bottom_color = np.mean(bottom_region, axis=(0, 1))
        
        # Calculate color variance (plastic bottles often have smooth colors)
        color_variance = np.var(img_array, axis=(0, 1))
        avg_variance = np.mean(color_variance)
        
        # Detect bottle-like shapes (cylindrical objects)
        # Look for vertical lines and smooth color transitions
        gray = np.mean(img_array, axis=2)
        
        # Edge detection for shape analysis
        edges = np.abs(np.diff(gray, axis=1))
        vertical_edges = np.sum(edges, axis=0)
        horizontal_edges = np.sum(np.abs(np.diff(gray, axis=0)), axis=1)
        
        # Bottle detection heuristics
        has_vertical_lines = np.max(vertical_edges) > np.mean(vertical_edges) * 1.5
        has_smooth_colors = avg_variance < 1000  # Low variance = smooth colors
        is_cylindrical = len(np.where(vertical_edges > np.mean(vertical_edges) * 1.2)[0]) > width * 0.1
        
        # Color-based classification
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Plastic bottle detection
        if (has_vertical_lines and has_smooth_colors and 
            (avg_color[0] > 100 or avg_color[1] > 100 or avg_color[2] > 100)):  # Not too dark
            return 'plastic'
        
        # Paper detection (white/light colors, high variance, text-like patterns)
        elif (avg_color[0] > 180 and avg_color[1] > 180 and avg_color[2] > 180 and 
              avg_variance > 2000 and not has_vertical_lines):
            return 'paper'
        
        # Electronics detection (dark colors, metallic)
        elif (avg_color[0] < 80 and avg_color[1] < 80 and avg_color[2] < 80):
            return 'electronics'
        
        # Food detection (organic colors - greens, browns, oranges, reds)
        elif ((avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2] and avg_color[1] > 100) or  # Green dominant
              (avg_color[0] > 120 and avg_color[1] > 80 and avg_color[2] < 100) or  # Orange/brown
              (avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2] and avg_color[0] > 120) or  # Red/orange
              (avg_color[0] > 150 and avg_color[1] < 100 and avg_color[2] < 100)):  # Red
            return 'food'
        
        # Glass detection (transparent/reflective)
        elif (avg_color[0] > 150 and avg_color[1] > 150 and avg_color[2] > 150 and 
              avg_variance < 1500 and has_smooth_colors):
            return 'plastic'  # Glass often gets classified as plastic
        
        # Default fallback
        else:
            # More intelligent fallback based on dominant color
            if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                return 'plastic' if avg_color[0] > 100 else 'electronics'
            elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                return 'food' if avg_color[1] > 100 else 'general'
            elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                return 'plastic' if avg_color[2] > 100 else 'electronics'
            else:
                return 'general'
    
    def _get_aspect_ratio(self, contour):
        """Calculate aspect ratio of the contour"""
        if not CV2_AVAILABLE:
            return 1.0
        _, _, w, h = cv2.boundingRect(contour)
        return w / h if h > 0 else 0
    
    def _analyze_texture(self, gray):
        """Analyze texture of the image"""
        # Calculate standard deviation as texture measure
        return np.std(gray)
    
    def _classify_by_features(self, features):
        """
        Classify trash type based on extracted features
        
        This is a simplified classification system.
        In a real implementation, you would use a trained ML model.
        """
        # Simple rule-based classification
        avg_hue = features['avg_hue']
        avg_saturation = features['avg_saturation']
        avg_value = features['avg_value']
        area = features['area']
        aspect_ratio = features['aspect_ratio']
        texture = features['texture']
        
        # Plastic detection (bottles, containers)
        if (avg_hue < 30 or avg_hue > 150) and avg_saturation > 50 and area > 1000:
            if aspect_ratio > 0.5 and aspect_ratio < 2.0:
                return 'plastic'
        
        # Paper detection (white/light colors, rectangular)
        if avg_value > 150 and avg_saturation < 50:
            if aspect_ratio > 1.2 or aspect_ratio < 0.8:
                return 'paper'
        
        # Electronics detection (dark colors, metallic)
        if avg_value < 100 and texture > 30:
            return 'electronics'
        
        # Food detection (organic colors, irregular shapes)
        if 20 < avg_hue < 80 and avg_saturation > 40:
            if aspect_ratio > 0.3 and aspect_ratio < 3.0:
                return 'food'
        
        # If no specific pattern matches, return general
        # Add some randomness for demo purposes
        if random.random() < 0.3:  # 30% chance of successful detection
            return random.choice(self.trash_types)
        
        return None
    
    def get_classification_confidence(self, _):
        """
        Get confidence score for the classification
        
        Args:
            _: Unused parameter (kept for compatibility)
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # This is a mock confidence calculation
        # In a real implementation, this would come from the ML model
        return random.uniform(0.4, 0.9)
    
    def get_supported_types(self):
        """Get list of supported trash types"""
        return self.trash_types.copy()
    
    def add_training_data(self, _, label):
        """
        Add training data for future model improvement
        
        Args:
            _: Unused parameter (kept for compatibility)
            label: str - correct classification label
        """
        # In a real implementation, this would save the image and label
        # for retraining the model
        st.info(f"Training data added: {label}")
    
    def reset_classifier(self):
        """Reset the classifier to default state"""
        # In a real implementation, this might reload the model
        st.info("Classifier reset")
    
    def set_model_type(self, model_type):
        """Set the classification model type"""
        if model_type in ['simple', 'enhanced', 'reciclapi', 'huggingface_free', 'resnet', 'huggingface', 'google_vision', 'azure_cognitive', 'aws_rekognition']:
            self.model_type = model_type
            self._initialize_model()
            st.success(f"Model type set to: {model_type}")
        else:
            st.error(f"Invalid model type: {model_type}")
    
    def get_model_info(self):
        """Get information about the current model"""
        info = {
            'model_type': self.model_type,
            'torch_available': TORCH_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'cv2_available': CV2_AVAILABLE
        }
        return info
