#!/usr/bin/env python3
"""
API Configuration Setup Script
Helps configure external APIs for better image classification
"""

import os
import sys

def create_secrets_file():
    """Create Streamlit secrets file for API configuration"""
    
    print("External API Configuration Setup")
    print("=" * 40)
    
    # Check if .streamlit directory exists
    if not os.path.exists('.streamlit'):
        os.makedirs('.streamlit')
        print("[OK] Created .streamlit directory")
    
    secrets_file = '.streamlit/secrets.toml'
    
    print("\nAvailable APIs:")
    print("1. ReciclAPI - Garbage Detection (Recommended)")
    print("2. Google Vision API")
    print("3. Azure Cognitive Services")
    print("4. AWS Rekognition")
    print("5. Skip (use enhanced model only)")
    
    choice = input("\nSelect API to configure (1-5): ").strip()
    
    if choice == '1':
        setup_reciclapi(secrets_file)
    elif choice == '2':
        setup_google_vision(secrets_file)
    elif choice == '3':
        setup_azure_cognitive(secrets_file)
    elif choice == '4':
        setup_aws_rekognition(secrets_file)
    elif choice == '5':
        print("[OK] Skipping API setup. Enhanced model will be used.")
        return
    else:
        print("[ERROR] Invalid choice. Skipping setup.")
        return

def setup_reciclapi(secrets_file):
    """Setup ReciclAPI configuration"""
    print("\nReciclAPI - Garbage Detection Setup")
    print("-" * 40)
    
    print("1. Go to: https://rapidapi.com/")
    print("2. Sign up for a free account")
    print("3. Go to: https://rapidapi.com/reciclapi/api/reciclapi-garbage-detection")
    print("4. Subscribe to the API (free tier available)")
    print("5. Copy your RapidAPI key")
    
    api_key = input("\nEnter your RapidAPI key: ").strip()
    
    if api_key:
        with open(secrets_file, 'w') as f:
            f.write(f'RAPIDAPI_KEY = "{api_key}"\n')
        print("[OK] ReciclAPI configured!")
        print("[INFO] You can now select 'ReciclAPI' in the app")
        print("[INFO] This will give you 90%+ accuracy for trash detection!")
    else:
        print("[ERROR] No API key provided. Skipping setup.")

def setup_google_vision(secrets_file):
    """Setup Google Vision API configuration"""
    print("\nGoogle Vision API Setup")
    print("-" * 25)
    
    print("1. Go to: https://console.cloud.google.com/")
    print("2. Create/select a project")
    print("3. Enable 'Cloud Vision API'")
    print("4. Create an API key")
    print("5. Copy the API key")
    
    api_key = input("\nEnter your Google Vision API key: ").strip()
    
    if api_key:
        with open(secrets_file, 'w') as f:
            f.write(f'GOOGLE_VISION_API_KEY = "{api_key}"\n')
        print("[OK] Google Vision API configured!")
        print("[INFO] You can now select 'Google Vision' in the app")
    else:
        print("[ERROR] No API key provided. Skipping setup.")

def setup_azure_cognitive(secrets_file):
    """Setup Azure Cognitive Services configuration"""
    print("\nAzure Cognitive Services Setup")
    print("-" * 35)
    
    print("1. Go to: https://portal.azure.com/")
    print("2. Create a 'Computer Vision' resource")
    print("3. Get the API key and endpoint")
    
    api_key = input("\nEnter your Azure API key: ").strip()
    endpoint = input("Enter your Azure endpoint: ").strip()
    
    if api_key and endpoint:
        with open(secrets_file, 'w') as f:
            f.write(f'AZURE_COGNITIVE_API_KEY = "{api_key}"\n')
            f.write(f'AZURE_COGNITIVE_ENDPOINT = "{endpoint}"\n')
        print("[OK] Azure Cognitive Services configured!")
        print("[INFO] You can now select 'Azure Cognitive' in the app")
    else:
        print("[ERROR] Missing API key or endpoint. Skipping setup.")

def setup_aws_rekognition(secrets_file):
    """Setup AWS Rekognition configuration"""
    print("\nAWS Rekognition Setup")
    print("-" * 25)
    
    print("1. Go to: https://aws.amazon.com/")
    print("2. Create IAM user with Rekognition access")
    print("3. Create access key")
    print("4. Install boto3: pip install boto3")
    
    access_key = input("\nEnter your AWS Access Key ID: ").strip()
    secret_key = input("Enter your AWS Secret Access Key: ").strip()
    region = input("Enter your AWS Region (default: us-east-1): ").strip() or "us-east-1"
    
    if access_key and secret_key:
        with open(secrets_file, 'w') as f:
            f.write(f'AWS_ACCESS_KEY_ID = "{access_key}"\n')
            f.write(f'AWS_SECRET_ACCESS_KEY = "{secret_key}"\n')
            f.write(f'AWS_REGION = "{region}"\n')
        print("[OK] AWS Rekognition configured!")
        print("[INFO] You can now select 'AWS Rekognition' in the app")
    else:
        print("[ERROR] Missing access key or secret key. Skipping setup.")

def show_current_config():
    """Show current API configuration"""
    secrets_file = '.streamlit/secrets.toml'
    
    if os.path.exists(secrets_file):
        print("\nCurrent Configuration:")
        print("-" * 25)
        with open(secrets_file, 'r') as f:
            content = f.read()
            if 'GOOGLE_VISION_API_KEY' in content:
                print("[OK] Google Vision API: Configured")
            if 'AZURE_COGNITIVE_API_KEY' in content:
                print("[OK] Azure Cognitive Services: Configured")
            if 'AWS_ACCESS_KEY_ID' in content:
                print("[OK] AWS Rekognition: Configured")
    else:
        print("\nNo API configuration found.")
        print("   Using enhanced model only.")

def main():
    """Main setup function"""
    print("Smart Bin Interface - API Setup")
    print("=" * 40)
    
    # Show current config
    show_current_config()
    
    # Ask if user wants to configure
    configure = input("\nConfigure external APIs? (y/n): ").strip().lower()
    
    if configure in ['y', 'yes']:
        create_secrets_file()
    else:
        print("[OK] Skipping API setup.")
    
    print("\nNext Steps:")
    print("1. Run the app: streamlit run main.py")
    print("2. Select your preferred model in the sidebar")
    print("3. Click 'Update Model'")
    print("4. Test with a plastic bottle photo!")
    
    print("\nFor detailed setup instructions, see: EXTERNAL_APIS.md")

if __name__ == "__main__":
    main()