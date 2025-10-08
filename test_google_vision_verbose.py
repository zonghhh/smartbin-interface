import requests, io, base64, json
from PIL import Image
import streamlit as st
import os

# Try Streamlit secrets first, then environment variable
api_key = ''
try:
    api_key = st.secrets.get('GOOGLE_VISION_API_KEY', '')
except Exception:
    api_key = ''

if not api_key:
    api_key = os.environ.get('GOOGLE_VISION_API_KEY', '')

if not api_key:
    # Try to parse .streamlit/secrets.toml manually as a fallback
    try:
        with open('.streamlit/secrets.toml', 'r', encoding='utf-8') as f:
            for line in f:
                if 'GOOGLE_VISION_API_KEY' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        val = parts[1].strip()
                        if val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        api_key = val
                        break
    except Exception:
        api_key = ''

if not api_key:
    print('No GOOGLE_VISION_API_KEY found in st.secrets, environment, or .streamlit/secrets.toml')
    raise SystemExit(1)
else:
    print('Found GOOGLE_VISION_API_KEY (redacted):', api_key[:6] + '...' + api_key[-4:])

img_path = 'test_bottle.png'
try:
    img = Image.open(img_path)
except Exception as e:
    print('Failed to open', img_path, e)
    raise SystemExit(1)

buf = io.BytesIO()
img.save(buf, format='JPEG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
payload = {
    'requests': [{
        'image': {'content': img_b64},
        'features': [
            {'type': 'LABEL_DETECTION', 'maxResults': 10},
            {'type': 'OBJECT_LOCALIZATION', 'maxResults': 10}
        ]
    }]
}

print('Sending request to Google Vision...')
try:
    r = requests.post(url, json=payload, timeout=30)
    print('HTTP', r.status_code)
    try:
        data = r.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print('Failed to parse JSON response:', e)
        print(r.text)
except Exception as e:
    print('Request failed:', e)
