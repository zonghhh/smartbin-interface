"""
Simple client for RapidAPI ReciclAPI - Garbage Detection endpoint.

Docs: https://rapidapi.com/roftcomp-laGmBwlWLm/api/reciclapi-garbage-detection

Usage example:
  from tools.reciclapi_client import detect_image
  resp = detect_image('test_bottle.png', rapidapi_key='YOUR_KEY')
  print(resp)

Note: This script uses only the public RapidAPI endpoint format. Provide your RapidAPI key via the rapidapi_key argument or
by setting the environment variable RAPIDAPI_KEY.
"""
from __future__ import annotations
import os
from pathlib import Path
import requests
from typing import Dict, Any, Optional

API_HOST = "reciclapi-garbage-detection.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}/detect"


def detect_image(path: str | Path, rapidapi_key: Optional[str] = None) -> Dict[str, Any]:
    """Send an image file to ReciclAPI Detect endpoint and return parsed JSON response.

    Args:
        path: path to image file
        rapidapi_key: your RapidAPI key (optional; will fallback to RAPIDAPI_KEY env var)
    Returns:
        Parsed JSON response from API
    Raises:
        FileNotFoundError if path not found
        requests.HTTPError on non-2xx responses
    """
    rapidapi_key = rapidapi_key or os.environ.get("RAPIDAPI_KEY")
    if not rapidapi_key:
        raise ValueError("Provide rapidapi_key or set RAPIDAPI_KEY environment variable")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with p.open("rb") as f:
        files = {"image": (p.name, f, "application/octet-stream")}
        headers = {
            "X-RapidAPI-Key": rapidapi_key,
            "X-RapidAPI-Host": API_HOST,
        }
        resp = requests.post(BASE_URL, headers=headers, files=files, timeout=30)
        if not resp.ok:
            # raise with response body helpful message
            try:
                _body = resp.json()
                err_text = str(_body)
            except ValueError:
                err_text = resp.text
            resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    # quick smoke test (won't run an actual API call unless RAPIDAPI_KEY exists)
    test_key = os.environ.get("RAPIDAPI_KEY")
    if not test_key:
        print("Set RAPIDAPI_KEY environment variable to run a live test.")
    else:
        # attempt to call using bundled test image if available
        test_img = Path(__file__).resolve().parents[1] / "test_bottle.png"
        if test_img.exists():
            print("Running live detect on test_bottle.png...")
            print(detect_image(test_img, rapidapi_key=test_key))
        else:
            print("No test image found (test_bottle.png). Provide a path to run detect.")
