from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock dependencies that might be hard to install in test env
sys.modules['paddleocr'] = MagicMock()
sys.modules['paddle'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Mock internal modules that allow importing api
with patch('src.ocr.load_ocr_model') as mock_load:
    mock_load.return_value = (MagicMock(), "mock_ocr")
    from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ocr_ready": True}

def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    assert "version" in response.json()
