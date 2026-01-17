# Malaysian LPR Microservice

A headless, Dockerized microservice for Malaysian License Plate Recognition. This service processes images to detect, recognize, and identify Malaysian license plates using a 9-phase image processing pipeline and PaddleOCR.

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose installed
- Images to test (supports JPG, PNG, etc.)

### 2. Start the Service
Run the following command in the project directory:
```bash
docker-compose up --build -d
```
The service will be available at `http://localhost:8000`.

### 3. Check Status
Verify the service is running and OCR is initialized:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri http://localhost:8000/health

# Linux/Mac
curl http://localhost:8000/health
```
**Expected Response:** `{"status":"ok","ocr_ready":true}`

---

## 🧪 How to Test

### Option A: Using the Provided Test Script
We've included a Python test script `test_api.py` that you can run locally.

1. **Install requests** (if not installed):
   ```bash
   pip install requests
   ```

2. **Run the test**:
   ```bash
   python test_api.py
   ```
   *Note: Edit `test_api.py` to point to your specific image files if needed.*

### Option B: Using Bruno / Postman
You can use any API client to send requests.

- **Method:** `POST`
- **URL:** `http://localhost:8000/detect`
- **Body Type:** `multipart/form-data`
- **Fields:**
    - `file`: (File) Select your image file
    - `return_phases`: (Text, optional) `true` or `false`
    - `enable_performance_logs`: (Text, optional) `true` or `false` (for timing breakdown)

### Option C: Using Curl
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@/path/to/your/image.jpg" \
     -F "return_phases=false"
```

### Option D: Performance Analysis
To analyze processing time breakdown (optimization experiments):
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@/path/to/your/image.jpg" \
     -F "enable_performance_logs=true"
```
Response will include `performance_logs` with `preprocessing_time_ms` and `ocr_time_ms`.

---

## 📡 API Reference

### `POST /detect`
Detects license plates in a single uploaded image.

**Request:**
- `file`: The image file (multipart/form-data)
- `return_phases`: Boolean (optional). If true, returns base64-encoded images of intermediate processing phases.

### `POST /detect/batch`
Detects license plates in multiple images in a single request.

**Request:**
- `files`: Multiple image files (multipart/form-data)
- `return_phases`: Boolean (optional).

**Response (Batch):**
```json
{
  "total_images": 2,
  "successful": 2,
  "failed": 0,
  "total_processing_time_ms": 2500.0,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "detections": [...],
      "error": null
    },
    {
      "filename": "image2.jpg",
      "success": false,
      "detections": [],
      "error": "Invalid image"
    }
  ]
}
```

### `GET /health`
Health check endpoint. Returns status and OCR readiness.

### `GET /info`
Returns service version and configuration info.

---

## 🛠 Troubleshooting

- **OCR Not Initialized**: If you get a 503 error immediately after starting, wait 10-20 seconds. proper model downloading and initialization takes a moment on the first run.
- **Empty Detections**: If the result keys are empty `[]`, ensuring the image is clear. The service analyzes 10 candidates per image; if none pass validation, the list will be empty.
- **Logs**: To check internal logs:
  ```bash
  docker logs lpr-microservice -f
  ```
