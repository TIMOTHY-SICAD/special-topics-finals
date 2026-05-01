# Traffic & Road Signs Detection - YOLO Implementation

A full-stack web application for detecting traffic and road signs using YOLOv8 object detection model. The project consists of a React frontend for image upload and a FastAPI backend for model inference.

## Project Overview

This is a special topics final project implementing real-time traffic and road sign detection. Users can upload images through a web interface, and the application returns annotated images highlighting detected signs with bounding boxes and confidence scores.

**Tech Stack:**
- **Frontend:** React 19 + Vite (with React Compiler)
- **Backend:** FastAPI + UltraLytics YOLO
- **ML Model:** YOLOv8 (nano) trained on traffic and road signs dataset

## Project Structure

```
.
├── src/                          # React frontend
│   ├── App.jsx                  # Main React component (image upload UI)
│   ├── App.css                  # Styling
│   ├── main.jsx                 # React entry point
│   └── index.css                # Global styles
│
├── server/                       # Python backend
│   ├── main.py                  # FastAPI server with /predict endpoint
│   ├── model/
│   │   └── train.py             # YOLOv8 model training script
│   ├── requirements.txt          # Python dependencies
│   └── data/                    # Training/test datasets
│
├── public/                       # Static assets
├── package.json                  # Frontend dependencies
├── vite.config.js               # Vite configuration
├── eslint.config.js             # ESLint configuration
└── index.html                    # HTML entry point
```

## Features

- **Image Upload & Detection:** Upload images through web interface to detect traffic signs
- **Real-time Annotation:** Server returns images with bounding boxes around detected signs
- **YOLO Model:** Uses pre-trained YOLOv8 (nano) model fine-tuned on traffic sign data
- **CORS Enabled:** Backend allows cross-origin requests for development

## Getting Started

### Prerequisites

- Node.js (v16+) for frontend
- Python 3.8+ for backend
- Virtual environment setup recommended for Python

### Frontend Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

The React app will be available at `http://localhost:5173` (Vite default port).

### Backend Setup

```bash
# Navigate to server directory
cd server

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt is populated)
pip install fastapi uvicorn ultralytics pillow opencv-python

# Run the FastAPI server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Using the Application

1. Start both servers (frontend and backend)
2. Navigate to `http://localhost:5173` in your browser
3. Click "Choose file" to upload an image
4. The original image will display on the left
5. Detected signs with bounding boxes will display on the right

## Model Training

To retrain the YOLO model with different data:

```bash
cd server
python model/train.py
```

The training script expects:
- Dataset in YOLOv8 format with `data.yaml` configuration
- Images and annotations organized by the data.yaml specification
- Trained weights saved to `model/traffic_sign_v2/weights/best.pt`

Current configuration trains for 30 epochs at 640x640 image size using YOLOv8 nano model.

## API Endpoints

### POST `/predict/`

Upload an image for sign detection.

**Request:**
- Body: `multipart/form-data` with `file` field containing image

**Response:**
- Returns annotated image (JPEG format) with bounding boxes around detected signs

**Example:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict/ -o result.jpg
```

## Known Limitations & Future Work

- Video detection not yet implemented (UI prepared for future video support)
- Confidence threshold currently not exposed as parameter
- No image size optimization - all images processed at 640x640
- GPU support available but commented out (requires CUDA setup)

## Development Notes

- The React Compiler is enabled for optimized builds (may impact Vite dev performance)
- ESLint is configured with React-specific rules
- CORS is set to allow all origins for development convenience (should be restricted in production)
