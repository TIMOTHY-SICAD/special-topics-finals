"""
FastAPI Backend for Traffic Sign Recognition
Handles image upload, preprocessing, and model inference
"""

import os
import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Sign Recognition API",
    description="CNN-based traffic sign classification for Intelligent Transportation Systems",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_sign_model.h5")
IMG_SIZE = (32, 32)  # Standard input size for the model
NUM_CLASSES = 43  # GTSRB has 43 classes

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global model variable
model = None


def load_model() -> Any:
    """Load the trained CNN model"""
    global model
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {MODEL_PATH}. Using placeholder.")
                model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
    return model


def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Steps:
    1. Decode image from bytes
    2. Resize to standard input size (32x32)
    3. Convert BGR to RGB (OpenCV uses BGR)
    4. Normalize pixel values to [0, 1]
    5. Add batch dimension
    """
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Resize to standard input size
    img = cv2.resize(img, IMG_SIZE)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def get_class_label(class_id: int) -> str:
    """Get human-readable label for traffic sign class"""
    # GTSRB class labels (common traffic signs)
    class_labels = {
        0: "Speed Limit 20",
        1: "Speed Limit 30",
        2: "Speed Limit 50",
        3: "Speed Limit 60",
        4: "Speed Limit 70",
        5: "Speed Limit 80",
        6: "End of Speed Limit 80",
        7: "Speed Limit 100",
        8: "Speed Limit 120",
        9: "No Passing",
        10: "No Passing for vehicles over 3.5 tons",
        11: "Right-of-way at next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "No vehicles over 3.5 tons",
        17: "No entry",
        18: "General danger",
        19: "Curve left",
        20: "Curve right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children",
        29: "Bicycles",
        30: "Snow or ice",
        31: "Wild animals",
        32: "End of all restrictions",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing for vehicles over 3.5 tons"
    }
    return class_labels.get(class_id, f"Unknown Class {class_id}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()
    logger.info("Traffic Sign Recognition API started")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Traffic Sign Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "model_info": "/model/info (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "input_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    """
    Predict traffic sign class from uploaded image
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        JSON with predicted class, label, and confidence score
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid image format. Use JPG or PNG.")
        
        # Read image data
        image_data = await file.read()
        
        # Check file size (max 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Max size is 10MB.")
        
        # Preprocess image
        preprocessed_img = preprocess_image(image_data)
        
        # Load model if not already loaded
        current_model = load_model()
        
        if current_model is None:
            # Return placeholder response if no model trained yet
            return JSONResponse({
                "message": "Model not trained yet. Please train the model first.",
                "predicted_class": None,
                "predicted_label": None,
                "confidence": None,
                "all_probabilities": []
            })
        
        # Make prediction
        predictions = current_model.predict(preprocessed_img, verbose=0)
        
        # Get top prediction
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = get_class_label(predicted_class)
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                "class": int(idx),
                "label": get_class_label(idx),
                "probability": float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        logger.info(f"Predicted: {predicted_label} (confidence: {confidence:.4f})")
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_5_predictions": top_5_predictions
        }
        
    except ValueError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/base64")
async def predict_sign_base64(data: Dict[str, str]):
    """
    Predict traffic sign from base64 encoded image
    
    Args:
        data: Dictionary with "image" key containing base64 string
    
    Returns:
        JSON with predicted class, label, and confidence score
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request body")
        
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        
        # Preprocess image
        preprocessed_img = preprocess_image(image_data)
        
        # Load model
        current_model = load_model()
        
        if current_model is None:
            return JSONResponse({
                "message": "Model not trained yet",
                "predicted_class": None,
                "predicted_label": None,
                "confidence": None
            })
        
        # Make prediction
        predictions = current_model.predict(preprocessed_img, verbose=0)
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = get_class_label(predicted_class)
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)