#!/usr/bin/env python3
"""
CLI Tool for Testing Traffic Sign Recognition Model

Allows direct testing of the trained CNN model with local image files
without needing to start the FastAPI server.

Usage:
    python predict_cli.py --image path/to/image.jpg
    python predict_cli.py --image path/to/image.jpg --top 5
"""

import argparse
import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_sign_model.h5")
IMG_SIZE = (32, 32)
NUM_CLASSES = 43

# GTSRB class labels
CLASS_LABELS = {
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


def validate_model_exists() -> bool:
    """Check if trained model exists."""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at: {MODEL_PATH}")
        logger.error("   Please train the model first using the training script.")
        return False
    logger.info(f"✓ Model found: {MODEL_PATH}")
    return True


def load_model():
    """Load trained Keras model."""
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(MODEL_PATH)
        logger.info("✓ Model loaded successfully")
        return model
    except ImportError:
        logger.error("❌ TensorFlow/Keras not installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess image for model inference.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (preprocessed image array, original image for display)
    """
    # Read image
    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        sys.exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"❌ Failed to read image: {image_path}")
        sys.exit(1)
    
    original_img = img.copy()
    logger.info(f"✓ Image loaded: {image_path}")
    logger.info(f"  Original size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Resize to model input size
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    logger.info(f"✓ Image preprocessed: resized to {IMG_SIZE}")
    
    return img_batch, original_img


def get_class_label(class_id: int) -> str:
    """Get human-readable label for traffic sign class."""
    return CLASS_LABELS.get(class_id, f"Unknown Class {class_id}")


def predict(model, image_batch: np.ndarray) -> np.ndarray:
    """Run model prediction."""
    logger.info("Running prediction...")
    predictions = model.predict(image_batch, verbose=0)
    logger.info("✓ Prediction complete")
    return predictions[0]


def format_output(predictions: np.ndarray, top_k: int = 5) -> str:
    """Format prediction results for display."""
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    predicted_label = get_class_label(predicted_class)
    
    # Get top K predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    output = []
    output.append("\n" + "="*70)
    output.append("PREDICTION RESULTS")
    output.append("="*70)
    output.append(f"\n🎯 Primary Prediction:")
    output.append(f"   Class ID: {predicted_class}")
    output.append(f"   Label: {predicted_label}")
    output.append(f"   Confidence: {confidence*100:.2f}%")
    
    output.append(f"\n📊 Top {top_k} Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        prob = predictions[idx] * 100
        label = get_class_label(idx)
        bar_length = int(prob / 5)  # Scale to reasonable bar length
        bar = "█" * bar_length
        output.append(f"   {rank}. [{bar:20s}] {prob:5.2f}% - Class {idx}: {label}")
    
    output.append("\n" + "="*70)
    return "\n".join(output)


def save_json_output(predictions: np.ndarray, image_path: str, output_file: str):
    """Save predictions to JSON file."""
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    predicted_label = get_class_label(predicted_class)
    
    # Top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_predictions = [
        {
            "rank": rank,
            "class": int(idx),
            "label": get_class_label(idx),
            "probability": float(predictions[idx])
        }
        for rank, idx in enumerate(top_indices, 1)
    ]
    
    result = {
        "image": image_path,
        "primary_prediction": {
            "class": predicted_class,
            "label": predicted_label,
            "confidence": confidence
        },
        "top_predictions": top_predictions,
        "all_probabilities": {str(i): float(predictions[i]) for i in range(len(predictions))}
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI Tool for Testing Traffic Sign Recognition Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_cli.py --image traffic_sign.jpg
  python predict_cli.py --image ./data/test.png --top 10
  python predict_cli.py --image sign.jpg --output results.json
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to image file to test"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Traffic Sign Recognition - CLI Prediction Tool")
    logger.info("="*70)
    
    # Validate model exists
    if not validate_model_exists():
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Preprocess image
    logger.info(f"\nLoading image: {args.image}")
    img_batch, original_img = preprocess_image(args.image)
    
    # Run prediction
    logger.info("")
    predictions = predict(model, img_batch)
    
    # Format and display results
    output = format_output(predictions, top_k=args.top)
    print(output)
    
    # Save JSON if requested
    if args.output:
        save_json_output(predictions, args.image, args.output)
    
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
