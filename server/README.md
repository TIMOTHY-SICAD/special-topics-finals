# Traffic Sign Recognition - Backend

FastAPI-based backend for CNN traffic sign classification with support for both German (GTSRB) and Philippine road signs.

## Model Performance (GTSRB)

The CNN model has been trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset with **43 traffic sign classes**.

### Evaluation Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.31% |
| **Precision** | 96.96% |
| **Recall** | 96.31% |
| **F1-Score** | 96.26% |

**Most Confused Class Pairs:**
- General Danger → Traffic Signals: 17 misclassifications
- Curve Left → Double Curve: 17 misclassifications
- Speed Limit 70 → Speed Limit 30: 8 misclassifications

**Worst Performing Classes:**
- Curve Left: 63.04% accuracy (F1=0.75)
- General Danger: 63.04% accuracy (F1=0.77)
- Double Curve: 91.30% accuracy (F1=0.79)

Results are saved in `server/results/`:
- `evaluation_results.json` - Full metrics in JSON format
- `confusion_matrix.png` - Confusion matrix visualization
- `per_class_accuracy.png` - Per-class accuracy chart
- `error_analysis.txt` - Detailed error analysis report

## Philippine Road Signs Integration

### Phase 1: Data Collection (In Progress)

Download Philippine road signs from Wikimedia Commons using `gallery-dl`:

```powershell
# Install dependencies first
pip install -r requirements.txt

# Run the download script
python download_philippine_signs.py
```

This will:
- Download images from: https://commons.wikimedia.org/wiki/Category:Road_signs_in_the_Philippines
- Filter for common formats: JPEG, PNG, GIF, WebP
- Skip SVG files (would require rasterization)
- Save to: `server/data/Philippine_Signs/`

### Phase 2: Format Compatibility (Completed)

Updated preprocessing pipeline to handle multiple image formats:
- **Supported formats**: PPM, JPEG, PNG, GIF, WebP, BMP, TIFF
- Uses OpenCV as primary loader, PIL as fallback
- Handles RGBA images with transparency
- Gracefully skips problematic files

Usage:
```python
from utils.preprocessing import load_and_preprocess_image, batch_preprocess_images

# Load single image (supports all formats)
image = load_and_preprocess_image("path/to/image.jpg")

# Batch load with error handling
images = batch_preprocess_images(image_paths)
```

### Phase 3: Cross-Dataset Evaluation (Planned)

Test GTSRB model on Philippine signs to evaluate transfer learning:
```python
from utils.philippine_signs import load_philippine_signs_dataset

# Load Philippine signs
images, file_paths = load_philippine_signs_dataset()

# Split into train/test
(train_imgs, train_paths), (test_imgs, test_paths) = split_dataset(images, file_paths)
```

## Setup

1. **Install dependencies:**
```powershell
cd server
pip install -r requirements.txt
```

2. **Download the GTSRB dataset:**
   - Training: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
   - Test: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
   - Extract to `server/data/`
   - Expected structure:
     ```
     server/data/Final_Training/Images/00000/*.ppm  (Speed Limit 20)
     server/data/Final_Training/Images/00001/*.ppm  (Speed Limit 30)
     ...
     server/data/Final_Training/Images/00042/*.ppm (End No Passing >3.5t)
     server/data/Final_Test/Images/*.ppm
     ```

## Usage

### Training the Model

```powershell
python train_model.py
```

This will:
- Load training data from `server/data/Final_Training/Images/`
- Train the CNN model with data augmentation
- Save the model to `server/models/traffic_sign_model.h5`

### Running the API Server

```powershell
python main.py
```

The server will start at `http://localhost:8000`

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Upload image for classification |
| POST | `/predict/base64` | Send base64 encoded image |

### Evaluating the Model

```powershell
python evaluate_model.py
```

This will:
- Load test data from `server/data/Final_Test/Images/`
- Compute accuracy, precision, recall, F1-score
- Generate confusion matrix and error analysis
- Save results to `server/results/`

## API Usage Examples

### Using cURL

```bash
# Predict from image file
curl -X POST -F "file=@traffic_sign.jpg" http://localhost:8000/predict

# Health check
curl http://localhost:8000/health
```

### Using Python

```python
import requests

# Upload image for prediction
with open("traffic_sign.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## Project Structure

```
server/
├── main.py              # FastAPI application
├── train_model.py       # CNN training script
├── evaluate_model.py    # Model evaluation script
├── requirements.txt     # Python dependencies
├── models/              # Trained model storage
│   └── traffic_sign_model.h5
├── utils/
│   └── preprocessing.py # Image preprocessing utilities
├── data/                # Dataset storage
│   ├── Final_Training/  # Training images (GTSRB)
│   └── Final_Test/      # Test images (GTSRB)
└── results/             # Evaluation results
    ├── evaluation_results.json
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    └── error_analysis.txt
```

## Model Architecture

The CNN model uses:
- 3 convolutional blocks with batch normalization
- MaxPooling and Dropout for regularization
- Dense layers for classification
- L2 regularization for weight decay

Input: 32x32 RGB images
Output: 43 classes (GTSRB traffic signs)

## Notes

- The model expects images resized to 32x32 pixels
- Pixel values are normalized to [0, 1] range
- Class labels follow the GTSRB dataset convention
- The API supports both file upload and base64 encoded images