# CLI Prediction Tool

## Overview

The `predict_cli.py` script provides a command-line interface for testing the trained traffic sign recognition model directly with local images. This eliminates the need to start the FastAPI server for quick model validation and testing.

## Features

- 🚀 **Direct Model Testing**: Load images and get predictions without server overhead
- 📊 **Top-K Predictions**: View top N predictions with confidence scores
- 📈 **Visual Output**: Bar charts showing prediction probabilities
- 💾 **JSON Export**: Save detailed results to JSON for further analysis
- ✅ **Error Handling**: Comprehensive validation and error messages

## Installation

No additional dependencies beyond the main project requirements. The script uses:
- `numpy` - Numerical operations
- `cv2` (OpenCV) - Image processing
- `tensorflow/keras` - Model loading (from main project)

## Usage

### Basic Prediction
```bash
python predict_cli.py --image path/to/traffic_sign.jpg
```

### View Top 10 Predictions
```bash
python predict_cli.py --image sign.jpg --top 10
```

### Save Results to JSON
```bash
python predict_cli.py --image sign.jpg --output results.json
```

### Verbose Output
```bash
python predict_cli.py --image sign.jpg --verbose
```

## Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--image` | `-i` | str | **required** | Path to image file to test |
| `--top` | `-t` | int | 5 | Number of top predictions to show |
| `--output` | `-o` | str | None | Optional: Save results to JSON file |
| `--verbose` | `-v` | flag | False | Enable verbose output |
| `--help` | `-h` | flag | - | Show help message |

## Example Output

```
======================================================================
Traffic Sign Recognition - CLI Prediction Tool
======================================================================
✓ Model found: models/traffic_sign_model.h5
✓ Model loaded successfully

Loading image: test_sign.jpg
✓ Image loaded: test_sign.jpg
  Original size: 200x200 pixels
✓ Image preprocessed: resized to (32, 32)

Running prediction...
✓ Prediction complete

======================================================================
PREDICTION RESULTS
======================================================================

🎯 Primary Prediction:
   Class ID: 14
   Label: Stop
   Confidence: 98.45%

📊 Top 5 Predictions:
   1. [████████████████████] 98.45% - Class 14: Stop
   2. [█                   ] 1.23% - Class 15: No vehicles
   3. [                    ] 0.18% - Class 13: Yield
   4. [                    ] 0.10% - Class 17: No entry
   5. [                    ] 0.04% - Class 18: General danger

======================================================================
```

## JSON Output Format

When using `--output`, results are saved in the following format:

```json
{
  "image": "path/to/image.jpg",
  "primary_prediction": {
    "class": 14,
    "label": "Stop",
    "confidence": 0.9845
  },
  "top_predictions": [
    {
      "rank": 1,
      "class": 14,
      "label": "Stop",
      "probability": 0.9845
    },
    ...
  ],
  "all_probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    ...
    "14": 0.9845,
    ...
  }
}
```

## Use Cases

1. **Quick Model Validation**: Test model with new images before deployment
2. **Batch Testing**: Combine with shell scripts to test multiple images
3. **Debugging**: JSON output helps identify prediction patterns and edge cases
4. **Integration**: Use JSON output in automated testing pipelines
5. **Demonstration**: Show model capabilities with specific examples

## Error Handling

The script provides clear error messages for:
- Missing model file: Guide user to train model first
- Invalid image path: Verify file location
- Corrupted image files: Check image format compatibility
- Missing dependencies: Install required packages

## Performance Notes

- First run loads the model (~2-5 seconds depending on system)
- Subsequent runs within same session reuse loaded model
- Image preprocessing is fast (~10-100ms)
- Model inference typically completes in 10-50ms

## Integration with Workflow

This CLI tool complements the FastAPI server:
- **Development**: Use CLI for quick testing during model development
- **Testing**: Automate image prediction testing with shell scripts
- **Production**: Use FastAPI server for web/API access
- **CI/CD**: Integrate CLI predictions into automated pipelines

## Example: Batch Testing

```bash
#!/bin/bash
# Test all images in a directory

for image in test_images/*.jpg; do
    echo "Testing: $image"
    python predict_cli.py --image "$image" --output "results/$(basename $image).json"
done
```

## Troubleshooting

**Q: "Model not found" error**
- A: Train the model first using the training script
- Make sure `models/traffic_sign_model.h5` exists

**Q: "Failed to read image" error**
- A: Verify image file path and format (JPG, PNG supported)

**Q: Model loads slowly**
- A: First load is slower. Model is cached in memory afterward.
- Consider running predictions in batches to amortize load time.

## Future Enhancements

- [ ] Support for batch image directories
- [ ] Confidence threshold filtering
- [ ] Image augmentation preview before prediction
- [ ] Model uncertainty estimation
- [ ] Integration with visualization tools

---

**Created**: 2024
**Version**: 1.0.0
