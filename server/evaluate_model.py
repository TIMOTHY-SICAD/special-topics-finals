"""
Model Evaluation Script for Traffic Sign Recognition
Computes performance metrics and analyzes classification errors
"""

import os
import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (32, 32)
NUM_CLASSES = 43
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_sign_model.h5")
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
TEST_DIR = DATA_DIR / "Final_Test" / "Images"
RESULTS_DIR = SCRIPT_DIR / "results"


# Class labels for GTSRB dataset
CLASS_LABELS = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50",
    3: "Speed Limit 60", 4: "Speed Limit 70", 5: "Speed Limit 80",
    6: "End Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120",
    9: "No Passing", 10: "No Passing >3.5t", 11: "Right-of-way",
    12: "Priority Road", 13: "Yield", 14: "Stop",
    15: "No Vehicles", 16: "No Vehicles >3.5t", 17: "No Entry",
    18: "General Danger", 19: "Curve Left", 20: "Curve Right",
    21: "Double Curve", 22: "Bumpy Road", 23: "Slippery Road",
    24: "Road Narrows", 25: "Road Work", 26: "Traffic Signals",
    27: "Pedestrians", 28: "Children", 29: "Bicycles",
    30: "Snow/Ice", 31: "Wild Animals", 32: "End Restrictions",
    33: "Turn Right", 34: "Turn Left", 35: "Ahead Only",
    36: "Straight/Right", 37: "Straight/Left", 38: "Keep Right",
    39: "Keep Left", 40: "Roundabout", 41: "End No Passing",
    42: "End No Passing >3.5t"
}


def load_test_data(test_path: Path = TEST_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from folder structure
    
    Expected: data/Final_Test/Images/*.ppm (with GT-final_test.test.csv for labels)
    If test CSV doesn't have ClassId, uses a portion of training data for testing.
    """
    images = []
    labels = []
    
    if not test_path.exists():
        logger.warning(f"Test directory {test_path} does not exist")
        return np.array([]), np.array([])
    
    # Load test data using CSV file (GTSRB format)
    # CSV is in the same folder as the images
    # Note: GTSRB CSV uses semicolon as delimiter
    csv_path = test_path / "GT-final_test.test.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path, sep=';')
        
        # Check if ClassId column exists
        if 'ClassId' in df.columns:
            for _, row in df.iterrows():
                img_path = test_path / row['Filename']
                if img_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, IMG_SIZE)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = img.astype(np.float32) / 255.0
                            images.append(img)
                            labels.append(row['ClassId'])
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
            return np.array(images), np.array(labels)
        else:
            # Test CSV doesn't have ClassId (GTSRB competition format)
            # Use training data split instead
            logger.info("Test CSV doesn't have labels. Loading training data for validation split...")
            return load_training_data_for_testing()
    
    # Fallback: try loading from class folders (if CSV doesn't exist)
    for class_id in range(NUM_CLASSES):
        class_dir = test_path / f"{class_id:05d}"
        if not class_dir.exists():
            continue
        
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.ppm', '.jpg', '.jpeg', '.png']:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(class_id)
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
    
    return np.array(images), np.array(labels)


def load_training_data_for_testing(test_size: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a portion of training data for testing/validation
    This is used when test labels are not available
    """
    import pandas as pd
    
    train_dir = DATA_DIR / "Final_Training" / "Images"
    images = []
    labels = []
    
    # Load from each class folder
    for class_id in range(NUM_CLASSES):
        class_dir = train_dir / f"{class_id:05d}"
        csv_file = class_dir / f"GT-{class_id:05d}.csv"
        
        if not class_dir.exists():
            continue
            
        # Try to load from CSV if available
        if csv_file.exists():
            df = pd.read_csv(csv_file, sep=';')
            # Take a portion from each class
            samples_per_class = max(1, test_size // NUM_CLASSES)
            for _, row in df.head(samples_per_class).iterrows():
                img_path = class_dir / row['Filename']
                if img_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, IMG_SIZE)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = img.astype(np.float32) / 255.0
                            images.append(img)
                            labels.append(class_id)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
        else:
            # Fallback: load directly from folder
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.ppm', '.jpg', '.jpeg', '.png'] and not img_file.name.startswith('GT-'):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.resize(img, IMG_SIZE)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = img.astype(np.float32) / 255.0
                            images.append(img)
                            labels.append(class_id)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {e}")
    
    logger.info(f"Loaded {len(images)} images from training data for testing")
    return np.array(images), np.array(labels)


def load_test_data_from_csv(csv_path: Path, img_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data from GTSRB CSV format"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = img_root / row['Filename']
        if img_path.exists():
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(row['ClassId'])
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
    
    return np.array(images), np.array(labels)


def evaluate_model(model_path: str = MODEL_PATH,
                   test_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
    """
    Evaluate model on test data
    
    Args:
        model_path: Path to trained model
        test_data: Optional tuple of (X_test, y_test). If None, loads from disk.
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Loading model...")
    model = load_model(model_path)
    
    # Load test data if not provided
    if test_data is None:
        logger.info("Loading test data...")
        X_test, y_test = load_test_data()
        if len(X_test) == 0:
            logger.warning("No test data found. Using dummy data for demonstration.")
            X_test = np.random.rand(100, 32, 32, 3).astype(np.float32)
            y_test = np.random.randint(0, NUM_CLASSES, 100)
    else:
        X_test, y_test = test_data
    
    logger.info(f"Evaluating on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted'))
    
    # Per-class metrics
    metrics['per_class'] = {}
    for class_id in range(NUM_CLASSES):
        class_mask = y_test == class_id
        if np.sum(class_mask) > 0:
            # Get predictions for this class only
            y_true_binary = (y_test == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)
            
            metrics['per_class'][class_id] = {
                'label': CLASS_LABELS.get(class_id, f"Class {class_id}"),
                'samples': int(np.sum(class_mask)),
                'accuracy': float(np.mean(y_pred[class_mask] == class_id)),
                'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=[CLASS_LABELS[i] for i in range(NUM_CLASSES)],
                                   output_dict=True,
                                   zero_division=0)
    metrics['classification_report'] = report
    
    # Find most confused pairs
    metrics['most_confused_pairs'] = find_most_confused_pairs(cm)
    
    # Find worst performing classes
    metrics['worst_classes'] = find_worst_classes(metrics['per_class'])
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return metrics


def find_most_confused_pairs(cm: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """Find the most commonly confused class pairs"""
    confused_pairs = []
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true_class': i,
                    'predicted_class': j,
                    'label_true': CLASS_LABELS.get(i, f"Class {i}"),
                    'label_predicted': CLASS_LABELS.get(j, f"Class {j}"),
                    'count': int(cm[i, j])
                })
    
    # Sort by count and return top N
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    return confused_pairs[:top_n]


def find_worst_classes(per_class_metrics: Dict, top_n: int = 5) -> List[Dict[str, Any]]:
    """Find classes with lowest performance"""
    class_list = []
    
    for class_id, metrics in per_class_metrics.items():
        class_list.append({
            'class_id': class_id,
            'label': metrics['label'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'samples': metrics['samples']
        })
    
    # Sort by F1 score
    class_list.sort(key=lambda x: x['f1'])
    return class_list[:top_n]


def save_results(metrics: Dict[str, Any], output_dir: Path = RESULTS_DIR):
    """Save evaluation results to files"""
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = json.dumps(metrics, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        f.write(json_metrics)
    logger.info(f"Results saved to {json_path}")
    
    # Save confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=[CLASS_LABELS[i][:10] for i in range(NUM_CLASSES)],
                yticklabels=[CLASS_LABELS[i][:10] for i in range(NUM_CLASSES)])
    plt.title('Confusion Matrix - Traffic Sign Recognition')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    # Save per-class accuracy plot
    accuracies = [metrics['per_class'][i]['accuracy'] for i in range(NUM_CLASSES)]
    labels = [CLASS_LABELS[i][:15] for i in range(NUM_CLASSES)]
    
    plt.figure(figsize=(20, 8))
    plt.bar(range(NUM_CLASSES), accuracies)
    plt.xticks(range(NUM_CLASSES), labels, rotation=90)
    plt.xlabel('Traffic Sign Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy - Traffic Sign Recognition')
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_accuracy.png", dpi=150)
    plt.close()
    logger.info(f"Per-class accuracy plot saved to {output_dir / 'per_class_accuracy.png'}")


def analyze_errors(metrics: Dict[str, Any]) -> str:
    """Generate error analysis report"""
    report = []
    report.append("=" * 60)
    report.append("TRAFFIC SIGN RECOGNITION - ERROR ANALYSIS REPORT")
    report.append("=" * 60)
    
    # Overall metrics
    report.append("\n## Overall Performance")
    report.append(f"- Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"- Precision: {metrics['precision']:.4f}")
    report.append(f"- Recall:    {metrics['recall']:.4f}")
    report.append(f"- F1-Score:  {metrics['f1_score']:.4f}")
    
    # Most confused pairs
    report.append("\n## Most Confused Class Pairs")
    report.append("(True class → Predicted class: Number of misclassifications)")
    for pair in metrics['most_confused_pairs'][:5]:
        report.append(f"  {pair['label_true']} → {pair['label_predicted']}: {pair['count']} times")
    
    # Worst performing classes
    report.append("\n## Worst Performing Classes")
    report.append("(Classes with lowest F1-score)")
    for cls in metrics['worst_classes'][:5]:
        report.append(f"  - {cls['label']}: F1={cls['f1']:.4f}, Accuracy={cls['accuracy']:.4f}")
    
    # Common error patterns
    report.append("\n## Common Error Patterns")
    report.append("Based on the confusion analysis:")
    
    # Analyze patterns
    confused_pairs = metrics['most_confused_pairs']
    if confused_pairs:
        report.append("  - The model most commonly confuses:")
        for pair in confused_pairs[:3]:
            report.append(f"    * {pair['label_true']} misclassified as {pair['label_predicted']}")
    
    report.append("\n## Recommendations")
    report.append("  1. Consider collecting more training data for poorly performing classes")
    report.append("  2. Apply more aggressive data augmentation for confused classes")
    report.append("  3. Consider class-specific fine-tuning or ensemble methods")
    report.append("  4. For similar-looking signs, consider adding distinguishing features")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def main():
    """Main evaluation pipeline"""
    logger.info("Starting Model Evaluation")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.info("Please train the model first using train_model.py")
        return
    
    # Load test data
    X_test, y_test = load_test_data()
    
    if len(X_test) == 0:
        logger.warning("No test data found. Using random test data for demonstration.")
        X_test = np.random.rand(200, 32, 32, 3).astype(np.float32)
        y_test = np.random.randint(0, NUM_CLASSES, 200)
    
    # Evaluate model
    metrics = evaluate_model(MODEL_PATH, (X_test, y_test))
    
    # Save results
    save_results(metrics)
    
    # Generate and print error analysis
    report = analyze_errors(metrics)
    print(report)
    
    # Save report
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "error_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Error analysis report saved to {report_path}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()