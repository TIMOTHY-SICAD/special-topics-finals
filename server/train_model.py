"""
CNN Model Training Script for Traffic Sign Recognition
Trains a convolutional neural network on the GTSRB dataset
"""

import os
import numpy as np
import cv2
import pickle
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force unbuffered output for real-time logging
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, write_through=True)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, AveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (32, 32)
NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Paths - relative to this script's location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_DIR = DATA_DIR / "Final_Training" / "Images"
TEST_DIR = DATA_DIR / "Final_Test" / "Images"
MODEL_DIR = SCRIPT_DIR / "models"
LOG_DIR = SCRIPT_DIR / "logs"


def load_data_from_folder(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from folder structure (GTSRB format)
    
    Expected structure:
    data/Final_Training/Images/00000/*.ppm
    data/Final_Training/Images/00001/*.ppm
    ...
    """
    images = []
    labels = []
    
    if not data_path.exists():
        logger.warning(f"Data directory {data_path} does not exist")
        return np.array([]), np.array([])
    
    total_loaded = 0
    print(f"Starting data load from {data_path}...", flush=True)
    for class_id in range(NUM_CLASSES):
        # GTSRB uses 5-digit zero-padded folder names (00000-00042)
        class_dir = data_path / f"{class_id:05d}"
        if not class_dir.exists():
            logger.warning(f"Class directory {class_dir} does not exist")
            continue
        
        class_count = 0
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
                        class_count += 1
                        total_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
        
        if class_id % 10 == 0:
            print(f"Loaded class {class_id}: {class_count} images (total: {total_loaded})", flush=True)
    
    print(f"Data loading complete: {total_loaded} images", flush=True)
    return np.array(images), np.array(labels)


def load_data_from_csv(csv_path: Path, img_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from GTSRB CSV file
    
    Args:
        csv_path: Path to CSV file with columns: Filename, ClassId
        img_root: Root directory containing images
    """
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


def build_cnn_model(input_shape: Tuple[int, int, int] = (32, 32, 3), 
                    num_classes: int = NUM_CLASSES) -> Sequential:
    """
    Build CNN model for traffic sign recognition
    
    Architecture:
    - 3 convolutional blocks with batch normalization
    - Dropout for regularization
    - Dense layers for classification
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', activation='relu', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_lightweight_model(input_shape: Tuple[int, int, int] = (32, 32, 3),
                            num_classes: int = NUM_CLASSES) -> Sequential:
    """
    Build a lighter CNN model for edge deployment
    
    Fewer parameters for lower computational requirements
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Second Convolutional Block
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def get_data_augmentation() -> ImageDataGenerator:
    """
    Create data augmentation generator for training
    
    Applies:
    - Random rotation (up to 15 degrees)
    - Random zoom (up to 10%)
    - Random width/height shift (up to 10%)
    - Random horizontal flip
    """
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,  # Traffic signs should not be flipped
        fill_mode='nearest'
    )


def train_model(model: Sequential, 
                train_data: Tuple[np.ndarray, np.ndarray],
                val_data: Tuple[np.ndarray, np.ndarray],
                model_path: str = None,
                use_augmentation: bool = True) -> Sequential:
    """
    Train the CNN model
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        model_path: Path to save the trained model
        use_augmentation: Whether to use data augmentation
    
    Returns:
        Trained model
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Use default model path if not provided
    if model_path is None:
        model_path = str(MODEL_DIR / "traffic_sign_model.h5")
    
    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOG_DIR),
            histogram_freq=1
        )
    ]
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Training
    if use_augmentation:
        datagen = get_data_augmentation()
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    
    # Save final model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model


def download_gtsrb_dataset() -> bool:
    """
    Download GTSRB dataset
    
    Returns:
        True if successful
    """
    import urllib.request
    import zipfile
    
    # GTSRB dataset URLs
    train_url = "https://benchmark.ini.rub.de/gtsrb/GTSRB-Training_fixed.zip"
    test_url = "https://benchmark.ini.rub.de/gtsrb/GT-ID_test.zip"
    
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download training data
    train_zip = DATA_DIR / "GTSRB-Training_fixed.zip"
    if not train_zip.exists():
        logger.info("Downloading GTSRB training data...")
        try:
            urllib.request.urlretrieve(train_url, train_zip)
            logger.info("Extracting training data...")
            with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            logger.info("Training data extracted")
        except Exception as e:
            logger.error(f"Failed to download training data: {e}")
            return False
    
    return True


def split_data(X: np.ndarray, y: np.ndarray, 
               val_split: float = 0.2) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                 Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into training and validation sets
    
    Args:
        X: Images
        y: Labels
        val_split: Fraction for validation
    
    Returns:
        ((X_train, y_train), (X_val, y_val))
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_val, y_val)


def main():
    """Main training pipeline"""
    logger.info("Starting Traffic Sign Recognition Model Training")
    
    # Check for data
    if TRAIN_DIR.exists():
        logger.info(f"Loading training data from {TRAIN_DIR}")
        X, y = load_data_from_folder(TRAIN_DIR)
        
        if len(X) > 0:
            logger.info(f"Loaded {len(X)} images")
            
            # Split data
            (X_train, y_train), (X_val, y_val) = split_data(X, y)
            logger.info(f"Training: {len(X_train)}, Validation: {len(X_val)}")
            
            # Build model
            logger.info("Building CNN model...")
            model = build_cnn_model()
            
            # Train
            logger.info("Training model...")
            trained_model = train_model(
                model, 
                (X_train, y_train), 
                (X_val, y_val),
                use_augmentation=True
            )
            
            logger.info("Training complete!")
            return
    
    logger.warning("No training data found. Please download GTSRB dataset.")
    logger.info("Download from: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip")
    
    # Create sample data for testing if no real data
    logger.info("Creating sample model for demonstration...")
    model = build_cnn_model()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy data
    X_dummy = np.random.rand(100, 32, 32, 3).astype(np.float32)
    y_dummy = np.random.randint(0, NUM_CLASSES, 100)
    
    # Quick training
    model.fit(X_dummy, y_dummy, epochs=2, verbose=1)
    
    # Save model to server/models/
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(str(MODEL_DIR / "traffic_sign_model.h5"))
    logger.info(f"Sample model saved to {MODEL_DIR / 'traffic_sign_model.h5'}")


if __name__ == "__main__":
    main()