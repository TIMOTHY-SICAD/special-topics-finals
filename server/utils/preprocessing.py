"""
Image Preprocessing Utilities for Traffic Sign Recognition
Handles all preprocessing steps for the CNN model
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Standard image size for the model
IMG_SIZE = (32, 32)


def resize_image(image: np.ndarray, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Resize image to standard input size
    
    Args:
        image: Input image as numpy array
        size: Target size (width, height)
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range
    
    Args:
        image: Input image with pixel values [0, 255]
    
    Returns:
        Normalized image with values [0, 1]
    """
    return image.astype(np.float32) / 255.0


def convert_color_space(image: np.ndarray, conversion: int = cv2.COLOR_BGR2RGB) -> np.ndarray:
    """
    Convert image color space
    
    Args:
        image: Input image
        conversion: OpenCV color conversion code
    
    Returns:
        Converted image
    """
    return cv2.cvtColor(image, conversion)


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def adjust_brightness(image: np.ndarray, alpha: float = 1.0, beta: float = 0) -> np.ndarray:
    """
    Adjust image brightness and contrast
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
    
    Returns:
        Adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def rotate_image(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Rotate image by specified angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Center of rotation (default: image center)
    
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Flip image horizontally or vertically
    
    Args:
        image: Input image
        horizontal: If True, flip horizontally; otherwise vertically
    
    Returns:
        Flipped image
    """
    if horizontal:
        return cv2.flip(image, 1)
    return cv2.flip(image, 0)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input image (grayscale or color)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for color images
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(image)


def add_noise(image: np.ndarray, noise_type: str = "gaussian") -> np.ndarray:
    """
    Add noise to image for data augmentation
    
    Args:
        image: Input image
        noise_type: Type of noise ("gaussian" or "salt_pepper")
    
    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == "salt_pepper":
        noisy = image.copy()
        prob = 0.01
        # Salt
        salt_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(image.shape[:2]) < prob / 2
        noisy[pepper_mask] = 0
        return noisy
    
    return image


def preprocess_for_training(image: np.ndarray, augment: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for training
    
    Args:
        image: Input image
        augment: Whether to apply data augmentation
    
    Returns:
        Preprocessed image ready for model input
    """
    # Resize to standard size
    img = resize_image(image)
    
    # Apply CLAHE for better contrast
    img = apply_clahe(img)
    
    # Normalize
    img = normalize_image(img)
    
    if augment:
        # Random rotation (-15 to +15 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = rotate_image(img, angle)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-20, 20)
            img = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=alpha, beta=beta)
            img = img.astype(np.float32) / 255.0
    
    return img


def preprocess_for_inference(image: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline for inference (no augmentation)
    
    Args:
        image: Input image
    
    Returns:
        Preprocessed image ready for model input
    """
    # Resize to standard size
    img = resize_image(image)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = normalize_image(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def load_and_preprocess_image(image_path: str, for_training: bool = False) -> np.ndarray:
    """
    Load image from file and preprocess it
    
    Args:
        image_path: Path to image file
        for_training: Whether this is for training (applies augmentation)
    
    Returns:
        Preprocessed image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Preprocess
    if for_training:
        return preprocess_for_training(image, augment=True)
    return preprocess_for_inference(image)


def batch_preprocess_images(image_paths: list, for_training: bool = False) -> np.ndarray:
    """
    Preprocess multiple images
    
    Args:
        image_paths: List of image file paths
        for_training: Whether this is for training
    
    Returns:
        Numpy array of preprocessed images
    """
    images = []
    for path in image_paths:
        try:
            img = load_and_preprocess_image(path, for_training)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to preprocess {path}: {e}")
    
    return np.array(images)


# Class labels for GTSRB dataset
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


def get_class_label(class_id: int) -> str:
    """Get human-readable label for traffic sign class"""
    return CLASS_LABELS.get(class_id, f"Unknown Class {class_id}")