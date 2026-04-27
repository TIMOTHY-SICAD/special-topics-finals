"""
Philippine Road Signs Dataset Utilities

Utilities for loading and managing Philippine road signs from Wikimedia Commons.
Provides functions for dataset splitting and format handling.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

from .preprocessing import load_and_preprocess_image, batch_preprocess_images

logger = logging.getLogger(__name__)

# Philippine Signs dataset path
PHILIPPINE_SIGNS_DIR = Path(__file__).parent.parent / "data" / "Philippine_Signs"


def get_philippine_signs_path() -> Path:
    """Get the path to Philippine signs directory."""
    return PHILIPPINE_SIGNS_DIR


def get_supported_image_extensions() -> set:
    """Get set of supported image extensions."""
    return {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.ppm'}


def get_image_files(directory: Path = None) -> List[Path]:
    """
    Get all supported image files from directory.
    
    Args:
        directory: Directory to search (defaults to Philippine_Signs)
    
    Returns:
        List of image file paths
    """
    if directory is None:
        directory = PHILIPPINE_SIGNS_DIR
    
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    supported_exts = get_supported_image_extensions()
    image_files = []
    
    for ext in supported_exts:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    return image_files


def load_philippine_signs_dataset(
    directory: Path = None,
    for_training: bool = False,
    limit: Optional[int] = None
) -> Tuple[np.ndarray, List[Path]]:
    """
    Load Philippine road signs as image arrays.
    
    Args:
        directory: Directory containing images (defaults to Philippine_Signs)
        for_training: Whether to apply data augmentation
        limit: Maximum number of images to load (None = all)
    
    Returns:
        Tuple of (images_array, file_paths)
    """
    if directory is None:
        directory = PHILIPPINE_SIGNS_DIR
    
    directory = Path(directory)
    
    logger.info(f"Loading Philippine signs from: {directory}")
    
    image_files = get_image_files(directory)
    
    if not image_files:
        logger.warning(f"No image files found in {directory}")
        return np.array([]), []
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} images, loading...")
    
    images = batch_preprocess_images([str(f) for f in image_files], for_training=for_training)
    
    if len(images) == 0:
        logger.warning("Failed to load any images")
        return np.array([]), []
    
    logger.info(f"Successfully loaded {len(images)} images")
    
    return images, image_files[:len(images)]


def split_dataset(
    images: np.ndarray,
    file_paths: List[Path],
    train_split: float = 0.7,
    random_seed: int = 42
) -> Tuple[Tuple[np.ndarray, List[Path]], Tuple[np.ndarray, List[Path]]]:
    """
    Split dataset into training and testing sets.
    
    Args:
        images: Image array
        file_paths: Corresponding file paths
        train_split: Fraction for training (0.7 = 70% train, 30% test)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of ((train_images, train_paths), (test_images, test_paths))
    """
    np.random.seed(random_seed)
    
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_split)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_images = images[train_indices]
    train_paths = [file_paths[i] for i in train_indices]
    
    test_images = images[test_indices]
    test_paths = [file_paths[i] for i in test_indices]
    
    logger.info(f"Dataset split: {len(train_images)} train, {len(test_images)} test")
    
    return (train_images, train_paths), (test_images, test_paths)


def get_dataset_statistics(images: np.ndarray, file_paths: List[Path]) -> dict:
    """
    Get statistics about the loaded dataset.
    
    Args:
        images: Image array
        file_paths: Corresponding file paths
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_images": len(images),
        "shape": images[0].shape if len(images) > 0 else None,
        "dtype": images.dtype,
        "min_value": float(np.min(images)) if len(images) > 0 else None,
        "max_value": float(np.max(images)) if len(images) > 0 else None,
        "mean_value": float(np.mean(images)) if len(images) > 0 else None,
        "std_value": float(np.std(images)) if len(images) > 0 else None,
    }
    
    # File format distribution
    format_dist = {}
    for path in file_paths:
        ext = path.suffix.lower()
        format_dist[ext] = format_dist.get(ext, 0) + 1
    
    stats["format_distribution"] = format_dist
    
    return stats


def main():
    """Example usage."""
    # Load Philippine signs
    images, file_paths = load_philippine_signs_dataset(for_training=False, limit=100)
    
    if len(images) > 0:
        # Get statistics
        stats = get_dataset_statistics(images, file_paths)
        
        print("\nDataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Image shape: {stats['shape']}")
        print(f"  Value range: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]")
        print(f"  Mean: {stats['mean_value']:.3f}, Std: {stats['std_value']:.3f}")
        print(f"  Format distribution: {stats['format_distribution']}")
        
        # Split dataset
        (train_imgs, train_paths), (test_imgs, test_paths) = split_dataset(images, file_paths)
        
        print(f"\n  Train set: {len(train_imgs)} images")
        print(f"  Test set: {len(test_imgs)} images")
    else:
        print("No images found. Run download_philippine_signs.py first.")


if __name__ == "__main__":
    main()
