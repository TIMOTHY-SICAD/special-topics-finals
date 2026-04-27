"""
Download Philippine road signs from Wikimedia Commons using gallery-dl

This script downloads images from the Wikimedia Commons category:
https://commons.wikimedia.org/wiki/Category:Road_signs_in_the_Philippines

The script filters for common image formats (JPEG, PNG, GIF, WebP) and
skips SVG files (which would need rasterization).

Usage:
    python download_philippine_signs.py

Output:
    Downloads images to: server/data/Philippine_Signs/
"""

import os
import json
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data" / "Philippine_Signs"
WIKIMEDIA_CATEGORY_URL = "https://commons.wikimedia.org/wiki/Category:Road_signs_in_the_Philippines"

# Supported image formats (excluding SVG for now)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# gallery-dl configuration
GALLERY_DL_CONFIG = {
    "extractor": {
        "wikimedia": {
            "filename": "{title}.{extension}",
            "directory": [str(OUTPUT_DIR)],
            "skip": "exists",  # Skip already downloaded files
        }
    },
    "output": {
        "logformat": "[{levelname}] {message}",
        "loglevel": "info",
    }
}


def check_gallery_dl_installed() -> bool:
    """Check if gallery-dl is installed."""
    try:
        import gallery_dl
        logger.info(f"gallery-dl version: {gallery_dl.__version__}")
        return True
    except ImportError:
        logger.error("gallery-dl is not installed. Install it with: pip install gallery-dl")
        return False


def create_gallery_dl_config() -> Path:
    """Create temporary gallery-dl configuration file."""
    config_path = Path.home() / ".config" / "gallery-dl" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing config or create new one
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Merge with our settings
    config.update(GALLERY_DL_CONFIG)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")
    return config_path


def download_images() -> bool:
    """
    Download images from Wikimedia Commons using gallery-dl.
    
    Returns:
        True if successful, False otherwise
    """
    if not check_gallery_dl_installed():
        logger.info("\nInstalling gallery-dl...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gallery-dl"])
            logger.info("gallery-dl installed successfully")
        except subprocess.CalledProcessError:
            logger.error("Failed to install gallery-dl")
            return False
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Prepare gallery-dl command
    # Using the Wikimedia Commons category URL directly
    cmd = [
        sys.executable, "-m", "gallery_dl",
        "--directory", str(OUTPUT_DIR),
        "--filename", "{title}.{extension}",
        "--filter", "extension in ('jpg', 'jpeg', 'png', 'gif', 'webp')",
        WIKIMEDIA_CATEGORY_URL
    ]
    
    logger.info(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            logger.info("\nDownload completed successfully!")
            return True
        else:
            logger.error(f"Download failed with return code: {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"Error during download: {e}")
        return False


def count_downloaded_images() -> int:
    """Count images in output directory."""
    if not OUTPUT_DIR.exists():
        return 0
    
    image_count = len(list(OUTPUT_DIR.glob("*")))
    return image_count


def get_image_statistics() -> dict:
    """Get statistics about downloaded images."""
    stats = {
        "total_images": 0,
        "by_format": {},
        "total_size_mb": 0
    }
    
    if not OUTPUT_DIR.exists():
        return stats
    
    for img_file in OUTPUT_DIR.iterdir():
        if img_file.is_file():
            stats["total_images"] += 1
            ext = img_file.suffix.lower()
            stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1
            stats["total_size_mb"] += img_file.stat().st_size / (1024 * 1024)
    
    return stats


def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("Philippine Road Signs Downloader (Wikimedia Commons)")
    logger.info("=" * 60)
    
    # Download images
    success = download_images()
    
    if success:
        stats = get_image_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Download Statistics")
        logger.info("=" * 60)
        logger.info(f"Total images downloaded: {stats['total_images']}")
        logger.info(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        if stats['by_format']:
            logger.info("\nImages by format:")
            for fmt, count in sorted(stats['by_format'].items()):
                logger.info(f"  {fmt}: {count}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Images saved to: {OUTPUT_DIR}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to download images")
        sys.exit(1)


if __name__ == "__main__":
    main()
