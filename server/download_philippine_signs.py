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
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data" / "Philippine_Signs"
WIKIMEDIA_CATEGORY_URL = "https://commons.wikimedia.org/wiki/Category:Road_signs_in_the_Philippines"

# Supported image formats (excluding SVG for now)
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# Delay settings (in seconds) to avoid rate limiting
REQUEST_DELAY = 2  # Delay between requests (Wikimedia recommends ~1-2s)
RETRY_DELAY = 10  # Initial delay for retries
MAX_RETRIES = 3  # Maximum retry attempts

# gallery-dl configuration with rate limiting
GALLERY_DL_CONFIG = {
    "extractor": {
        "wikimedia": {
            "filename": "{title}.{extension}",
            "directory": [str(OUTPUT_DIR)],
            "skip": "exists",  # Skip already downloaded files
            "delay": REQUEST_DELAY,  # Add delay between requests
            "retries": MAX_RETRIES,  # Retry failed downloads
            "timeout": 30,  # 30 second timeout per request
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
    
    Includes automatic retry logic with exponential backoff for rate limiting.
    If rate limiting persists, consider using thumbnail images or downloading
    in smaller batches.
    
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
    
    # Prepare gallery-dl command with rate-limiting settings
    cmd = [
        sys.executable, "-m", "gallery_dl",
        "--directory", str(OUTPUT_DIR),
        "--filename", "{title}.{extension}",
        "--filter", "extension in ('jpg', 'jpeg', 'png', 'gif', 'webp')",
        "--sleep", str(REQUEST_DELAY),  # Add delay between requests
        WIKIMEDIA_CATEGORY_URL
    ]
    
    logger.info(f"Downloading from: {WIKIMEDIA_CATEGORY_URL}")
    logger.info(f"Request delay: {REQUEST_DELAY}s (to respect Wikimedia rate limits)")
    logger.info(f"Output directory: {OUTPUT_DIR}\n")
    logger.info(f"Running: gallery-dl with rate limiting\n")
    
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            result = subprocess.run(cmd, capture_output=False, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("\n✓ Download completed successfully!")
                return True
            elif result.returncode == 1 and retry_count < MAX_RETRIES:
                # 429 or other temporary error - retry with exponential backoff
                retry_count += 1
                wait_time = RETRY_DELAY * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"\nRate limited or temporary error. Retry {retry_count}/{MAX_RETRIES}")
                logger.info(f"Waiting {wait_time}s before retry...\n")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed with return code: {result.returncode}")
                if result.returncode == 429:
                    logger.error("\nRate limited by Wikimedia Commons.")
                    logger.error("Recommendation: Wait longer before retrying, or use a different approach:")
                    logger.error("  1. Download in smaller batches")
                    logger.error("  2. Use thumbnail images instead of full-size")
                    logger.error("  3. Space out downloads over multiple sessions")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Download timed out (exceeded 1 hour)")
            return False
        except Exception as e:
            logger.error(f"Error during download: {e}")
            return False
    
    return False


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
        if img_file.is_file() and img_file.name != '.gitkeep':
            stats["total_images"] += 1
            ext = img_file.suffix.lower()
            stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1
            stats["total_size_mb"] += img_file.stat().st_size / (1024 * 1024)
    
    return stats


def download_with_api(use_thumbnails=True, thumb_width=400) -> bool:
    """
    Download using Wikimedia Commons API directly.
    
    This provides better control over rate limiting and supports thumbnails
    which have less restrictive rate limits than full-size images.
    
    Args:
        use_thumbnails: If True, download thumbnail images (400px width)
                       If False, download full-size images
        thumb_width: Width in pixels for thumbnail images (default: 400)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available for API downloads")
        return False
    
    logger.info("\n" + "=" * 60)
    if use_thumbnails:
        logger.info(f"Downloading {thumb_width}px THUMBNAILS via API (low rate limit)")
    else:
        logger.info("Downloading FULL-SIZE images via API (with rate limiting)")
    logger.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Wikimedia Commons API endpoint
    api_url = "https://commons.wikimedia.org/w/api.php"
    
    # Headers for API requests (Wikimedia requires a User-Agent)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Query parameters for category pages
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Road signs in the Philippines",
        "cmtype": "file",
        "cmlimit": "500",  # Max results per request
        "format": "json"
    }
    
    downloaded = 0
    failed = 0
    skipped = 0
    continued = True
    cmcontinue = None
    
    try:
        while continued:
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            
            logger.info(f"\nFetching batch (offset: {cmcontinue or 'start'})...")
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            members = data.get("query", {}).get("categorymembers", [])
            
            if not members:
                break
            
            logger.info(f"Processing {len(members)} files...")
            
            # Download each file
            for member in members:
                try:
                    # Check namespace: 6 = File, 14 = Category
                    # Skip categories, only process files
                    if member.get("ns") != 6:
                        skipped += 1
                        continue
                    
                    file_name = member.get("title", "")
                    if not file_name:
                        skipped += 1
                        continue
                    
                    # Skip non-image files
                    ext = Path(file_name).suffix.lower()
                    if ext not in SUPPORTED_FORMATS and ext != '.svg':
                        skipped += 1
                        continue
                    if ext == '.svg':  # Skip SVG for now (no rasterization)
                        skipped += 1
                        continue
                    
                    # Get file info
                    file_params = {
                        "action": "query",
                        "titles": file_name,
                        "prop": "imageinfo",
                        "iiprop": "url" + ("|thumburl|thumbwidth" if use_thumbnails else ""),
                        "iiurlwidth": str(thumb_width) if use_thumbnails else "",
                        "format": "json"
                    }
                    
                    file_response = requests.get(api_url, params=file_params, headers=headers, timeout=30)
                    file_data = file_response.json()
                    
                    pages = file_data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        imageinfo = page.get("imageinfo", [])
                        if imageinfo and len(imageinfo) > 0:
                            img_info = imageinfo[0]
                            
                            # Determine which URL to use
                            if use_thumbnails and "thumburl" in img_info:
                                img_url = img_info.get("thumburl")
                                source = "thumbnail"
                            else:
                                img_url = img_info.get("url")
                                source = "full-size"
                            
                            if img_url:
                                # Download the image
                                output_file = OUTPUT_DIR / file_name
                                
                                if not output_file.exists():
                                    try:
                                        img_response = requests.get(img_url, headers=headers, timeout=30)
                                        img_response.raise_for_status()
                                        with open(output_file, 'wb') as f:
                                            f.write(img_response.content)
                                        downloaded += 1
                                        logger.info(f"  ✓ {file_name} ({source})")
                                    except Exception as e:
                                        failed += 1
                                        logger.warning(f"  ✗ {file_name}: {str(e)[:50]}")
                                else:
                                    logger.info(f"  - {file_name} (exists)")
                    
                    # Rate limiting delay (lighter for thumbnails)
                    delay = REQUEST_DELAY / 2 if use_thumbnails else REQUEST_DELAY
                    time.sleep(delay)
                
                except Exception as member_error:
                    failed += 1
                    logger.warning(f"  ✗ Error: {str(member_error)[:50]}")
            
            # Check for continuation
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            continued = cmcontinue is not None
    
    except Exception as e:
        logger.error(f"API download error: {e}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Results ({('Thumbnails' if use_thumbnails else 'Full-size')})")
    logger.info(f"{'='*60}")
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Total: {downloaded + failed + skipped}")
    
    return downloaded > 0


def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("Philippine Road Signs Downloader (Wikimedia Commons)")
    logger.info("=" * 60)
    
    # Wikimedia is aggressively rate-limiting full-size downloads
    # Skip gallery-dl and use API with thumbnails directly
    logger.info("\nUsing Wikimedia API with THUMBNAILS (400px)")
    logger.info("This approach has much better rate limiting...")
    logger.info("")
    
    success = download_with_api(use_thumbnails=True, thumb_width=400)
    
    if success:
        stats = get_image_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics")
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
        
        if stats['total_images'] > 0:
            logger.info("\n✓ Ready for preprocessing with philippine_signs.py utilities")
            logger.info("✓ Note: Downloaded as thumbnails (400px) - suitable for testing")
        else:
            logger.warning("\n⚠ No images downloaded yet. Try again later.")
    else:
        logger.error("\n✗ Failed to download images")
        logger.error("\nAlternative approaches:")
        logger.error("1. Wait longer between attempts (Wikimedia may have temporary rate limits)")
        logger.error("2. Download manually from: https://commons.wikimedia.org/wiki/Category:Road_signs_in_the_Philippines")
        logger.error("3. Use a VPN if your IP is temporarily banned")
        sys.exit(1)


if __name__ == "__main__":
    main()
