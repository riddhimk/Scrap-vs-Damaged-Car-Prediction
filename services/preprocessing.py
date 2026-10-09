import os
from PIL import Image, UnidentifiedImageError

def validate_and_load_image(filepath, target_size=None):
    """
    Validates and verifies that an image file can be opened.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"Invalid or corrupted image file: {e}")

def validate_image_file(file, max_size_bytes=10 * 1024 * 1024):
    """
    Basic checks before saving the file.
    """
    if file.filename == '':
        return False, "Empty filename"
    
    # Check extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return False, "Unsupported file format. Please upload PNG, JPG, JPEG, or WEBP."
        
    # Check file size by seeking to the end
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > max_size_bytes:
        return False, f"File exceeds maximum allowed size of {max_size_bytes / (1024*1024):.1f}MB"
        
    if file_size == 0:
        return False, "File is empty"
        
    return True, None

