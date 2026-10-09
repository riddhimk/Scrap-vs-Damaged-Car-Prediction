import numpy as np

def generate_tta_batch(img_array):
    """
    Generates a batch of Test-Time Augmented images from a single preprocessed image array.
    Args:
        img_array: Numpy array of shape (1, H, W, C), already normalized.
    Returns:
        Numpy array of shape (N, H, W, C) where N is the number of augmentations + original.
    """
    # img_array is (1, H, W, C)
    batch = [img_array[0]]
    
    # 1. Horizontal Flip
    batch.append(np.fliplr(img_array[0]))
    
    # 2. Brightness adjusted (+10%)
    # Since it's normalized to 0-1, we clip at 1.0
    bright = np.clip(img_array[0] * 1.1, 0.0, 1.0)
    batch.append(bright)
    
    # 3. Brightness adjusted (-10%)
    dark = np.clip(img_array[0] * 0.9, 0.0, 1.0)
    batch.append(dark)
    
    # Convert list to numpy array of shape (4, H, W, C)
    return np.array(batch)
