"""
Image loading and utility functions for QR code processing.
"""

from typing import Dict, Tuple, Union, List
import cv2
import numpy as np
from PIL import Image

# Intensity modulation mapping constants
IM_INTENSITY_MAP: Dict[Tuple[int, int], int] = {
    (0, 0): 10,
    (0, 1): 55,
    (1, 0): 137,
    (1, 1): 255,
}
# Auto-generate the reverse map to ensure consistency
IM_INTENSITY_REVERSE_MAP = {v: k for k, v in IM_INTENSITY_MAP.items()}

def load_image_for_decode(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Helper to load an image for decoding. Accepts file path, PIL Image, or numpy array.
    Returns a numpy array (BGR for OpenCV/pyzbar).
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {image}")
        return img
    elif isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError("Input must be a file path, PIL.Image.Image, or numpy.ndarray.")

def nearest_intensity(val: int) -> int:
    """
    Find the nearest valid intensity value for IM decoding.
    """
    return min(IM_INTENSITY_REVERSE_MAP.keys(), key=lambda k: abs(k - val))

def split_and_pad_data(data: str, n_parts: int, pad_char: str) -> List[str]:
    """
    Split data into n_parts, padding each part to the same length with pad_char.
    """
    n = len(data)
    k, m = divmod(n, n_parts)
    parts = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]
    maxlen = max(len(p) for p in parts)
    return [p.ljust(maxlen, pad_char) for p in parts]
