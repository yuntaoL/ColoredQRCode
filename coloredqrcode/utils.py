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


def load_image_for_decode(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Helper to load an image for decoding. Accepts file path, PIL Image, or numpy array.
    Returns a PIL Image in RGB mode, ready for QR code decoding.

    Args:
        image: File path, PIL Image object, or numpy array.
            For numpy arrays:
            - If from cv2.imread(): BGR format (need conversion)
            - If from PIL.Image.array(): RGB format (no conversion needed)
    Returns:
        PIL Image in RGB mode.
    Raises:
        FileNotFoundError: If the image file cannot be found.
        TypeError: If the input type is not supported.
    """
    if isinstance(image, str):
        # Use PIL to load image directly in RGB mode
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image in RGB mode
        img = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError("Input must be a file path, PIL.Image.Image, or numpy.ndarray.")

    return img


def nearest_intensity(val: int) -> int:
    """
    Find the nearest valid intensity value for IM decoding.
    """
    # Find the intensity key with the smallest absolute difference from val
    return min(IM_INTENSITY_REVERSE_MAP.keys(), key=lambda k: abs(int(k) - int(val)))


def split_and_pad_data(data: str, n_parts: int, pad_char: str) -> List[str]:
    """
    Split data into n_parts, padding each part to the same length with pad_char.
    """
    n = len(data)
    k, m = divmod(n, n_parts)
    parts = [
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_parts)
    ]
    maxlen = max(len(p) for p in parts)
    return [p.ljust(maxlen, pad_char) for p in parts]
