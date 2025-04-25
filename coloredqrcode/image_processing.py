"""
Pre and post-processing functions for QR code image processing.
"""

from typing import List, Sequence
import numpy as np
from PIL import Image, ImageFilter

from .utils import nearest_intensity
from .utils import IM_INTENSITY_MAP, IM_INTENSITY_REVERSE_MAP, nearest_intensity

def preprocess_colored_qr(img: Image.Image) -> List[Image.Image]:
    """
    Preprocess colored QR code image for decoding.
    Extracts and processes R, G, B channels.
    """
    channels = img.split()  # R, G, B
    qr_imgs = []
    for channel in channels:
        arr = np.array(channel)
        binary = (arr > 128).astype(np.uint8) * 255
        inverted = 255 - binary
        qr_img = Image.fromarray(inverted, "L").convert("RGB")
        qr_imgs.append(qr_img)
    return qr_imgs

def preprocess_colored_qr_im(img: Image.Image) -> List[Image.Image]:
    """
    Preprocess intensity-modulated colored QR code image for decoding.
    Extracts and processes R, G, B channels with intensity modulation.
    """
    channels = img.split()  # R, G, B
    qr_imgs = []
    for channel in channels:
        arr = np.array(channel, dtype=np.uint8)
        # Reconstruct the two QR code images for this channel
        qr_dim = np.full(arr.shape, 255, dtype=np.uint8)
        qr_bright = np.full(arr.shape, 255, dtype=np.uint8)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                val = arr[y, x]
                nearest = nearest_intensity(val)
                bits = IM_INTENSITY_REVERSE_MAP[nearest]
                qr_dim[y, x] = 0 if bits[0] else 255
                qr_bright[y, x] = 0 if bits[1] else 255
        qr_img_dim = Image.fromarray(qr_dim, "L").filter(ImageFilter.MedianFilter(size=3)).convert("RGB")
        qr_img_bright = Image.fromarray(qr_bright, "L").filter(ImageFilter.MedianFilter(size=3)).convert("RGB")
        qr_imgs.extend([qr_img_dim, qr_img_bright])
    return qr_imgs

def postprocess_colored_qr(qr_imgs: Sequence[Image.Image]) -> Image.Image:
    """
    Post-process QR code images for colored QR code generation.
    Combines R, G, B channels into a single colored QR code.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    color_layers = []
    for img, color in zip(qr_imgs, colors):
        gray = img.convert("L")
        inv = Image.eval(gray, lambda x: 255 - x)
        arr = np.array(inv)
        color_img = np.zeros((*arr.shape, 3), dtype=np.uint8)
        mask = arr == 255
        for c in range(3):
            color_img[mask, c] = color[c]
        color_layers.append(Image.fromarray(color_img, "RGB"))
    merged = np.maximum.reduce([np.array(layer) for layer in color_layers])
    return Image.fromarray(merged, "RGB")

def postprocess_colored_qr_im(qr_imgs: Sequence[Image.Image]) -> Image.Image:
    """
    Post-process QR code images for intensity-modulated colored QR code generation.
    Combines intensity-modulated R, G, B channels into a single colored QR code.
    """
    size = qr_imgs[0].size
    result_arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i, color in enumerate([(0,), (1,), (2,)]):  # R, G, B
        img_dim = qr_imgs[2 * i].convert("L")
        img_bright = qr_imgs[2 * i + 1].convert("L")
        arr_dim = np.array(img_dim)
        arr_bright = np.array(img_bright)
        channel = np.zeros_like(arr_dim, dtype=np.uint8)
        for y in range(arr_dim.shape[0]):
            for x in range(arr_dim.shape[1]):
                bit_dim = 1 if arr_dim[y, x] < 128 else 0
                bit_bright = 1 if arr_bright[y, x] < 128 else 0
                val = IM_INTENSITY_MAP[(bit_dim, bit_bright)]
                channel[y, x] = val
        result_arr[..., color[0]] = channel
    return Image.fromarray(result_arr, "RGB")
