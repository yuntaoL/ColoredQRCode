from qrcode import QRCode
from PIL import Image, ImageFilter
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
from typing import Optional, Literal, Tuple, List, Union, Callable, Sequence, cast
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)

IM_INTENSITY_MAP = {
    (0, 0): 10,
    (0, 1): 55,
    (1, 0): 137,
    (1, 1): 255,
}
# Auto-generate the reverse map to ensure consistency
IM_INTENSITY_REVERSE_MAP = {v: k for k, v in IM_INTENSITY_MAP.items()}

class QRCodeDataTooLongError(ValueError):
    """Exception raised when the input data exceeds the maximum supported QR code length."""
    pass

# =====================
# Public API functions
# =====================

def generate_qr_code(
    data: str,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    error_correction: Optional[int] = None,
) -> Image.Image:
    """
    Generates a QR code image from the input data and optionally saves it to output_path.
    Allows specifying error correction level (ERROR_CORRECT_L, _M, _Q, _H).
    If not provided, auto-selects the highest possible error correction level for the data length.
    Raises QRCodeDataTooLongError if the data is too long for a QR code.
    Returns a PIL Image object.
    Example:
        img = generate_qr_code('Hello World', output_path='qrcode.png')
    """
    if error_correction is not None:
        ec_level = error_correction
    else:
        ec_level = _auto_select_error_correction(data)
    max_bytes = _get_max_bytes(ec_level)
    if len(data.encode("utf-8")) > max_bytes:
        raise QRCodeDataTooLongError(
            f"Input data is too long for a QR code (max {max_bytes} bytes at error correction level, got {len(data.encode('utf-8'))} bytes)."
        )
    qr = QRCode(box_size=box_size, border=border, error_correction=ec_level)
    qr.add_data(data)
    qr.make(fit=True)
    img = (
        qr.make_image(fill_color="black", back_color="white").get_image().convert("RGB")
    )
    if output_path:
        img.save(output_path)
    return img

def generate_colored_qr_code(
    data: str,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    pad_char: str = "\0",
) -> Image.Image:
    """
    Generates a colored QR code by splitting the data into 3 pieces, encoding each as a QR code,
    coloring each channel, and merging into a single RGB image.
    """
    def postprocess(qr_imgs: Sequence[Image.Image]) -> Image.Image:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        color_layers = []
        for img, color in zip(qr_imgs, colors):
            gray = img.convert("L")
            inv = Image.eval(gray, lambda x: 255 - x)
            arr = np.array(inv)
            color_img = np.zeros((*arr.shape, 3), dtype=np.uint8)
            mask = arr == 255
            for c in range(3):
                color_img[..., c][mask] = color[c]
            color_layers.append(Image.fromarray(color_img, "RGB"))
        merged = np.maximum.reduce([np.array(layer) for layer in color_layers])
        return Image.fromarray(merged, "RGB")
    return _encode_qr_parts(
        data,
        n_parts=3,
        postprocess=postprocess,
        output_path=output_path,
        box_size=box_size,
        border=border,
        pad_char=pad_char,
    )

def generate_colored_qr_code_im(
    data: str,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    pad_char: str = "\0",
) -> Image.Image:
    """
    Generates a colored QR code with Intensity Modulation (IM) to double data density per RGB channel.
    """
    def postprocess(qr_imgs: Sequence[Image.Image]) -> Image.Image:
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
                    bit_dim = int(arr_dim[y, x] < 128)
                    bit_bright = int(arr_bright[y, x] < 128)
                    val = IM_INTENSITY_MAP[(bit_dim, bit_bright)]
                    channel[y, x] = val
            result_arr[..., color[0]] = channel
        return Image.fromarray(result_arr, "RGB")
    return _encode_qr_parts(
        data,
        n_parts=6,
        postprocess=postprocess,
        output_path=output_path,
        box_size=box_size,
        border=border,
        pad_char=pad_char,
    )

def decode_qr_code(image: Union[str, Image.Image, np.ndarray]) -> Optional[str]:
    """
    Decodes a QR code from the given image (file path, PIL Image, or numpy array) and returns the decoded string, or None if not found.
    Args:
        image: File path, PIL.Image.Image, or numpy.ndarray.
    Returns:
        Decoded string, or None if not found.
    """
    img = _load_image_for_decode(image)
    decoded_objs = pyzbar_decode(img)
    if decoded_objs:
        return decoded_objs[0].data.decode("utf-8")
    return None

def decode_colored_qr_code(image: Union[str, Image.Image, np.ndarray], pad_char: str = "\0") -> str:
    """
    Decodes a colored QR code generated by generate_colored_qr_code.
    Accepts file path, PIL Image, or numpy array.
    The process is:
        1. Load the image and split into R, G, B channels.
        2. For each channel, convert to binary (white/black) using a threshold.
        3. Invert the image (swap black and white).
        4. Decode each QR code from the processed channel.
        5. Remove padding character from the end of each part.
        6. Concatenate the decoded strings in RGB order.
    Args:
        image: File path, PIL.Image.Image, or numpy.ndarray.
        pad_char: The character used for padding during encoding.
    Returns:
        The combined decoded string from all three QR codes.
    Raises:
        ValueError: If any of the QR codes cannot be decoded.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError("Input must be a file path, PIL.Image.Image, or numpy.ndarray.")
    channels = img.split()  # R, G, B
    decoded_pieces = []
    for channel in channels:
        arr = np.array(channel)
        binary = (arr > 128).astype(np.uint8) * 255
        inverted = 255 - binary
        qr_img = Image.fromarray(inverted, "L").convert("RGB")
        qr_arr = np.array(qr_img)
        decoded_objs = pyzbar_decode(qr_arr)
        if not decoded_objs:
            raise ValueError("Failed to decode one of the QR code channels.")
        part = decoded_objs[0].data.decode("utf-8")
        part = part.rstrip(pad_char)
        decoded_pieces.append(part)
    return "".join(decoded_pieces)

def decode_colored_qr_code_im(image: Union[str, Image.Image, np.ndarray], pad_char: str = "\0") -> str:
    """
    Decodes a colored QR code generated by generate_colored_qr_code_im.
    Accepts file path, PIL Image, or numpy array.
    For each channel, uses IM_INTENSITY_REVERSE_MAP to extract two QR codes, decodes all 6, and concatenates the result.
    Applies a median filter to reconstructed QR images to reduce JPEG noise (using Pillow).
    Args:
        image: File path, PIL.Image.Image, or numpy.ndarray.
        pad_char: The character used for padding during encoding.
    Returns:
        The combined decoded string from all six QR codes.
    Raises:
        ValueError: If any of the QR codes cannot be decoded.
    """
    from PIL import ImageFilter
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError("Input must be a file path, PIL.Image.Image, or numpy.ndarray.")
    channels = img.split()  # R, G, B
    decoded_pieces = []
    for idx, channel in enumerate(channels):
        arr = np.array(channel)
        # Reconstruct the two QR code images for this channel
        qr_dim = np.full(arr.shape, 255, dtype=np.uint8)  # start as white
        qr_bright = np.full(arr.shape, 255, dtype=np.uint8)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                val = int(arr[y, x])
                nearest = _nearest_intensity(val)
                bits = IM_INTENSITY_REVERSE_MAP[nearest]
                # If bit is 1, set black (0), else white (255)
                qr_dim[y, x] = 0 if bits[0] else 255
                qr_bright[y, x] = 0 if bits[1] else 255
        # Apply median filter to reduce JPEG noise (Pillow)
        qr_img_dim = Image.fromarray(qr_dim, "L").filter(ImageFilter.MedianFilter(size=3)).convert("RGB")
        qr_arr_dim = np.array(qr_img_dim)
        decoded_objs_dim = pyzbar_decode(qr_arr_dim)
        if not decoded_objs_dim:
            raise ValueError(f"Failed to decode dim QR code in channel {idx}.")
        part_dim = decoded_objs_dim[0].data.decode("utf-8").rstrip(pad_char)
        decoded_pieces.append(part_dim)
        qr_img_bright = Image.fromarray(qr_bright, "L").filter(ImageFilter.MedianFilter(size=3)).convert("RGB")
        qr_arr_bright = np.array(qr_img_bright)
        decoded_objs_bright = pyzbar_decode(qr_arr_bright)
        if not decoded_objs_bright:
            raise ValueError(f"Failed to decode bright QR code in channel {idx}.")
        part_bright = decoded_objs_bright[0].data.decode("utf-8").rstrip(pad_char)
        decoded_pieces.append(part_bright)
    return "".join(decoded_pieces)

# =====================
# Internal helper functions
# =====================

def _get_max_bytes(error_correction: int) -> int:
    """
    Returns the maximum number of bytes supported for a QR code (version 40, Byte mode)
    at the given error correction level.
    """
    if error_correction == ERROR_CORRECT_L:
        return 2953
    elif error_correction == ERROR_CORRECT_M:
        return 2331
    elif error_correction == ERROR_CORRECT_Q:
        return 1663
    elif error_correction == ERROR_CORRECT_H:
        return 1273
    else:
        raise ValueError("Invalid error correction level.")

def _auto_select_error_correction(data: str) -> int:
    """
    Selects the highest error correction constant that can fit the data.
    Returns the error correction constant (ERROR_CORRECT_H, _Q, _M, _L).
    """
    length = len(data.encode("utf-8"))
    for level in [ERROR_CORRECT_H, ERROR_CORRECT_Q, ERROR_CORRECT_M, ERROR_CORRECT_L]:
        if length <= _get_max_bytes(level):
            return level
    raise QRCodeDataTooLongError(
        f"Input data is too long for a QR code (max {_get_max_bytes(ERROR_CORRECT_L)} bytes in Byte mode, got {length} bytes)."
    )

def _split_and_pad_data(data: str, n_parts: int, pad_char: str) -> List[str]:
    """
    Split data into n_parts, padding each part to the same length with pad_char.
    """
    n = len(data)
    k, m = divmod(n, n_parts)
    parts = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]
    maxlen = max(len(p) for p in parts)
    return [p.ljust(maxlen, pad_char) for p in parts]

def _load_image_for_decode(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
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

def _nearest_intensity(val: int) -> int:
    """
    Find the nearest valid intensity value for IM decoding.
    """
    return min(IM_INTENSITY_REVERSE_MAP.keys(), key=lambda k: abs(k - val))

def _encode_qr_parts(
    data: str,
    n_parts: int,
    postprocess: Callable[[Sequence[Image.Image]], Image.Image],
    *,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    pad_char: str = "\0",
) -> Image.Image:
    """
    Generic helper for encoding data into one or more QR codes, then post-processing the images.
    Args:
        data: The input string to encode.
        n_parts: Number of parts to split the data into.
        postprocess: Function that takes a sequence of QR code images and returns the final image.
        output_path: Optional path to save the result.
        box_size: QR code box size.
        border: QR code border size.
        pad_char: Padding character for data splitting.
    Returns:
        The final PIL Image.
    Raises:
        QRCodeDataTooLongError: If input data is empty.
    """
    if not data:
        raise QRCodeDataTooLongError("Input data must have at least 1 character to generate a QR code.")
    data_pieces = _split_and_pad_data(data, n_parts, pad_char)
    ec_level = _auto_select_error_correction(data_pieces[0])
    qr_imgs = [
        generate_qr_code(
            piece,
            box_size=box_size,
            border=border,
            error_correction=ec_level
        )
        for piece in data_pieces
    ]
    # Ensure all images are the same size
    size = qr_imgs[0].size
    qr_imgs = [img.resize(size, resample=Image.Resampling.NEAREST) for img in qr_imgs]
    result_img = postprocess(qr_imgs)
    if output_path:
        result_img.save(output_path)
    return result_img
