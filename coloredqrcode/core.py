from qrcode import QRCode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
from typing import Optional, Literal, Tuple, List
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)


class QRCodeDataTooLongError(ValueError):
    """Exception raised when the input data exceeds the maximum supported QR code length."""

    pass


# Usage:
# img = generate_qr_code('Hello World', output_path='qrcode.png')
# result = decode_qr_code('qrcode.png')


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
    Selects the highest error correction level that can fit the data.
    """
    length = len(data.encode("utf-8"))
    for level in [ERROR_CORRECT_H, ERROR_CORRECT_Q, ERROR_CORRECT_M, ERROR_CORRECT_L]:
        if length <= _get_max_bytes(level):
            return level
    raise QRCodeDataTooLongError(
        f"Input data is too long for a QR code (max {_get_max_bytes(ERROR_CORRECT_L)} bytes in Byte mode, got {length} bytes)."
    )


def generate_qr_code(
    data: str,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    error_correction: Optional[Literal["L", "M", "Q", "H"]] = None,
) -> Image.Image:
    """
    Generates a QR code image from the input data and optionally saves it to output_path.
    Allows specifying error correction level ('L', 'M', 'Q', 'H').
    If not provided, auto-selects the highest possible error correction level for the data length.
    Raises QRCodeDataTooLongError if the data is too long for a QR code.
    Returns a PIL Image object.
    Example:
        img = generate_qr_code('Hello World', output_path='qrcode.png')
    """
    ec_map = {
        "L": ERROR_CORRECT_L,
        "M": ERROR_CORRECT_M,
        "Q": ERROR_CORRECT_Q,
        "H": ERROR_CORRECT_H,
    }
    if error_correction:
        ec_level = ec_map[error_correction]
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


# Usage:
# result = decode_qr_code('qrcode.png')
def decode_qr_code(image_path: str) -> Optional[str]:
    # Decodes a QR code from the given image path and returns the decoded string, or None if not found.
    # Example:
    #   result = decode_qr_code('qrcode.png')
    img = cv2.imread(image_path)
    decoded_objs = pyzbar_decode(img)
    if decoded_objs:
        return decoded_objs[0].data.decode("utf-8")
    return None


def generate_colored_qr_code(
    data: str,
    output_path: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    pad_char: str = "\0",
) -> Image.Image:
    """
    Generates a colored QR code by splitting the data into 3 pieces, padding each to the same length with a special character, encoding each as a QR code,
    inverting each QR code, coloring the white part as Red, Green, or Blue, and merging them into a single RGB image.
    Optionally saves the result to output_path.
    Returns a PIL Image object.
    """
    if len(data) == 0:
        raise QRCodeDataTooLongError("Input data must have at least 1 character to generate a colored QR code.")
    def split_and_pad_data(data: str, pad_char: str) -> List[str]:
        n = len(data)
        k, m = divmod(n, 3)
        parts = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(3)]
        maxlen = max(len(p) for p in parts)
        return [p.ljust(maxlen, pad_char) for p in parts]

    data_pieces = split_and_pad_data(data, pad_char)
    # All parts have the same length after padding
    ec_level = _auto_select_error_correction(data_pieces[0])
    ec_map_inv = {
        ERROR_CORRECT_L: "L",
        ERROR_CORRECT_M: "M",
        ERROR_CORRECT_Q: "Q",
        ERROR_CORRECT_H: "H",
    }
    ec_letter = ec_map_inv[ec_level]
    qr_imgs = [generate_qr_code(piece, box_size=box_size, border=border, error_correction=ec_letter) for piece in data_pieces]  # type: ignore[arg-type]
    size = qr_imgs[0].size
    qr_imgs = [img.resize(size, resample=Image.Resampling.NEAREST) for img in qr_imgs]
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
    result_img = Image.fromarray(merged, "RGB")
    if output_path:
        result_img.save(output_path)
    return result_img


def decode_colored_qr_code(image_path: str, pad_char: str = "\0") -> str:
    """
    Decodes a colored QR code generated by generate_colored_qr_code.
    The process is:
        1. Load the image and split into R, G, B channels.
        2. For each channel, convert to binary (white/black) using a threshold.
        3. Invert the image (swap black and white).
        4. Decode each QR code from the processed channel.
        5. Remove padding character from the end of each part.
        6. Concatenate the decoded strings in RGB order.
    Args:
        image_path: Path to the colored QR code image.
        pad_char: The character used for padding during encoding.
    Returns:
        The combined decoded string from all three QR codes.
    Raises:
        ValueError: If any of the QR codes cannot be decoded.
    """
    img = Image.open(image_path).convert("RGB")
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
