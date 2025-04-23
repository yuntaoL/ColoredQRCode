from qrcode import QRCode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
from typing import Optional, Literal
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H

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
    length = len(data.encode('utf-8'))
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
    error_correction: Optional[Literal['L', 'M', 'Q', 'H']] = None
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
    ec_map = {'L': ERROR_CORRECT_L, 'M': ERROR_CORRECT_M, 'Q': ERROR_CORRECT_Q, 'H': ERROR_CORRECT_H}
    if error_correction:
        ec_level = ec_map[error_correction]
    else:
        ec_level = _auto_select_error_correction(data)
    max_bytes = _get_max_bytes(ec_level)
    if len(data.encode('utf-8')) > max_bytes:
        raise QRCodeDataTooLongError(
            f"Input data is too long for a QR code (max {max_bytes} bytes at error correction level, got {len(data.encode('utf-8'))} bytes)."
        )
    qr = QRCode(box_size=box_size, border=border, error_correction=ec_level)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").get_image().convert('RGB')
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
        return decoded_objs[0].data.decode('utf-8')
    return None
