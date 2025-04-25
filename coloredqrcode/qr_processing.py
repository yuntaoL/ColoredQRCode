"""
Core encoding and decoding functions for QR code processing.
"""

"""Core encoding and decoding functions for QR code processing."""
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
from qrcode import QRCode

from .error_correction import auto_select_error_correction, get_max_bytes
from .exceptions import QRCodeDataTooLongError
from .utils import split_and_pad_data


def create_basic_qr_code(
    data: str,
    box_size: int,
    border: int,
    error_correction: int,
) -> Image.Image:
    """
    Internal function to create a basic QR code with the given parameters.
    Auto-selects error correction level if not provided and validates data length.

    Args:
        data: The string data to encode
        box_size: Size of each QR code box
        border: Width of the QR code border
        error_correction: QR code error correction level, if not provided will be auto-selected

    Returns:
        PIL Image object containing the QR code

    Raises:
        QRCodeDataTooLongError: If the data is too long for the QR code at the given error correction level
    """
    max_bytes = get_max_bytes(error_correction)
    if len(data.encode("utf-8")) > max_bytes:
        raise QRCodeDataTooLongError(
            f"Input data is too long for a QR code (max {max_bytes} bytes at error correction level, got {len(data.encode('utf-8'))} bytes)."
        )

    qr = QRCode(box_size=box_size, border=border, error_correction=error_correction)
    qr.add_data(data)
    qr.make(fit=True)
    return (
        qr.make_image(fill_color="black", back_color="white").get_image().convert("RGB")
    )


def decode_qr_parts(
    image: Image.Image,
    preprocess: Callable[[Image.Image], List[Image.Image]],
    pad_char: str = "\0",
) -> str:
    """
    Generic helper for decoding one or more QR codes from an image after preprocessing.
    Args:
        image: PIL Image object containing the QR code(s).
        preprocess: Function that takes a PIL Image and returns a list of QR code images to decode.
        pad_char: Character to strip from the end of each decoded part.
    Returns:
        The concatenated decoded string from all QR codes.
    Raises:
        ValueError: If any QR code cannot be decoded.
    """
    img = image.convert("RGB")
    qr_imgs = preprocess(img)
    decoded_pieces = []
    for qr_img in qr_imgs:
        arr = np.array(qr_img)
        decoded_objs = pyzbar_decode(arr)
        if not decoded_objs:
            raise ValueError("Failed to decode one of the QR code images.")
        part = decoded_objs[0].data.decode("utf-8")
        if pad_char is not None:
            part = part.rstrip(pad_char)
        decoded_pieces.append(part)
    return "".join(decoded_pieces)


def encode_qr_parts(
    data: str,
    n_parts: int,
    postprocess: Callable[[Sequence[Image.Image]], Image.Image],
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
        box_size: QR code box size.
        border: QR code border size.
        pad_char: Padding character for data splitting.
    Returns:
        The final PIL Image.
    Raises:
        QRCodeDataTooLongError: If input data is empty.
    """
    if not data:
        from .exceptions import QRCodeDataTooLongError

        raise QRCodeDataTooLongError(
            "Input data must have at least 1 character to generate a QR code."
        )

    data_pieces = split_and_pad_data(data, n_parts, pad_char)
    ec_level = auto_select_error_correction(data_pieces[0])

    qr_imgs = [
        create_basic_qr_code(
            piece, box_size=box_size, border=border, error_correction=ec_level
        )
        for piece in data_pieces
    ]
    # Ensure all images are the same size
    size = qr_imgs[0].size
    qr_imgs = [img.resize(size, resample=Image.Resampling.NEAREST) for img in qr_imgs]
    result_img = postprocess(qr_imgs)
    return result_img
