from qrcode import QRCode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np


from typing import Optional

def generate_qr_code(data: str, output_path: Optional[str] = None, box_size: int = 10, border: int = 4) -> Image.Image:
    """
    Generate a QR code image from the input data.
    Args:
        data (str): The message or data to encode.
        output_path (str, optional): If provided, saves the image to this path.
        box_size (int): Size of each QR code box.
        border (int): Border size (boxes).
    Returns:
        Image.Image: The generated QR code image.
    """
    qr = QRCode(box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").get_image().convert('RGB')
    if output_path:
        img.save(output_path)
    return img


def decode_qr_code(image_path: str) -> Optional[str]:
    """
    Decode a QR code image to extract the message/data.
    Args:
        image_path (str): Path to the QR code image.
    Returns:
        str: The decoded message/data, or None if not found.
    """
    img = cv2.imread(image_path)
    decoded_objs = pyzbar_decode(img)
    if decoded_objs:
        return decoded_objs[0].data.decode('utf-8')
    return None
