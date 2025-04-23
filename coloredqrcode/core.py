from qrcode import QRCode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
from typing import Optional

# Usage:
# img = generate_qr_code('Hello World', output_path='qrcode.png')
# result = decode_qr_code('qrcode.png')

def generate_qr_code(data: str, output_path: Optional[str] = None, box_size: int = 10, border: int = 4) -> Image.Image:
    # Generates a QR code image from the input data and optionally saves it to output_path.
    # Returns a PIL Image object.
    # Example:
    #   img = generate_qr_code('Hello World', output_path='qrcode.png')
    qr = QRCode(box_size=box_size, border=border)
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
