from qrcode import QRCode
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
from typing import Optional

def generate_qr_code(data: str, output_path: Optional[str] = None, box_size: int = 10, border: int = 4) -> Image.Image:
    qr = QRCode(box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").get_image().convert('RGB')
    if output_path:
        img.save(output_path)
    return img


def decode_qr_code(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path)
    decoded_objs = pyzbar_decode(img)
    if decoded_objs:
        return decoded_objs[0].data.decode('utf-8')
    return None
