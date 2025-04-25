import os
import tempfile
import numpy as np
from PIL import Image
import pytest
from coloredqrcode.core import _encode_qr_parts, _decode_qr_parts, QRCodeDataTooLongError

def test_encode_qr_parts_basic():
    """Test that _encode_qr_parts splits and encodes data into multiple QR images."""
    data = "ABCDEFGHIJKL"
    n_parts = 3
    called = {}
    def postprocess(qr_imgs):
        called['count'] = len(qr_imgs)
        # Just return the first image for test
        return qr_imgs[0]
    img = _encode_qr_parts(data, n_parts, postprocess)
    assert isinstance(img, Image.Image)
    assert called['count'] == n_parts

def test_encode_qr_parts_empty():
    """Test that _encode_qr_parts raises on empty input."""
    with pytest.raises(QRCodeDataTooLongError):
        _encode_qr_parts("", 3, lambda imgs: imgs[0])

def test_decode_qr_parts_roundtrip():
    """Test that _decode_qr_parts decodes what _encode_qr_parts encodes (single part)."""
    data = "Hello, QR!"
    def postprocess(qr_imgs):
        return qr_imgs[0]
    img = _encode_qr_parts(data, 1, postprocess)
    def preprocess(img):
        return [img]
    decoded = _decode_qr_parts(img, preprocess)
    assert decoded == data

def test_decode_qr_parts_invalid():
    """Test that _decode_qr_parts raises on non-QR image."""
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    def preprocess(img):
        return [img]
    with pytest.raises(ValueError):
        _decode_qr_parts(img, preprocess)
