"""Unit tests for the core QR code generation and decoding functionality."""
import os
import tempfile
import unittest

import numpy as np
import pytest
from PIL import Image

from coloredqrcode import QRCodeDataTooLongError, decode_qr_code, generate_qr_code


class TestColoredQRCode(unittest.TestCase):
    def test_generate_and_decode_qr_code(self):
        """Test generating and decoding a standard QR code, and save output to qr_outputs/normal/ for documentation/demo purposes."""
        message = "Hello, QR!"
        output_dir = os.path.join(os.path.dirname(__file__), "qr_outputs", "normal")
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, "test_qr_basic.png")
        # Generate QR code
        img = generate_qr_code(message, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        # Decode QR code
        decoded = decode_qr_code(img_path)
        self.assertEqual(decoded, message)

    def test_generate_and_decode_qr_code_max(self):
        """Test generating and decoding a standard QR code with max data, and save output to qr_outputs/normal/ for documentation/demo purposes."""
        message = "a" * 1273  # Max for error correction H (auto-selected for best quality)
        output_dir = os.path.join(os.path.dirname(__file__), "qr_outputs", "normal")
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, "test_qr_max.png")
        # Generate QR code
        img = generate_qr_code(message, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        # Decode QR code
        decoded = decode_qr_code(img_path)
        self.assertEqual(decoded, message)

    def test_decode_invalid_image(self):
        # Try to decode a non-QR image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "blank.png")
            from PIL import Image

            Image.new("RGB", (100, 100), color="white").save(img_path)
            decoded = decode_qr_code(img_path)
            self.assertIsNone(decoded)

    def test_generate_qr_code_too_long(self):
        """Test that the public API properly handles data that is too long"""
        data = "a" * 2954  # Exceeds max possible length
        with pytest.raises(QRCodeDataTooLongError) as excinfo:
            generate_qr_code(data)
        assert "Input data is too long for a QR code" in str(excinfo.value)

    def test_generate_qr_code_auto_error_correction(self):
        # Should auto-select the highest error correction that fits
        # These assertions check that the function does not raise and returns an image
        assert generate_qr_code("a" * 1000) is not None  # Should use H
        assert generate_qr_code("a" * 1500) is not None  # Should use Q
        assert generate_qr_code("a" * 2400) is not None  # Should use M
        assert generate_qr_code("a" * 2953) is not None  # Should use L
        with pytest.raises(QRCodeDataTooLongError):
            generate_qr_code("a" * 2954)

    def test_decode_qr_code_from_image_object(self):
        """Test decoding from a PIL.Image.Image object, not just file path."""
        message = "Hello, QR! (image object)"
        img = generate_qr_code(message)
        decoded = decode_qr_code(img)
        self.assertEqual(decoded, message)

    def test_decode_qr_code_from_numpy_array(self):
        """Test decoding from a numpy array input."""
        message = "Hello, QR! (numpy array)"
        img = generate_qr_code(message)
        arr = np.array(img)
        decoded = decode_qr_code(arr)
        self.assertEqual(decoded, message)


if __name__ == "__main__":
    unittest.main()
