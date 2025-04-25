import os
import tempfile
import unittest
import pytest
import numpy as np
from coloredqrcode import generate_qr_code, decode_qr_code, QRCodeDataTooLongError
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H


class TestColoredQRCode(unittest.TestCase):
    def test_generate_and_decode_qr_code(self):
        message = "Hello, QR!"
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_qr.png")
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
        # Data that exceeds the QR code max byte length (2953 bytes for L)
        data = "a" * 2954
        with pytest.raises(QRCodeDataTooLongError) as excinfo:
            generate_qr_code(data, error_correction=ERROR_CORRECT_L)
        assert "Input data is too long for a QR code" in str(excinfo.value)

    def test_generate_qr_code_error_correction_levels(self):
        # Test that different error correction levels accept correct data sizes
        levels = [
            (ERROR_CORRECT_L, 2953),
            (ERROR_CORRECT_M, 2331),
            (ERROR_CORRECT_Q, 1663),
            (ERROR_CORRECT_H, 1273),
        ]
        for level, max_bytes in levels:
            data = "a" * max_bytes
            try:
                img = generate_qr_code(data, error_correction=level)
                assert img is not None
            except QRCodeDataTooLongError:
                pytest.fail(f"Should not fail for {level} with {max_bytes} bytes")
            # Exceeding should raise
            with pytest.raises(QRCodeDataTooLongError):
                generate_qr_code("a" * (max_bytes + 1), error_correction=level)

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
