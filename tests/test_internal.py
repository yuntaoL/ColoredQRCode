"""Unit tests for internal QR code processing functions."""
import unittest

import pytest
from PIL import Image
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

from coloredqrcode import QRCodeDataTooLongError
from coloredqrcode.qr_processing import create_basic_qr_code


class TestInternalFunctions(unittest.TestCase):
    def test_create_basic_qr_code_error_correction_levels(self):
        """Test that different error correction levels are handled correctly by create_basic_qr_code"""
        levels = [
            (ERROR_CORRECT_L, 2953),
            (ERROR_CORRECT_M, 2331),
            (ERROR_CORRECT_Q, 1663),
            (ERROR_CORRECT_H, 1273),
        ]
        for level, max_bytes in levels:
            data = "a" * max_bytes
            try:
                img = create_basic_qr_code(data, box_size=10, border=4, error_correction=level)
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.mode, "RGB")
            except QRCodeDataTooLongError:
                pytest.fail(f"Should not fail for {level} with {max_bytes} bytes")
            
            # Test exceeding the limit
            with pytest.raises(QRCodeDataTooLongError):
                create_basic_qr_code("a" * (max_bytes + 1), box_size=10, border=4, error_correction=level)

    def test_create_basic_qr_code_auto_error_correction(self):
        """Test that auto error correction selection works correctly"""
        test_cases = [
            (1000, ERROR_CORRECT_H),  # Should use H
            (1500, ERROR_CORRECT_Q),  # Should use Q
            (2400, ERROR_CORRECT_M),  # Should use M
            (2953, ERROR_CORRECT_L),  # Should use L
        ]
        for length, expected_level in test_cases:
            data = "a" * length
            img = create_basic_qr_code(data, box_size=10, border=4)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
