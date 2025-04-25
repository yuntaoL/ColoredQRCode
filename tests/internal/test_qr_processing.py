"""Unit tests for QR code processing module."""
from typing import List, Tuple
import unittest

from PIL import Image
import pytest
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

from coloredqrcode import QRCodeDataTooLongError
from coloredqrcode.qr_processing import (
    create_basic_qr_code,
    decode_qr_parts,
    encode_qr_parts,
)


class TestQRProcessing(unittest.TestCase):
    """Tests for qr_processing.py functionality."""

    def test_create_basic_qr_code_error_correction_levels(self):
        """Test that different error correction levels are handled correctly."""
        levels: List[Tuple[int, int]] = [
            (ERROR_CORRECT_L, 2953),
            (ERROR_CORRECT_M, 2331),
            (ERROR_CORRECT_Q, 1663),
            (ERROR_CORRECT_H, 1273),
        ]
        for level, max_bytes in levels:
            data = "a" * max_bytes
            try:
                img = create_basic_qr_code(
                    data, box_size=10, border=4, error_correction=level
                )
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.mode, "RGB")
            except QRCodeDataTooLongError:
                pytest.fail(f"Should not fail for {level} with {max_bytes} bytes")

            # Test exceeding the limit
            with pytest.raises(QRCodeDataTooLongError):
                create_basic_qr_code(
                    "a" * (max_bytes + 1), box_size=10, border=4, error_correction=level
                )

    def test_encode_decode_qr_parts(self):
        """Test encoding and decoding QR code parts."""
        test_data = "Hello, World!" * 100  # Large enough to split
        # Test with different numbers of parts
        for n_parts in [2, 3, 4]:
            # Store QR code parts for simulating merge/split
            stored_qr_parts = []

            def mock_postprocess(imgs):
                """Simulate merging by storing the input images."""
                nonlocal stored_qr_parts
                stored_qr_parts = imgs.copy()  # Store the QR parts
                return Image.new("RGB", imgs[0].size)  # Return dummy merged image

            def mock_preprocess(_):
                """Simulate splitting by returning the stored images."""
                return stored_qr_parts

            # Encode
            img = encode_qr_parts(
                test_data,
                n_parts,
                mock_postprocess,
                box_size=10,
                border=4,
            )
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(len(stored_qr_parts), n_parts)  # Verify correct number of parts

            # Decode
            decoded = decode_qr_parts(
                img,
                mock_preprocess,
                pad_char="\0",
            )
            self.assertEqual(decoded, test_data)


if __name__ == "__main__":
    unittest.main()
