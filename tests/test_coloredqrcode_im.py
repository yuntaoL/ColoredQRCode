import os
import unittest
import shutil
import pytest
from PIL import Image
import numpy as np
from coloredqrcode import generate_colored_qr_code_im, decode_colored_qr_code_im, QRCodeDataTooLongError
import pytest

class TestGenerateColoredQRCodeIM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(os.path.dirname(__file__), "qr_outputs", "color_im")
        # Clean the output subdirectory before all tests
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)
        os.makedirs(cls.output_dir, exist_ok=True)

    def test_generate_colored_qr_code_im_basic(self):
        data = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        img_path = os.path.join(self.output_dir, "colored_qr_im_basic.png")
        img = generate_colored_qr_code_im(data, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.size[0], 0)
        self.assertGreater(img.size[1], 0)

    def test_generate_colored_qr_code_im_short_data(self):
        with self.assertRaises(QRCodeDataTooLongError):
            generate_colored_qr_code_im("")

    def test_generate_and_decode_colored_qr_code_im(self):
        data = "The quick brown fox jumps over the lazy dog 1234567890!@#"
        img_path = os.path.join(self.output_dir, "colored_qr_im_visual.png")
        img = generate_colored_qr_code_im(data, output_path=img_path)
        decoded = decode_colored_qr_code_im(img_path)
        self.assertEqual(decoded, data)

    def test_colored_qr_code_im_max_length(self):
        # The max length for a normal QR code (L, version 40, byte mode) is 2953 bytes
        max_normal = 2953
        import random
        import string
        data_parts = [
            ''.join(random.choices(string.ascii_letters + string.digits, k=max_normal))
            for _ in range(6)
        ]
        data = ''.join(data_parts)
        img_path = os.path.join(self.output_dir, "colored_qr_im_max.png")
        img = generate_colored_qr_code_im(data, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        decoded = decode_colored_qr_code_im(img_path)
        self.assertEqual(decoded, data)
        # Exceeding max should raise
        with self.assertRaises(QRCodeDataTooLongError):
            too_long_data = ''.join([
                ''.join(random.choices(string.ascii_letters + string.digits, k=max_normal))
                for _ in range(6)
            ]) + 'x'
            generate_colored_qr_code_im(too_long_data)

    def test_decode_colored_qr_code_im_jpeg_compression(self):
        """
        Test decoding colored QR code (IM) after saving as JPEG with various compression levels.
        Save all JPEGs with quality in filename for visual inspection. Use a subfolder for color IM QR code tests.
        """
        data = "JPEG compression test: The quick brown fox jumps over the lazy dog 1234567890!@#"
        img_path = os.path.join(self.output_dir, "colored_qr_im_jpeg_original.png")
        img = generate_colored_qr_code_im(data, output_path=img_path)
        # Test different JPEG quality levels
        for quality in [95, 75, 50]:
            jpeg_path = os.path.join(self.output_dir, f"colored_qr_im_jpeg_q{quality}.jpg")
            img.save(jpeg_path, format="JPEG", quality=quality)
            decoded = decode_colored_qr_code_im(jpeg_path)
            self.assertEqual(decoded, data, f"Failed at JPEG quality {quality}")
        # Removed multiple re-encoding test for color IM

    def test_decode_colored_qr_code_im_from_image_object(self):
        """Test decoding IM QR from a PIL.Image.Image object, not just file path."""
        data = "Test image object input for decode_colored_qr_code_im"
        img = generate_colored_qr_code_im(data)
        decoded = decode_colored_qr_code_im(img)
        self.assertEqual(decoded, data)

    def test_decode_colored_qr_code_im_from_numpy_array(self):
        """Test decoding IM QR from a numpy array input."""
        data = "Test numpy array input for decode_colored_qr_code_im"
        img = generate_colored_qr_code_im(data)
        arr = np.array(img)
        decoded = decode_colored_qr_code_im(arr)
        self.assertEqual(decoded, data)

if __name__ == "__main__":
    unittest.main()
