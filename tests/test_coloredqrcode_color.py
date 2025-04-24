import os
import tempfile
import unittest
from PIL import Image
from coloredqrcode import (
    generate_colored_qr_code,
    QRCodeDataTooLongError,
    decode_colored_qr_code,
)
import shutil
import numpy as np


class TestGenerateColoredQRCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent_dir = os.path.join(os.path.dirname(__file__), "qr_outputs")
        cls.output_dir = os.path.join(parent_dir, "color")
        # Only remove the 'color' subdirectory, not the whole qr_outputs tree
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Optionally clean up after all tests, or leave for manual inspection
        pass

    def test_generate_colored_qr_code_basic(self):
        data = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        img_path = os.path.join(self.output_dir, "colored_qr_basic.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.size[0], 0)
        self.assertGreater(img.size[1], 0)

    def test_generate_colored_qr_code_short_data(self):
        # Now only empty input should raise
        with self.assertRaises(QRCodeDataTooLongError):
            generate_colored_qr_code("")

    def test_generate_colored_qr_code_visual(self):
        data = "The quick brown fox jumps over the lazy dog 1234567890!@#"
        img_path = os.path.join(self.output_dir, "colored_qr_visual.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        self.assertIsInstance(img, Image.Image)
        self.assertTrue(os.path.exists(img_path))
        # img.save("/tmp/colored_qr_visual.png")

    def test_decode_colored_qr_code_basic(self):
        data = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        img_path = os.path.join(self.output_dir, "colored_qr_basic.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        decoded = decode_colored_qr_code(img_path)
        print(f"Decoded (basic): {decoded}")
        self.assertEqual(decoded, data)

    def test_decode_colored_qr_code_visual(self):
        data = "The quick brown fox jumps over the lazy dog 1234567890!@#"
        img_path = os.path.join(self.output_dir, "colored_qr_visual.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        decoded = decode_colored_qr_code(img_path)
        print(f"Decoded (visual): {decoded}")
        self.assertEqual(decoded, data)

    def test_colored_qr_code_max_length(self):
        # The max length for a normal QR code (L, version 40, byte mode) is 2953 bytes
        max_normal = 2953
        max_colored = max_normal * 3
        import random
        import string
        # Generate random data so that when split into 3 parts, each part is different
        data_parts = [
            ''.join(random.choices(string.ascii_letters + string.digits, k=max_normal))
            for _ in range(3)
        ]
        data = ''.join(data_parts)
        img_path = os.path.join(self.output_dir, "colored_qr_max.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        decoded = decode_colored_qr_code(img_path)
        print(f"Decoded (max): length={len(decoded)}")
        self.assertEqual(decoded, data)
        # Exceeding max should raise
        with self.assertRaises(QRCodeDataTooLongError):
            too_long_data = ''.join([
                ''.join(random.choices(string.ascii_letters + string.digits, k=max_normal))
                for _ in range(3)
            ]) + 'x'
            generate_colored_qr_code(too_long_data)

    def test_decode_colored_qr_code_jpeg_compression(self):
        """
        Test decoding colored QR code after saving as JPEG with various compression levels and after multiple re-encodings.
        Save all JPEGs with quality in filename for visual inspection. Use a subfolder for color QR code tests.
        """
        data = "JPEG compression test: The quick brown fox jumps over the lazy dog 1234567890!@#"
        img_path = os.path.join(self.output_dir, "colored_qr_jpeg_original.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        # Test different JPEG quality levels
        for quality in [95, 75, 50]:
            jpeg_path = os.path.join(self.output_dir, f"colored_qr_jpeg_q{quality}.jpg")
            img.save(jpeg_path, format="JPEG", quality=quality)
            decoded = decode_colored_qr_code(jpeg_path)
            self.assertEqual(decoded, data, f"Failed at JPEG quality {quality}")
        # Test multiple JPEG re-encodings (5 times) for each quality
        for quality in [95, 75, 50]:
            jpeg_path = os.path.join(self.output_dir, f"colored_qr_jpeg_q{quality}_5x.jpg")
            img_temp = img.copy()
            img_temp.save(jpeg_path, format="JPEG", quality=quality)
            for i in range(5):
                img_temp = Image.open(jpeg_path)
                img_temp.save(jpeg_path, format="JPEG", quality=quality)
            decoded = decode_colored_qr_code(jpeg_path)
            self.assertEqual(decoded, data, f"Failed after 5x JPEG re-encodings at quality {quality}")

    def test_decode_colored_qr_code_from_image_object(self):
        """Test decoding from a PIL.Image.Image object, not just file path."""
        data = "Test image object input for decode_colored_qr_code"
        img = generate_colored_qr_code(data)
        # Decode directly from image object
        decoded = decode_colored_qr_code(img)
        self.assertEqual(decoded, data)

    def test_decode_colored_qr_code_from_numpy_array(self):
        """Test decoding from a numpy array input."""
        data = "Test numpy array input for decode_colored_qr_code"
        img = generate_colored_qr_code(data)
        arr = np.array(img)
        decoded = decode_colored_qr_code(arr)
        self.assertEqual(decoded, data)


if __name__ == "__main__":
    unittest.main()
