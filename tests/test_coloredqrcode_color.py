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


class TestGenerateColoredQRCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(os.path.dirname(__file__), "qr_outputs")
        # Clean the output directory before all tests
        if os.path.exists(cls.output_dir):
            for f in os.listdir(cls.output_dir):
                fp = os.path.join(cls.output_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
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
        with self.assertRaises(QRCodeDataTooLongError):
            generate_colored_qr_code("ab")

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
        data = "a" * max_colored
        img_path = os.path.join(self.output_dir, "colored_qr_max.png")
        img = generate_colored_qr_code(data, output_path=img_path)
        self.assertTrue(os.path.exists(img_path))
        decoded = decode_colored_qr_code(img_path)
        print(f"Decoded (max): length={len(decoded)}")
        self.assertEqual(decoded, data)
        # Exceeding max should raise
        with self.assertRaises(QRCodeDataTooLongError):
            generate_colored_qr_code("a" * (max_colored + 1))


if __name__ == "__main__":
    unittest.main()
