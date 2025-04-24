import os
import unittest
from PIL import Image
from coloredqrcode import generate_colored_qr_code_im, decode_colored_qr_code_im, QRCodeDataTooLongError

class TestGenerateColoredQRCodeIM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(os.path.dirname(__file__), "qr_outputs")
        if os.path.exists(cls.output_dir):
            for f in os.listdir(cls.output_dir):
                fp = os.path.join(cls.output_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
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

if __name__ == "__main__":
    unittest.main()
