import os
import tempfile
import unittest
from coloredqrcode import generate_qr_code, decode_qr_code

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
            Image.new('RGB', (100, 100), color='white').save(img_path)
            decoded = decode_qr_code(img_path)
            self.assertIsNone(decoded)

if __name__ == "__main__":
    unittest.main()
