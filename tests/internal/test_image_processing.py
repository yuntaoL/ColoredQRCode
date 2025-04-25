"""Unit tests for image processing module."""
import unittest

from PIL import Image
import numpy as np

from coloredqrcode.image_processing import (
    preprocess_colored_qr,
    preprocess_colored_qr_im,
    postprocess_colored_qr,
    postprocess_colored_qr_im,
)


class TestImageProcessing(unittest.TestCase):
    """Tests for image_processing.py functionality."""

    def setUp(self):
        """Set up test data."""
        self.test_image = Image.new("RGB", (3, 3), color="white")

    def test_preprocess_colored_qr(self):
        """Test colored QR code preprocessing."""
        result = preprocess_colored_qr(self.test_image)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # RGB channels
        for img in result:
            self.assertIsInstance(img, Image.Image)

    def test_preprocess_colored_qr_im(self):
        """Test intensity-modulated colored QR code preprocessing."""
        result = preprocess_colored_qr_im(self.test_image)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)  # 2 levels per RGB channel
        for img in result:
            self.assertIsInstance(img, Image.Image)

    def test_postprocess_colored_qr(self):
        """Test colored QR code postprocessing."""
        qr_images = [Image.new("RGB", (50, 50), color=c) for c in ["red", "green", "blue"]]
        result = postprocess_colored_qr(qr_images)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, qr_images[0].size)

    def test_postprocess_colored_qr_im(self):
        """Test intensity-modulated colored QR code postprocessing."""
        qr_images = [Image.new("RGB", (50, 50)) for _ in range(6)]
        result = postprocess_colored_qr_im(qr_images)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, qr_images[0].size)


if __name__ == "__main__":
    unittest.main()
