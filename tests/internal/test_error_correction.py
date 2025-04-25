"""Unit tests for error correction module."""
import unittest

from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

from coloredqrcode.error_correction import (
    auto_select_error_correction,
    get_max_bytes,
)


class TestErrorCorrection(unittest.TestCase):
    """Tests for error_correction.py functionality."""

    def test_auto_select_error_correction(self):
        """Test that error correction level is correctly auto-selected based on data length."""
        test_cases = [
            # Normal cases within ranges
            (1000, ERROR_CORRECT_H),  # Well within H range (0-1273)
            (1500, ERROR_CORRECT_Q),  # Well within Q range (1274-1663)
            (2000, ERROR_CORRECT_M),  # Well within M range (1664-2331)
            (2500, ERROR_CORRECT_L),  # Well within L range (2332-2953)
            
            # Edge cases at boundaries
            (1273, ERROR_CORRECT_H),  # Max for H
            (1274, ERROR_CORRECT_Q),  # Min for Q
            (1663, ERROR_CORRECT_Q),  # Max for Q
            (1664, ERROR_CORRECT_M),  # Min for M
            (2331, ERROR_CORRECT_M),  # Max for M
            (2332, ERROR_CORRECT_L),  # Min for L
            (2953, ERROR_CORRECT_L),  # Max for L
        ]
        for length, expected_level in test_cases:
            data = "a" * length
            level = auto_select_error_correction(data)
            self.assertEqual(
                level,
                expected_level,
                f"Expected level {expected_level} for length {length}, got {level}",
            )

    def test_get_max_bytes(self):
        """Test that correct maximum bytes are returned for each error correction level."""
        expected_max_bytes = {
            ERROR_CORRECT_L: 2953,
            ERROR_CORRECT_M: 2331,
            ERROR_CORRECT_Q: 1663,
            ERROR_CORRECT_H: 1273,
        }
        for level, expected in expected_max_bytes.items():
            self.assertEqual(
                get_max_bytes(level),
                expected,
                f"Incorrect max bytes for level {level}",
            )


if __name__ == "__main__":
    unittest.main()
