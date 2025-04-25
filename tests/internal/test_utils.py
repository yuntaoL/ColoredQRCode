"""Unit tests for utils module."""
import unittest

from coloredqrcode.utils import split_and_pad_data


class TestUtils(unittest.TestCase):
    """Tests for utils.py functionality."""

    def test_split_and_pad_data(self):
        """Test data splitting and padding functionality."""
        test_cases = [
            ("abc", 3, "\0", ["a", "b", "c"]),  # Equal parts
            ("abcd", 3, "\0", ["ab", "c\0", "d\0"]),  # Uneven split (2-1-1 characters)
            ("abcde", 3, "\0", ["ab", "cd", "e\0"]),  # Uneven split (2-2-1 characters)
            ("a", 3, "*", ["a", "*", "*"]),  # Single character
            ("", 3, "\0", ["", "", ""]),  # Empty string
        ]
        for data, n_parts, pad_char, expected in test_cases:
            result = split_and_pad_data(data, n_parts, pad_char)
            self.assertEqual(
                result,
                expected,
                f"Incorrect split for data='{data}', n_parts={n_parts}",
            )


if __name__ == "__main__":
    unittest.main()
