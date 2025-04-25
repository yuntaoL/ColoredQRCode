"""
Functions for handling QR code error correction and data splitting.
"""

from typing import List
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)

def get_max_bytes(error_correction: int) -> int:
    """
    Returns the maximum number of bytes supported for a QR code (version 40, Byte mode)
    at the given error correction level.
    """
    if error_correction == ERROR_CORRECT_L:
        return 2953
    elif error_correction == ERROR_CORRECT_M:
        return 2331
    elif error_correction == ERROR_CORRECT_Q:
        return 1663
    elif error_correction == ERROR_CORRECT_H:
        return 1273
    else:
        raise ValueError("Invalid error correction level.")

def auto_select_error_correction(data: str) -> int:
    """
    Selects the highest error correction constant that can fit the data.
    Returns the error correction constant (ERROR_CORRECT_H, _Q, _M, _L).
    """
    length = len(data.encode("utf-8"))
    for level in [ERROR_CORRECT_H, ERROR_CORRECT_Q, ERROR_CORRECT_M, ERROR_CORRECT_L]:
        if length <= get_max_bytes(level):
            return level
    from .exceptions import QRCodeDataTooLongError
    raise QRCodeDataTooLongError(
        f"Input data is too long for a QR code (max {get_max_bytes(ERROR_CORRECT_L)} bytes in Byte mode, got {length} bytes)."
    )
