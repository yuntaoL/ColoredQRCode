"""
Custom exceptions for the coloredqrcode package.
"""

class QRCodeDataTooLongError(ValueError):
    """Exception raised when the input data exceeds the maximum supported QR code length."""
    pass
