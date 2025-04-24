# ColoredQRCode

ColoredQRCode is a Python library for generating and decoding standard, colored, and intensity-modulated (IM) QR codes. It supports high data density and robust decoding, even after lossy compression (e.g., JPEG). The library is compatible with Python 3.9+ and uses only standard and widely available Python packages.

## Features

- Generate standard black-and-white QR codes
- Generate colored QR codes (RGB channels encode more data)
- Generate intensity-modulated (IM) colored QR codes (even higher data density)
- Decode all supported QR code types
- Robust to JPEG compression and re-encoding

## Installation

Install dependencies (preferably in a virtual environment):

```sh
pip install -r requirements.txt
```

## Usage

### Generate a Standard QR Code

```python
from coloredqrcode import generate_qr_code

img = generate_qr_code("Hello, QR!", output_path="qrcode.png")
img.show()  # Display the QR code
```

### Decode a Standard QR Code

```python
from coloredqrcode import decode_qr_code

data = decode_qr_code("qrcode.png")
print(data)  # Output: Hello, QR!
```

### Generate a Colored QR Code (RGB)

```python
from coloredqrcode import generate_colored_qr_code

data = "The quick brown fox jumps over the lazy dog 1234567890!@#"
img = generate_colored_qr_code(data, output_path="colored_qr.png")
img.show()
```

### Decode a Colored QR Code

```python
from coloredqrcode import decode_colored_qr_code

decoded = decode_colored_qr_code("colored_qr.png")
print(decoded)
```

### Generate an Intensity-Modulated (IM) Colored QR Code

```python
from coloredqrcode import generate_colored_qr_code_im

data = "A long string to encode with high density..."
img = generate_colored_qr_code_im(data, output_path="colored_qr_im.png")
img.show()
```

### Decode an Intensity-Modulated (IM) Colored QR Code

```python
from coloredqrcode import decode_colored_qr_code_im

decoded = decode_colored_qr_code_im("colored_qr_im.png")
print(decoded)
```

## Error Handling

If the input data is too long for the selected QR code type, a `QRCodeDataTooLongError` is raised.

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies (qrcode, Pillow, pyzbar, opencv-python)

## Testing

Run all tests with:

```sh
.venv/bin/python -m pytest
```

## License

MIT License
