"""Framebuffer decoding / encoding.

Why this module exists: `adb exec-out screencap -p` asks the *emulator* to zlib-
compress a 1280x720 frame, which measured 315 ms per call on this host and caps
the loop at ~3 Hz. Fetching the raw RGBA framebuffer takes 132 ms, and encoding
JPEG here costs ~10 ms. Same pixels, less than half the latency.

Measured on MuMuPlayerGlobal 12.0 (SM_A528B, 1280x720), 2026-10-29:

    screencap -p (on-device PNG)   median 315.1 ms   1.23 MB
    screencap    (raw RGBA)        median 132.6 ms   3.69 MB
    local JPEG encode, q=80        ~10 ms            ~0.13 MB
"""

from __future__ import annotations

import numpy as np

import cv2

# `screencap` without -p prefixes the pixel buffer with w, h, format(, colorspace).
RAW_HEADER_SIZES = (12, 16)


class FrameDecodeError(RuntimeError):
    pass


def decode_screencap_raw(data: bytes) -> np.ndarray:
    """Decode a raw `screencap` dump into a BGR image."""
    if len(data) < 16:
        raise FrameDecodeError(f"raw framebuffer too short: {len(data)} bytes")

    width = int.from_bytes(data[0:4], "little")
    height = int.from_bytes(data[4:8], "little")
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise FrameDecodeError(f"implausible raw dimensions {width}x{height}")

    expected = width * height * 4
    header = next((h for h in RAW_HEADER_SIZES if len(data) - h == expected), None)
    if header is None:
        raise FrameDecodeError(
            f"raw framebuffer size mismatch: {len(data)} bytes for {width}x{height}"
        )

    rgba = np.frombuffer(data, dtype=np.uint8, count=expected, offset=header)
    rgba = rgba.reshape(height, width, 4)
    # RGBA -> BGR, dropping alpha.
    return np.ascontiguousarray(rgba[:, :, 2::-1])


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise FrameDecodeError("JPEG encode failed")
    return buffer.tobytes()
