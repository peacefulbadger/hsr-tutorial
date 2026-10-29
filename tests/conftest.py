"""Shared fixtures: a fake adb client so the whole suite runs with no emulator."""

from __future__ import annotations

import struct
import threading
import zlib

import pytest

from sidecar_windows.adb import AdbResult


def make_png(width: int = 1280, height: int = 720) -> bytes:
    """Build a real, decodable PNG of the requested size (solid grey)."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    row = b"\x00" + b"\x80\x80\x80" * width
    raw = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def make_raw(width: int = 1280, height: int = 720) -> bytes:
    """Build a raw `screencap` dump: w, h, format, colorspace, then RGBA pixels."""
    header = struct.pack("<IIII", width, height, 1, 0)
    pixels = bytes([200, 120, 60, 255]) * (width * height)
    return header + pixels


class FakeAdb:
    """Duck-typed stand-in for AdbClient: records calls, hands back canned frames."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fail: bool = False,
        fail_raw: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fail = fail
        self.fail_raw = fail_raw
        self.png = make_png(width, height)
        self.raw = make_raw(width, height)
        self.calls: list[tuple[str, tuple]] = []
        self._lock = threading.Lock()
        self.adb_path = r"C:\fake\adb.exe"
        self.serial = "fake-0001"

    def screencap_png(self) -> tuple[bytes, float]:
        if self.fail:
            raise RuntimeError("screencap exploded")
        return self.png, 315.0

    def screencap_raw(self) -> tuple[bytes, float]:
        if self.fail or self.fail_raw:
            raise RuntimeError("screencap exploded")
        return self.raw, 132.0

    def _record(self, name: str, *args) -> float:
        with self._lock:
            self.calls.append((name, args))
        return 1.5

    def tap(self, x: int, y: int, duration_ms: int = 0) -> float:
        return self._record("tap", x, y, duration_ms)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> float:
        return self._record("swipe", x1, y1, x2, y2, duration_ms)

    def back(self) -> float:
        return self._record("back")

    def shell(self, *args: str) -> AdbResult:
        return AdbResult(0, b"", "", 1.0)


@pytest.fixture()
def fake_adb() -> FakeAdb:
    return FakeAdb()
