"""Sidecar contract tests. No emulator, no adb, no network."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sidecar_windows.capture import CaptureLoop, png_size
from sidecar_windows.imaging import FrameDecodeError, decode_screencap_raw, encode_jpeg
from sidecar_windows.server import VERSION, create_app
from tests.conftest import FakeAdb, make_png, make_raw

JPEG_MAGIC = b"\xff\xd8\xff"


# --------------------------------------------------------------------- imaging


def test_png_size_reads_ihdr():
    assert png_size(make_png(1280, 720)) == (1280, 720)


def test_png_size_rejects_non_png():
    with pytest.raises(Exception):
        png_size(b"not a png at all, definitely not")


def test_decode_raw_preserves_geometry_and_colour():
    raw = make_raw(64, 32)
    bgr = decode_screencap_raw(raw)
    assert bgr.shape == (32, 64, 3)
    # RGBA(200,120,60) -> BGR(60,120,200)
    assert tuple(int(v) for v in bgr[0, 0]) == (60, 120, 200)


def test_decode_raw_rejects_truncated_buffer():
    raw = make_raw(64, 32)
    with pytest.raises(FrameDecodeError):
        decode_screencap_raw(raw[:-1000])


def test_decode_raw_rejects_garbage_header():
    with pytest.raises(FrameDecodeError):
        decode_screencap_raw(b"\x00" * 64)


def test_encode_jpeg_roundtrip():
    bgr = decode_screencap_raw(make_raw(64, 32))
    data = encode_jpeg(bgr, quality=90)
    assert data.startswith(JPEG_MAGIC)


# --------------------------------------------------------------------- capture


def test_capture_loop_publishes_jpeg_by_default(fake_adb):
    with CaptureLoop(fake_adb, target_hz=20.0) as loop:
        frame = loop.wait_for_frame(timeout_s=2.0)
        assert frame is not None
        assert frame.width == 1280 and frame.height == 720
        assert frame.content_type == "image/jpeg"
        assert frame.data.startswith(JPEG_MAGIC)
        assert frame.capture_epoch_ms > 0
        assert frame.age_ms >= 0

        first_seq = frame.seq
        time.sleep(0.15)
        latest = loop.latest()
        assert latest is not None
        assert latest.seq > first_seq, "sequence number must advance"

        health = loop.health()
        assert health["running"] is True
        assert health["frames_captured"] >= 2
        assert health["capture_errors"] == 0
        assert health["frame_format"] == "jpeg"
        assert health["effective_hz"] > 0


def test_capture_loop_png_mode(fake_adb):
    with CaptureLoop(fake_adb, target_hz=20.0, frame_format="png") as loop:
        frame = loop.wait_for_frame(timeout_s=2.0)
        assert frame is not None
        assert frame.content_type == "image/png"
        assert frame.data.startswith(b"\x89PNG")
        assert loop.health()["frame_format"] == "png"


def test_capture_loop_rejects_bad_format(fake_adb):
    with pytest.raises(ValueError):
        CaptureLoop(fake_adb, frame_format="webp")  # type: ignore[arg-type]


def test_capture_loop_survives_adb_failures():
    broken = FakeAdb(fail=True)
    loop = CaptureLoop(broken, target_hz=20.0, failure_backoff_s=0.01)
    loop.start()
    time.sleep(0.2)
    try:
        health = loop.health()
        assert health["capture_errors"] >= 1
        assert health["last_error"] and "screencap exploded" in health["last_error"]
        assert health["running"] is True, "loop must not die on error"
        assert loop.latest() is None
    finally:
        loop.stop()


def test_capture_loop_survives_raw_decode_failure():
    broken = FakeAdb(fail_raw=True)
    loop = CaptureLoop(broken, target_hz=20.0, failure_backoff_s=0.01)
    loop.start()
    time.sleep(0.2)
    try:
        assert loop.latest() is None
        assert loop.health()["capture_errors"] >= 1
    finally:
        loop.stop()


# ---------------------------------------------------------------------- server


@pytest.fixture()
def client(fake_adb):
    app = create_app(fake_adb, target_hz=20.0)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_capture_state(client):
    body = client.get("/health").json()
    assert body["version"] == VERSION
    assert body["device_serial"] == "fake-0001"
    assert "capture" in body and body["capture"]["frames_captured"] >= 1


def test_frame_returns_image_with_timing_headers(client):
    response = client.get("/frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(JPEG_MAGIC)

    headers = response.headers
    assert headers["x-content-format"] == "image/jpeg"
    assert headers["x-frame-width"] == "1280"
    assert headers["x-frame-height"] == "720"
    capture_ms = int(headers["x-capture-epoch-ms"])
    response_ms = int(headers["x-response-epoch-ms"])
    assert response_ms >= capture_ms, "response cannot predate the capture"
    assert float(headers["x-server-age-ms"]) >= 0
    assert int(headers["x-frame-seq"]) >= 1


def test_frame_png_mode_serves_png():
    app = create_app(FakeAdb(), target_hz=20.0, frame_format="png")
    with TestClient(app) as client:
        response = client.get("/frame")
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")


def test_frame_meta_agrees_with_frame_headers(client):
    meta = client.get("/frame/meta").json()
    assert meta["width"] == 1280 and meta["height"] == 720
    assert meta["bytes"] > 0
    assert meta["server_age_ms"] >= 0


def test_action_tap_dispatches_coordinates(client, fake_adb):
    response = client.post("/action", json={"type": "tap", "x": 640, "y": 600})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["type"] == "tap"
    assert fake_adb.calls[-1] == ("tap", (640, 600, 100))


def test_action_swipe_and_back(client, fake_adb):
    assert client.post("/action", json={
        "type": "swipe", "x": 900, "y": 400, "x2": 200, "y2": 400, "duration_ms": 250,
    }).status_code == 200
    assert client.post("/action", json={"type": "back"}).status_code == 200
    assert fake_adb.calls[-2] == ("swipe", (900, 400, 200, 400, 250))
    assert fake_adb.calls[-1] == ("back", ())


def test_action_wait_does_not_touch_adb(client, fake_adb):
    before = len(fake_adb.calls)
    assert client.post("/action", json={"type": "wait", "duration_ms": 10}).status_code == 200
    assert len(fake_adb.calls) == before


@pytest.mark.parametrize("payload", [
    {"type": "tap", "y": 10},
    {"type": "tap", "x": 10},
    {"type": "swipe", "x": 1, "y": 2, "x2": 3},
    {"type": "teleport", "x": 1, "y": 2},
    {"type": "tap", "x": 10, "y": 10, "nonsense": True},
    {"type": "tap", "x": 99999, "y": 10},
])
def test_action_rejects_malformed_payloads(client, payload):
    assert client.post("/action", json=payload).status_code == 422


def test_no_frame_yet_returns_503():
    app = create_app(FakeAdb(fail=True), target_hz=20.0)
    with TestClient(app) as client:
        assert client.get("/frame").status_code == 503
        assert client.get("/frame/meta").status_code == 503
        assert client.get("/health").json()["ok"] is False
