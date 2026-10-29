"""Adb wiring tests. These never touch a device: subprocess.run is faked."""

from __future__ import annotations

import pytest

import sidecar_windows.adb as adb_module
from sidecar_windows.adb import AdbClient, AdbConfig, AdbError

DEVICES_OUTPUT = (
    b"List of devices attached\n"
    b"127.0.0.1:7555\tdevice\n"
    b"127.0.0.1:16384\tdevice\n"
    b"\n"
)


class FakeProcess:
    def __init__(self, stdout: bytes = b"", returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture()
def fake_run(monkeypatch):
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        calls.append(list(argv))
        if "devices" in argv:
            return FakeProcess(DEVICES_OUTPUT)
        if "screencap" in argv:
            return FakeProcess(b"not a png")
        return FakeProcess(b"")

    monkeypatch.setattr(adb_module.subprocess, "run", _run)
    return calls


def test_serial_autodetect_prefers_mumu_port(fake_run):
    client = AdbClient(AdbConfig())
    # 16384 outranks 7555 in the candidate list even though it appears later.
    assert client.serial == "127.0.0.1:16384"


def test_explicit_serial_skips_discovery(fake_run):
    client = AdbClient(AdbConfig(serial="emulator-5554"))
    assert client.serial == "emulator-5554"
    assert not any("devices" in call for call in fake_run), "should not probe devices"


def test_no_device_raises(fake_run, monkeypatch):
    monkeypatch.setattr(
        adb_module.subprocess,
        "run",
        lambda argv, **kwargs: FakeProcess(b"List of devices attached\n\n"),
    )
    with pytest.raises(AdbError, match="no online adb device"):
        AdbClient(AdbConfig())


def test_tap_builds_expected_argv(fake_run):
    client = AdbClient(AdbConfig(serial="emulator-5554"))
    client.tap(640, 680)
    assert fake_run[-1] == [
        client.adb_path, "-s", "emulator-5554", "shell", "input", "tap", "640", "680",
    ]


def test_back_uses_keycode_4(fake_run):
    client = AdbClient(AdbConfig(serial="emulator-5554"))
    client.back()
    assert fake_run[-1][3:] == ["shell", "input", "keyevent", "4"]


def test_screencap_rejects_non_png(fake_run):
    client = AdbClient(AdbConfig(serial="emulator-5554"))
    with pytest.raises(AdbError, match="did not return a PNG"):
        client.screencap_png()


def test_failed_command_raises_with_stderr(fake_run, monkeypatch):
    monkeypatch.setattr(
        adb_module.subprocess,
        "run",
        lambda argv, **kwargs: FakeProcess(b"", returncode=1, stderr=b"device offline"),
    )
    client = AdbClient(AdbConfig(serial="emulator-5554", adb_path=""))
    with pytest.raises(AdbError, match="device offline"):
        client.screencap_raw()
