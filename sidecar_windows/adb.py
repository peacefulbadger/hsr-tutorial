"""Thin, synchronous wrapper around the Android Debug Bridge binary.

Deliberately dumb: no game logic, no retries-and-heuristics, no state machine.
The sidecar's job is to turn adb invocations into a byte stream and a dispatch
call, nothing more. Everything interesting happens in the orchestrator.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# MuMuPlayerGlobal ships several adb binaries; nx_device/.../shell/adb.exe is the
# one that matches the emulator's adb server version.
DEFAULT_ADB_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\nx_device\12.0\shell\adb.exe",
    r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\nx_main\adb.exe",
    r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\shell\adb.exe",
)

# MuMu listens on 16384 by default; 7555 is the classic MuMu port and
# emulator-5554 shows up when the emulator registers itself.
DEFAULT_SERIAL_CANDIDATES: tuple[str, ...] = (
    "127.0.0.1:16384",
    "127.0.0.1:7555",
    "emulator-5554",
)

KEYCODE_BACK = 4


class AdbError(RuntimeError):
    """An adb invocation failed or the device is not usable."""


@dataclass(slots=True)
class AdbConfig:
    adb_path: str = ""
    serial: str = ""
    connect_timeout_s: float = 5.0
    command_timeout_s: float = 15.0
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class AdbResult:
    returncode: int
    stdout: bytes
    stderr: str
    duration_ms: float


class AdbClient:
    def __init__(self, config: AdbConfig | None = None) -> None:
        self.config = config or AdbConfig()
        self._adb_path = self._resolve_adb_path()
        self._serial = self._resolve_serial()

    # ---------------------------------------------------------------- discovery

    def _resolve_adb_path(self) -> str:
        configured = self.config.adb_path
        if configured:
            path = Path(configured)
            if not path.is_file():
                raise AdbError(f"configured adb_path does not exist: {path}")
            return str(path)

        for candidate in DEFAULT_ADB_CANDIDATES:
            if Path(candidate).is_file():
                return candidate

        # Fall back to PATH.
        try:
            found = subprocess.run(
                ["where", "adb"], capture_output=True, text=True, timeout=10
            )
        except OSError as exc:  # pragma: no cover - platform dependent
            raise AdbError(f"could not locate adb: {exc}") from exc
        lines = [ln.strip() for ln in found.stdout.splitlines() if ln.strip()]
        if not lines:
            raise AdbError(
                "could not locate adb.exe; pass --adb-path or set it in the config"
            )
        return lines[0]

    def _resolve_serial(self) -> str:
        if self.config.serial:
            return self.config.serial
        result = self._run(["devices"], timeout=self.config.connect_timeout_s, with_serial=False)
        text = result.stdout.decode("utf-8", "replace")
        online = [
            ln.split("\t")[0].strip()
            for ln in text.splitlines()[1:]
            if "\tdevice" in ln
        ]
        for candidate in DEFAULT_SERIAL_CANDIDATES:
            if candidate in online:
                return candidate
        if online:
            return online[0]
        raise AdbError(f"no online adb device; `adb devices` said:\n{text.strip()}")

    @property
    def adb_path(self) -> str:
        return self._adb_path

    @property
    def serial(self) -> str:
        return self._serial

    # ------------------------------------------------------------------ plumbing

    def _run(
        self,
        args: list[str],
        *,
        binary: bool = False,
        timeout: float | None = None,
        with_serial: bool = True,
    ) -> AdbResult:
        argv = [self._adb_path]
        # `self._serial` does not exist until discovery finishes; `adb devices`
        # is one of the calls used to discover it.
        if with_serial and getattr(self, "_serial", ""):
            argv += ["-s", self._serial]
        argv += args

        t0 = time.perf_counter()
        effective_timeout = timeout or self.config.command_timeout_s
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(
                f"adb timed out after {effective_timeout:g}s: {' '.join(args)}"
            ) from exc
        except OSError as exc:
            raise AdbError(f"failed to run {argv[0]}: {exc}") from exc
        duration_ms = (time.perf_counter() - t0) * 1000.0

        stderr = proc.stderr.decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            raise AdbError(
                f"adb {' '.join(args)} exited {proc.returncode}: {stderr or '<no stderr>'}"
            )
        return AdbResult(proc.returncode, proc.stdout, stderr, duration_ms)

    # --------------------------------------------------------------- operations

    def shell(self, *args: str) -> AdbResult:
        """Run an arbitrary `adb shell` command. Escape hatch, used sparingly."""
        return self._run(["shell", *args])

    def state(self) -> str:
        result = self._run(["get-state"], timeout=self.config.connect_timeout_s)
        return result.stdout.decode("utf-8", "replace").strip() or "unknown"

    def screencap_png(self) -> tuple[bytes, float]:
        """Return (png_bytes, adb_duration_ms).

        `exec-out` keeps the PNG binary-clean; shell redirection would corrupt it
        on Windows (CRLF translation).

        Note: the PNG is compressed *on the device*. On the MuMu emulator that
        costs ~315 ms per frame, which caps the capture loop at ~3 Hz. Prefer
        `screencap_raw` unless you specifically want PNG.
        """
        result = self._run(["exec-out", "screencap", "-p"])
        if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
            head = result.stdout[:32]
            raise AdbError(f"screencap did not return a PNG (head={head!r})")
        return result.stdout, result.duration_ms

    def screencap_raw(self) -> tuple[bytes, float]:
        """Return (raw_framebuffer_bytes, adb_duration_ms).

        ~132 ms on the same emulator: no on-device compression. The payload is
        w, h, format(, colorspace) followed by RGBA pixels; see
        `sidecar_windows.imaging.decode_screencap_raw`.
        """
        result = self._run(["exec-out", "screencap"])
        if len(result.stdout) < 64:
            raise AdbError(f"screencap returned {len(result.stdout)} bytes; device may be asleep")
        return result.stdout, result.duration_ms

    def tap(self, x: int, y: int, duration_ms: int = 0) -> float:
        args = ["shell", "input", "tap", str(int(x)), str(int(y))]
        return self._run(args).duration_ms

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> float:
        args = [
            "shell", "input", "swipe",
            str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
            str(int(duration_ms)),
        ]
        return self._run(args).duration_ms

    def back(self) -> float:
        return self._run(["shell", "input", "keyevent", str(KEYCODE_BACK)]).duration_ms

    def display_size(self) -> tuple[int, int]:
        """Return the *current* (rotated) display size, i.e. framebuffer pixels."""
        out = self._run(["shell", "wm", "size"]).stdout.decode("utf-8", "replace")
        for line in out.splitlines():
            if ":" in line and "x" in line:
                try:
                    dims = line.split(":", 1)[1].strip()
                    w, h = dims.split("x", 1)
                    return int(w), int(h)
                except ValueError:
                    continue
        raise AdbError(f"could not parse `wm size` output: {out!r}")

    def current_focus(self) -> str:
        out = self._run(["shell", "dumpsys", "window"]).stdout.decode("utf-8", "replace")
        for line in out.splitlines():
            if "mCurrentFocus=" in line:
                return line.split("mCurrentFocus=", 1)[1].strip()
        return ""

    def package_running(self, package: str) -> bool:
        out = self._run(["shell", "pidof", package]).stdout.decode("utf-8", "replace")
        return bool(out.strip())
