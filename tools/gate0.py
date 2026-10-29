"""Gate 0: one script that resets, launches, captures, taps, and captures again.

    reset -> relaunch -> wait for the title screen -> tap through -> capture
          -> tap -> capture -> assert the pixels changed

Game specifics (which points to tap, how bright a "loaded" screen is) live in a
JSON step file, not in this script, so the same runner works for another game by
dropping in another file.

Usage:
    python tools/gate0.py --config configs/gate0.hsr.json
    python tools/gate0.py --config configs/gate0.hsr.json --no-reset   # skip relaunch

Exit code 0 = gate passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass
class Frame:
    body: bytes
    seq: int
    age_ms: float
    width: int
    height: int


class Sidecar:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/health", timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def frame(self) -> Frame:
        t0 = time.monotonic()
        with urllib.request.urlopen(f"{self.base}/frame", timeout=self.timeout_s) as r:
            body = r.read()
            headers = r.headers
        rtt_ms = (time.monotonic() - t0) * 1000.0
        return Frame(
            body=body,
            seq=int(headers["X-Frame-Seq"]),
            age_ms=float(headers["X-Server-Age-Ms"]) + rtt_ms,
            width=int(headers["X-Frame-Width"]),
            height=int(headers["X-Frame-Height"]),
        )

    def tap(self, x: int, y: int) -> dict:
        payload = json.dumps({"type": "tap", "x": x, "y": y}).encode()
        request = urllib.request.Request(
            f"{self.base}/action", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as r:
            return json.loads(r.read())


def decode(body: bytes):
    import io

    import numpy as np
    from PIL import Image

    if not (body.startswith(PNG_MAGIC) or body.startswith(JPEG_MAGIC)):
        raise ValueError("sidecar returned something that is not an image")
    with Image.open(io.BytesIO(body)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def brightness(image) -> float:
    return float(image.mean())


def mad(a, b) -> float:
    import numpy as np

    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def wait_until_loaded(sc: Sidecar, *, min_brightness: float, timeout_s: float,
                      settle_s: float, poll_s: float = 1.0) -> Frame:
    """Poll until the framebuffer is no longer black/splash, then let it settle."""
    deadline = time.monotonic() + timeout_s
    best = 0.0
    while time.monotonic() < deadline:
        frame = sc.frame()
        image = decode(frame.body)
        best = brightness(image)
        if best >= min_brightness:
            if settle_s:
                time.sleep(settle_s)
            return sc.frame()
        time.sleep(poll_s)
    raise TimeoutError(f"screen never reached brightness {min_brightness} (peak {best:.1f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 gate runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-reset", action="store_true", help="skip the relaunch step")
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase0/gate"))
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sc = Sidecar(config["sidecar_url"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    started = time.time()
    print(f"gate0: {config.get('name', args.config.stem)}  sidecar={sc.base}")

    health = sc.health()
    record("sidecar healthy", bool(health.get("ok")),
           f"device={health.get('device_serial')} hz={health.get('capture', {}).get('effective_hz')}")

    adb = None

    def get_adb():
        nonlocal adb
        if adb is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from sidecar_windows.adb import AdbClient, AdbConfig

            adb = AdbClient(AdbConfig(
                adb_path=config.get("adb_path", ""), serial=config.get("serial", ""),
            ))
        return adb

    if not args.no_reset:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from emulator_control.reset_app import EmulatorController

        controller = EmulatorController(get_adb())
        t0 = time.time()
        reset = controller.reset_app(wait_running_s=45.0)
        record("client relaunched", bool(reset["ok"]),
               f"running={reset['running_after']} in {reset['elapsed_s']}s")
        del t0

    for step in config.get("steps", []):
        kind = step["type"]
        if kind == "wait_loaded":
            frame = wait_until_loaded(
                sc,
                min_brightness=step.get("min_brightness", 20.0),
                timeout_s=step.get("timeout_s", 120.0),
                settle_s=step.get("settle_s", 0.0),
            )
            image = decode(frame.body)
            record(f"wait_loaded({step.get('label', '')})", True,
                   f"brightness={brightness(image):.1f} seq={frame.seq}")
            (args.out_dir / f"{len(results):02d}-{step.get('label', 'loaded')}.jpg").write_bytes(frame.body)
        elif kind == "tap":
            result = sc.tap(step["x"], step["y"])
            record(f"tap({step['x']},{step['y']})", bool(result.get("ok")),
                   f"dispatch={result.get('dispatch_ms')}ms")
            time.sleep(step.get("after_s", 1.0))
        elif kind == "wait_focus":
            # A relaunch can land on the emulator launcher; only trust the
            # framebuffer once the game itself owns the foreground.
            package = step["package"]
            deadline = time.monotonic() + step.get("timeout_s", 90.0)
            focus = ""
            while time.monotonic() < deadline:
                focus = get_adb().current_focus()
                if package in focus:
                    break
                time.sleep(1.0)
            record(f"focus({package.rsplit('.', 1)[-1]})", package in focus, focus or "<none>")
        elif kind == "tap_until_brightness":
            # The title screen has a drifting background, so pixel diff is a
            # false positive there. Brightness separates the screens cleanly:
            # title ~47, "Tap to Start" ~120, in-game ~97.
            threshold = float(step["min_brightness"])
            attempts = int(step.get("max_attempts", 6))
            interval = float(step.get("interval_s", 7.0))
            reached = False
            level = 0.0
            for attempt in range(1, attempts + 1):
                result = sc.tap(step["x"], step["y"])
                print(f"    tap({step['x']},{step['y']}) attempt {attempt}"
                      f" dispatch={result.get('dispatch_ms')}ms")
                time.sleep(interval)
                level = brightness(decode(sc.frame().body))
                if level >= threshold:
                    reached = True
                    record(f"tap_until_brightness({step.get('label', '')})", True,
                           f"brightness={level:.1f} after {attempt} tap(s)")
                    break
            if not reached:
                record(f"tap_until_brightness({step.get('label', '')})", False,
                       f"stuck at brightness={level:.1f} after {attempts} taps")
        elif kind == "tap_until_change":
            # A single tap can be swallowed by a fade-in or a loading overlay.
            # Tap, watch, and retry until the pixels actually move.
            before = decode(sc.frame().body)
            attempts = int(step.get("max_attempts", 5))
            interval = float(step.get("interval_s", 8.0))
            min_mad = float(step.get("min_mad", 2.0))
            changed = False
            for attempt in range(1, attempts + 1):
                result = sc.tap(step["x"], step["y"])
                print(f"    tap({step['x']},{step['y']}) attempt {attempt}"
                      f" dispatch={result.get('dispatch_ms')}ms")
                time.sleep(interval)
                after = decode(sc.frame().body)
                delta = mad(before, after)
                if delta >= min_mad:
                    changed = True
                    record(f"tap_until_change({step.get('label', '')})", True,
                           f"changed after {attempt} tap(s), mad={delta:.2f}")
                    break
                before = after
            if not changed:
                record(f"tap_until_change({step.get('label', '')})", False,
                       f"no change after {attempts} taps (mad={delta:.2f})")
        elif kind == "capture":
            frame = sc.frame()
            image = decode(frame.body)
            path = args.out_dir / f"{step.get('label', 'frame')}.jpg"
            path.write_bytes(frame.body)
            record(f"capture({step.get('label', '')})", frame.age_ms < step.get("max_age_ms", 1000.0),
                   f"seq={frame.seq} age<={frame.age_ms:.0f}ms brightness={brightness(image):.1f}")
        elif kind == "assert_change":
            before = decode((args.out_dir / step["before"]).read_bytes()
                            if (args.out_dir / step["before"]).exists() else b"")
            after = decode((args.out_dir / step["after"]).read_bytes())
            delta = mad(before, after) if before.size else 0.0
            record("screen changed", delta >= step.get("min_mad", 1.0),
                   f"mean abs diff={delta:.2f} (threshold {step.get('min_mad', 1.0)})")
        else:
            raise SystemExit(f"unknown step type: {kind}")

    failed = [name for name, ok, _ in results if not ok]
    elapsed = time.time() - started
    print(f"\ngate0: {len(results) - len(failed)}/{len(results)} checks passed in {elapsed:.1f}s")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    print("GATE 0: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
