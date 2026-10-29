"""Check that the game is actually receiving input, before blaming the game logic.

The failure this exists for: the emulator kept rendering and kept answering screencap, so
every metric looked healthy, while taps, swipes and the Back key all went nowhere. The
frame was not frozen either -- animations and the ping counter kept ticking -- it was just
never *changed* by anything sent to it.

What was ruled out, in order, because each one is the obvious suspect:

  transport   `adb shell input` succeeded and the sidecar reported 36-45 ms dispatch
  window      the emulator was not minimised; restoring and foregrounding changed nothing
  focus       `mCurrentFocus` was `com.mihoyo.combosdk.ComboSDKActivity` both while input
              was dead *and* after it recovered, so the same window can do both -- focus
              was a red herring, not the cause

What actually recovered it: force-stop plus relaunch of the package, then Start Game and
Tap to Start. Which means the honest test is a frame-change probe -- measure whether the
game *reacted* -- rather than asking any component whether it thinks it sent something.

    python tools/input_check.py            # measure, report, exit 0/1
    python tools/input_check.py --fix      # relaunch the package if input is dead
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecar_windows.adb import AdbClient, AdbConfig  # noqa: E402

ADB = r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\nx_device\12.0\shell\adb.exe"
PACKAGE = "com.HoYoverse.hkrpgoversea"
# A camera drag on the right half: it changes the frame even when the character is wedged
# against a wall, unlike a stick push, so it isolates "input works" from "movement works".
PROBE = {"type": "swipe", "x": 950, "y": 360, "x2": 1200, "y2": 360, "duration_ms": 600}
MIN_FRAME_CHANGE = 3.0


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> dict:
    import json

    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(request, timeout=15).read().decode())


def frame_change(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())


def focus(adb: AdbClient) -> str:
    result = adb.shell("dumpsys", "window")
    text = result.stdout.decode("utf-8", "replace") if isinstance(result.stdout, bytes) else result.stdout
    for line in text.splitlines():
        if "mCurrentFocus" in line:
            return line.strip()
    return "(unknown)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the client is receiving input")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--adb-path", default=ADB)
    parser.add_argument("--serial", default="127.0.0.1:16384")
    parser.add_argument("--fix", action="store_true",
                        help="force-stop and relaunch the package when input is dead")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    adb = AdbClient(AdbConfig(adb_path=args.adb_path, serial=args.serial))

    before = fetch(base)
    dispatch = act(base, PROBE)
    time.sleep(1.4)
    after = fetch(base)
    change = frame_change(before, after)

    print(f"focus:    {focus(adb)}")
    print(f"dispatch: {dispatch.get('dispatch_ms')} ms, frame_seq {dispatch.get('frame_seq_at_dispatch')}")
    print(f"change:   {change:.2f} (needs > {MIN_FRAME_CHANGE})")

    if change > MIN_FRAME_CHANGE:
        print("input:    OK -- the game reacted")
        return 0

    print("input:    DEAD -- the game is rendering but not accepting presses")
    if not args.fix:
        print("          re-run with --fix to relaunch the package")
        return 1

    print("          relaunching", PACKAGE)
    adb.shell("am", "force-stop", PACKAGE)
    time.sleep(1.0)
    adb.shell("monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")
    print("          relaunched; wait ~60 s for the title, then tap Start and Tap to Start")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
