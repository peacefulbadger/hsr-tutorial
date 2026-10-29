"""Record a replay buffer from the sidecar.

Pulls frames, writes only the ones that differ from their predecessor, and keeps
a JSONL index. Offline perception work then runs against disk, never against the
emulator, which is the whole point of the replay harness in the plan.

    python tools/record.py --out data/replays/prologue --duration 180

Index line (one per stored frame):
    {"path": "frames/000123.jpg", "seq": 1234, "capture_epoch_ms": ..., "age_ms": ...,
     "width": 1280, "height": 720, "held": 7}
`held` is how many subsequent identical frames were collapsed into this one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def fetch_frame(base: str, timeout: float = 10.0):
    t0 = time.monotonic()
    with urllib.request.urlopen(f"{base}/frame", timeout=timeout) as response:
        body = response.read()
        headers = response.headers
    rtt_ms = (time.monotonic() - t0) * 1000.0
    return body, headers, rtt_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a replay buffer")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120.0, help="seconds")
    parser.add_argument("--hz", type=float, default=5.0, help="sampling rate")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"

    period = 1.0 / args.hz
    started = time.monotonic()
    stored = 0
    seen = 0
    previous_hash: str | None = None
    pending_held = 0
    first_capture_ms = None
    last_entry: dict | None = None

    with index_path.open("w", encoding="utf-8") as index:
        while time.monotonic() - started < args.duration:
            tick = time.monotonic()
            try:
                body, headers, rtt = fetch_frame(base)
            except Exception as exc:  # noqa: BLE001
                print(f"frame fetch failed: {exc}")
                time.sleep(period)
                continue

            seen += 1
            digest = hashlib.sha256(body).hexdigest()[:16]

            if not args.keep_duplicates and digest == previous_hash:
                pending_held += 1
            else:
                if last_entry is not None:
                    last_entry["held"] = pending_held
                    index.write(json.dumps(last_entry) + "\n")
                capture_ms = int(headers["X-Capture-Epoch-Ms"])
                first_capture_ms = first_capture_ms or capture_ms
                relative = f"frames/{stored:06d}.jpg"
                (args.out / relative).write_bytes(body)
                last_entry = {
                    "path": relative,
                    "seq": int(headers["X-Frame-Seq"]),
                    "capture_epoch_ms": capture_ms,
                    "t_rel_ms": capture_ms - first_capture_ms,
                    "age_ms": round(float(headers["X-Server-Age-Ms"]) + rtt, 1),
                    "width": int(headers["X-Frame-Width"]),
                    "height": int(headers["X-Frame-Height"]),
                    "bytes": len(body),
                    "held": 0,
                }
                stored += 1
                pending_held = 0
                previous_hash = digest

            if stored and stored % 50 == 0 and seen % 50 == 0:
                print(f"  {stored} stored / {seen} seen")

            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)

        if last_entry is not None:
            last_entry["held"] = pending_held
            index.write(json.dumps(last_entry) + "\n")

    elapsed = time.monotonic() - started
    print(f"recorded {stored} distinct frames (from {seen}) in {elapsed:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
