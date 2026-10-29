"""LAN probe: run this on the orchestrator host (10.0.0.2), not on the sidecar host.

Measures the two things that can silently ruin a closed loop:

  1. round-trip latency, and
  2. how old the pixels actually are when they arrive (frame age).

Frame age is measured without needing clock sync between the two machines:

    server_age_ms = response_epoch_ms - capture_epoch_ms   (both from the sidecar's clock)
    rtt_ms        = t_after - t_before                     (client's clock)
    age_upper_ms  = server_age_ms + rtt_ms

The bound adds the whole transport time to the age, so it is pessimistic by at
most one leg of the trip. Good enough to reject stale frames; if this bound is
already inside budget, the truth is too.

Usage:
    python tools/check_lan.py --url http://10.0.0.1:8765 --n 20
    python tools/check_lan.py --url http://10.0.0.1:8765 --tap 640 600
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _image_size(body: bytes) -> tuple[int, int] | None:
    """Validate that a body is a real image and return its size."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(body)) as image:
            return image.size
    except Exception:  # noqa: BLE001
        return None


def _get(url: str, timeout: float = 5.0) -> tuple[bytes, dict[str, str], float]:
    t_before = time.monotonic()
    request = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        headers = {k.lower(): v for k, v in response.headers.items()}
    rtt_ms = (time.monotonic() - t_before) * 1000.0
    return body, headers, rtt_ms


def _post(url: str, payload: dict, timeout: float = 10.0) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t_before = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    rtt_ms = (time.monotonic() - t_before) * 1000.0
    return json.loads(body), rtt_ms


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure LAN capture latency and frame freshness")
    parser.add_argument("--url", default="http://10.0.0.1:8765")
    parser.add_argument("--n", type=int, default=20, help="number of frames to sample")
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between samples")
    parser.add_argument("--budget-ms", type=float, default=500.0, help="max acceptable frame age")
    parser.add_argument("--tap", nargs=2, type=int, metavar=("X", "Y"), help="also test POST /action")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    print(f"probe: {base}  n={args.n}  budget={args.budget_ms:.0f}ms")

    try:
        health_body, _, _ = _get(f"{base}/health")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not reach sidecar: {exc}")
        return 1
    health = json.loads(health_body)
    print(f"health: ok={health.get('ok')} device={health.get('device_serial')} "
          f"effective_hz={health.get('capture', {}).get('effective_hz')}")

    rtts: list[float] = []
    ages: list[float] = []
    sizes: list[int] = []
    seqs: list[int] = []
    failures = 0

    for i in range(args.n):
        try:
            body, headers, rtt_ms = _get(f"{base}/frame")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [{i:02d}] request failed: {exc}")
            time.sleep(args.interval)
            continue

        if not (body.startswith(PNG_MAGIC) or body.startswith(JPEG_MAGIC)):
            failures += 1
            print(f"  [{i:02d}] response was not an image ({len(body)} bytes)")
            continue
        size = _image_size(body)
        if size is None:
            failures += 1
            print(f"  [{i:02d}] image failed to decode ({len(body)} bytes)")
            continue

        capture_ms = float(headers.get("x-capture-epoch-ms", "0"))
        response_ms = float(headers.get("x-response-epoch-ms", "0"))
        seq = int(headers.get("x-frame-seq", "-1"))
        server_age = response_ms - capture_ms
        age_upper = server_age + rtt_ms

        rtts.append(rtt_ms)
        ages.append(age_upper)
        sizes.append(len(body))
        seqs.append(seq)

        flag = "" if age_upper <= args.budget_ms else "  <-- STALE"
        print(f"  [{i:02d}] seq={seq:5d} {size[0]}x{size[1]} {len(body) / 1024:7.1f} KiB  "
              f"rtt={rtt_ms:6.1f}ms  server_age={server_age:6.0f}ms  "
              f"age<={age_upper:6.1f}ms{flag}")

        time.sleep(args.interval)

    ok = failures == 0 and bool(rtts)
    if rtts:
        unique = len(set(seqs))
        mean_bytes = statistics.fmean(sizes)
        bitrate_mbps = mean_bytes * 8 * (1.0 / max(args.interval + statistics.fmean(rtts) / 1000.0, 1e-6)) / 1e6
        print()
        print(f"frames        : {len(rtts)} ok, {failures} failed, {unique} distinct seq")
        print(f"rtt ms        : p50={_pct(rtts, 0.5):.1f}  p95={_pct(rtts, 0.95):.1f}  max={max(rtts):.1f}")
        print(f"frame age ms  : p50={_pct(ages, 0.5):.1f}  p95={_pct(ages, 0.95):.1f}  max={max(ages):.1f}")
        print(f"frame size    : mean={mean_bytes / 1024:.1f} KiB")
        print(f"pull bitrate  : ~{bitrate_mbps:.1f} Mbps at this sampling rate")
        ok = ok and _pct(ages, 0.95) <= args.budget_ms

    if args.tap:
        x, y = args.tap
        try:
            result, action_rtt = _post(f"{base}/action", {"type": "tap", "x": x, "y": y})
            print(f"\naction tap({x},{y}): ok={result.get('ok')} "
                  f"dispatch={result.get('dispatch_ms')}ms rtt={action_rtt:.1f}ms")
        except Exception as exc:  # noqa: BLE001
            print(f"\naction tap FAILED: {exc}")
            ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
